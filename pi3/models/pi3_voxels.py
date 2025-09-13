import torch
import torch.nn as nn
from functools import partial
from copy import deepcopy
import math

from .dinov2.layers import Mlp
from ..utils.geometry import homogenize_points, ref_points_generator
from ..utils.camera import affine_transform, reproject, scale_intrinsics
from .layers.pos_embed import RoPE2D, PositionGetter
from .layers.block import BlockRope
from .layers.attention import FlashAttentionRope
from .layers.transformer_head import TransformerDecoder, LinearPts3d
from .layers.camera_head import CameraHead
from .dinov2.hub.backbones import dinov2_vitl14, dinov2_vitl14_reg, dinov2_vits14_reg, dinov2_vitb14_reg
from huggingface_hub import PyTorchModelHubMixin

class Pi3Voxels(nn.Module, PyTorchModelHubMixin):
    def __init__(
            self,
            pos_type='rope100',
            encoder_size='small',
            decoder_size='small',
        ):
        super().__init__()

        # ----------------------
        #        Encoder
        # ----------------------
        if encoder_size == 'small':
            self.encoder = dinov2_vits14_reg(pretrained=False)
        elif encoder_size == 'base':
            self.encoder = dinov2_vitb14_reg(pretrained=False)
        elif encoder_size == 'large':
            self.encoder = dinov2_vitl14(pretrained=False)
        else:
            raise NotImplementedError
        
        self.patch_size = 14
        del self.encoder.mask_token

        # ----------------------
        #  Positonal Encoding
        # ----------------------
        self.pos_type = pos_type if pos_type is not None else 'none'
        self.rope=None
        if self.pos_type.startswith('rope'): # eg rope100 
            if RoPE2D is None: raise ImportError("Cannot find cuRoPE2D, please install it following the README instructions")
            freq = float(self.pos_type[len('rope'):])
            self.rope = RoPE2D(freq=freq)
            self.position_getter = PositionGetter()
        else:
            raise NotImplementedError

        # ----------------------
        #        Decoder
        # ----------------------
        enc_embed_dim = self.encoder.blocks[0].attn.qkv.in_features        # 1024
        if decoder_size == 'small':
            dec_embed_dim = 384
            dec_num_heads = 6
            mlp_ratio = 4
            dec_depth = 24
        elif decoder_size == 'base':
            dec_embed_dim = 768
            dec_num_heads = 12
            mlp_ratio = 4
            dec_depth = 24
        elif decoder_size == 'large':
            dec_embed_dim = 1024
            dec_num_heads = 16
            mlp_ratio = 4
            dec_depth = 36
        else:
            raise NotImplementedError
        self.decoder = nn.ModuleList([
            BlockRope(
                dim=dec_embed_dim,
                num_heads=dec_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                drop_path=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=0.01,
                qk_norm=True,
                attn_class=FlashAttentionRope,
                rope=self.rope
            ) for _ in range(dec_depth)])
        self.dec_embed_dim = dec_embed_dim

        # ----------------------
        #     Register_token
        # ----------------------
        num_register_tokens = 5
        self.patch_start_idx = num_register_tokens
        self.register_token = nn.Parameter(torch.randn(1, 1, num_register_tokens, self.dec_embed_dim))
        nn.init.normal_(self.register_token, std=1e-6)

        # ----------------------
        # Voxel Position Encoder
        # ----------------------
        self.roi = [-9, 9, -3, 3, 0, 30] 
        self.query_shape = (6, 2, 10) 
        self.xyz = math.prod(self.query_shape) 
        self.voxel_size = 3 
        self.voxel_position_encoder = VoxelPositionEncoder(enc_embed_dim)
        self.ooi_embedding = nn.Parameter(torch.zeros((1, enc_embed_dim, 1, 1)), requires_grad=True)

        # For ImageNet Normalize
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        self.register_buffer("image_mean", image_mean)
        self.register_buffer("image_std", image_std)

    def voxel_encoder(self, hidden, *, cam_intrinsics, cam_extrinsics, original_spatial_dims):
        """
        Args: 
            hidden: [BN, HW, C] (image features from the image encoder)
            cam_intrinsics: [BN, 3, 3] (camera intrinsic matrices for all images in the batch)
            cam_extrinsics: [BN, 4, 4] (camera extrinsic matrices (w2c transformation) all images in the batch) 
            original_spatial_dims: (N, H, W) (spatial dimensions of the original image before patchification)
        Returns:
            voxel_pos_emb: [BN, xyz, C] (voxel positional embeddings with pixel-aligned features)
        
        Procedure: 
            1. reference points: [BN, xyz, 3] (voxel centroids in the world coordinate frame)
            2. Bring reference points to camera coordinate frame (using camera extrinsics) 
            3. Project reference points (now in camera coordinate frame) to image plane (using camera intrinsics) a.k.a Reprojection
            4. Use these reprojected points to index (bilinear interpolation) into the image feature map to get pixel-aligned features for each voxel centroid
        
        Note: 
            N = number of views
            xyz = number of voxel centroids 
        """
        BN, HW, C = hidden.shape # [11, 1300, 1024]
        N, H, W = original_spatial_dims # N = number of views | H, W = spatial dimensions of the image
        B = BN // N

        patch_w, patch_h = W//self.patch_size, H//self.patch_size # spatial dimensions of the image feature map

        # Positional encoding for voxel centroids
        voxel_centroids = ref_points_generator(start=[self.roi[0], self.roi[2], self.roi[4]],
                                        shape=self.query_shape,
                                        voxel_size=self.voxel_size,
                                        normalize=True).view(-1, 3).to(hidden.device) # [xyz, 3]

        voxel_centroids = voxel_centroids.unsqueeze(0).expand(B, -1, -1) # [B, xyz, 3]
        positional_encoding = self.voxel_position_encoder(voxel_centroids) # [B, xyz, C]
        
        # Pixel-aligned features for voxel centroids

        # 1. Bring voxel centroids from world coordinate frame to image plane
        coords_world = ref_points_generator(
            start=[self.roi[0], self.roi[2], self.roi[4]],
            shape=self.query_shape,
            voxel_size=self.voxel_size,
            normalize=False) # voxel centroids in the world coordinate frame 
        coords_world = coords_world.view(-1, 3).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1).to(hidden.device) # [B, N, xyz, 3]

        # 1.1 World (3D) to Camera coordinates (3D)
        coords_cam = affine_transform(coords_world, cam_extrinsics) # [B, N, xyz, 3]
        # 1.2 Camera (3D) to Image (2D) coordinates
        coords_img, valid_mask = reproject(coords_cam, cam_intrinsics, (patch_w, patch_h)) # [B, N, xyz, 2], [B, N, xyz]

        # 2. Get pixel-aligned features for all reprojected voxel centroids for every image in each batch
        # This is where the voxel queries become image view specific
        
        # bilinear interpolation (grid sampling) requires normalized coordinates in the range [-1, 1] 
        
        # 2.1 normalize the target coordinages
        img_grid = torch.stack([2 * coords_img[..., 0] / (W - 1) - 1,
                                2 * coords_img[..., 1] / (H - 1) - 1], dim=-1) # [B, N, xyz, 2], values in [-1, 1]
        B, N, xyz, _ = img_grid.shape
        img_grid = img_grid.view(B*N, 1, xyz, 2) # [BN, 1, xyz, 2]

        # 2.2 sample pixel aligned features using bilinear interpolation
        # sampling the 'hidden' at 120 query points per image in all batches combined, where each query point is
        # a voxel centroid in the image plane, given in normalized coordinates.
        hidden = hidden.permute(0, 2, 1).contiguous().view(B*N, C, patch_h, patch_w) # [BN, C, ph, pw]
        pixel_aligned_features = nn.functional.grid_sample(hidden, img_grid, padding_mode='zeros', align_corners=True) # [BN, C, 1, xyz]

        # 2.3 Handle out of image points with OOI embedding
        # Since grid sampling samples features irrespective of whether the query point is inside or outside the image,
        # we use a valid mask to filter out the 'out of image points' and replace their features with the same learnable out-of-image embedding
        valid_mask = valid_mask.view(B*N, 1, 1, xyz).float()
        pixel_aligned_features = pixel_aligned_features * valid_mask + self.ooi_embedding * (1 - valid_mask) # [BN, C, 1, xyz]
        pixel_aligned_features = pixel_aligned_features.view(B, N, C, xyz).permute(0, 1, 3, 2).contiguous() # [B, N, xyz, C]

        # average the pixel aligned features so that there is one set of voxel queries for each batch element
        pixel_aligned_features = pixel_aligned_features.mean(dim=1) # [B, xyz, C]

        # 3. Combine positional encoding and pixel-aligned features to get voxel positional embeddings
        voxel_pos_emb = positional_encoding + pixel_aligned_features # [B, xyz, C]
        return voxel_pos_emb.view(B, xyz, C) # [B, xyz, C]

    def decode(self, hidden, N, H, W, *, camera_intrinsics, camera_extrinsics):

        BN, hw, _ = hidden.shape # [11, 1300, 1024]
        B = BN // N

        # get pixel-aligned voxel positional embeddings
        scale = (W//self.patch_size / W, H//self.patch_size / H)
        scaled_cam_intrinsics = scale_intrinsics(camera_intrinsics, scale) # [BN, 3, 3]
        pixel_aligned_voxel_feats = self.voxel_encoder(hidden, 
                                        cam_intrinsics=scaled_cam_intrinsics, 
                                        cam_extrinsics=camera_extrinsics, 
                                        original_spatial_dims=(N, H, W)) # [B, xyz, C]
        xyz = pixel_aligned_voxel_feats.shape[1]

        final_output = []
        
        hidden = hidden.reshape(B*N, hw, -1)

        register_token = self.register_token.repeat(B, N, 1, 1).reshape(B*N, *self.register_token.shape[-2:])

        # Concatenate special tokens with patch tokens
        hidden = torch.cat([register_token, hidden], dim=1)
        hw = hidden.shape[1]

        if self.pos_type.startswith('rope'):
            pos = self.position_getter(B * N, H//self.patch_size, W//self.patch_size, hidden.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * N, self.patch_start_idx, 2).to(hidden.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)
        

        # prepare hidden such that it contains image features first and then followed by all the voxel features
        pixel_aligned_voxel_feats = pixel_aligned_voxel_feats.reshape(B, xyz, -1) # [B, xyz, C]
        hidden = hidden.reshape(B, N*hw, -1) # [B, N*hw, C]
        hidden = torch.cat([hidden, pixel_aligned_voxel_feats], dim=1) # [B, N*hw, C] + [B, xyz, C] -> [B, N*hw | xyz, C]
        
        for i in range(len(self.decoder)):
            blk = self.decoder[i]
            
            # Prepare input according to Alternating Attention mechanism, as described in VGGT paper
            if i % 2 == 0:
                # sequence dimension is hw -> frame-wise/local self attention
                pos = pos.reshape(B*N, hw, -1)
                # concat x and voxel_feats along sequence dimension
                pixel_aligned_voxel_feats = hidden[:, N*hw:, :].contiguous().unsqueeze(1).expand(B, N, self.xyz, -1).reshape(B*N, self.xyz, -1) # [B*N, xyz, C]
                hidden = hidden[:, :N*hw, :].contiguous().view(B*N, hw, -1) # [BN, hw, C]
                hidden = torch.cat([hidden, pixel_aligned_voxel_feats], dim=1) # [BN, hw, C] + [BN, xyz, C] -> [BN, hw | xyz, C]
                hidden = blk(hidden, Np=hw, xpos=pos)
            else:
                # sequence dimension is N*hw -> global self attention -> attends to tokens across all frames jointly
                pos = pos.reshape(B, N*hw, -1)
                # hidden -> [BN, hw | xyz, C]
                pixel_aligned_voxel_feats = hidden[:, hw:, :].contiguous().view(B, N, xyz, -1).mean(dim=1) # [B, N, xyz, C] -> [B, xyz, C]
                hidden = hidden[:, :hw, :].contiguous().view(B, N*hw, -1) # [B, N*hw, C]
                hidden = torch.cat([hidden, pixel_aligned_voxel_feats], dim=1) # [B, N*hw, C] + [B, xyz, C] -> [B, N*hw | xyz, C]
                hidden = blk(hidden, Np=N*hw, xpos=pos)

            if i+1 in [len(self.decoder)-1, len(self.decoder)]:
                # [B, N*hw | xyz, C] and [BN, hw | xyz, C]
                if hidden.shape[0] == B*N:
                    pixel_aligned_voxel_feats = hidden[:, hw:, :].contiguous().view(B, N, self.xyz, -1).mean(dim=1) # [B, N, xyz, C] -> [B, xyz, C]
                    hidden_distilled = torch.cat([hidden[:, :hw, :].contiguous().view(B, N*hw, -1), pixel_aligned_voxel_feats], dim=1) # [B, N*hw | xyz, C]
                    final_output.append(hidden_distilled)
                else:
                    final_output.append(hidden)

        return torch.cat([final_output[0], final_output[1]], dim=-1), pos.reshape(B*N, hw, -1)
    
    def forward(self, imgs, camera_intrinsics, camera_extrinsics):

        imgs = (imgs - self.image_mean) / self.image_std

        B, N, C, H, W = imgs.shape
        
        # encode by dinov2
        imgs = imgs.reshape(B*N, C, H, W) # [11, 3, 364, 700]
        hidden = self.encoder(imgs, is_training=True)

        if isinstance(hidden, dict):
            hidden = hidden["x_norm_patchtokens"] # [11, 1300, 1024]

        # Now, perform decoding
        hidden, _ = self.decode(hidden, N, H, W, 
                                  camera_intrinsics=camera_intrinsics,
                                  camera_extrinsics=camera_extrinsics) # [1, -1, 2048] (N*hw + 120) (2048 -> outputs of last two decoder layers concatenated)

        # remove register tokens and patch tokens
        hidden = hidden[:, -self.xyz:, :].permute(0, 2, 1).contiguous() # [B, xyz, 2C] -> [B, 2C, xyz]
        hidden = hidden.view(B, -1, self.query_shape[0], self.query_shape[1], self.query_shape[2]) # [B, 2C, x, y, z]

        return hidden # [B, 2C, x, y, z]

class VoxelPositionEncoder(nn.Module):
    """
    This is the MLP that will be used to encode the position of the voxel centroids, will first be encoded using Fourier Positional Encoding)
    """
    def __init__(self, in_dim):
        super(VoxelPositionEncoder, self).__init__()
        self.position_encoder = torch.nn.Sequential(
            torch.nn.Linear(in_dim * 3, in_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(in_dim, in_dim),
        )
        self.in_dim = in_dim

    def pos2posemb3d(self, pos, num_pos_feats=128, temperature=10000):
        """
        Performs Fourier Positional encoding for 3D voxel centroids. This is the first step for creating voxel features.
        """
        # https://github.com/megvii-research/PETR/blob/main/projects/mmdet3d_plugin/models/dense_heads/petr_head.py#L29
        scale = 2 * math.pi
        pos = pos * scale
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        pos_x = pos[..., 0, None] / dim_t
        pos_y = pos[..., 1, None] / dim_t
        pos_z = pos[..., 2, None] / dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_z = torch.stack((pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()), dim=-1).flatten(-2)
        posemb = torch.cat((pos_y, pos_x, pos_z), dim=-1)
        return posemb

    def forward(self, reference_points):
        return self.position_encoder(self.pos2posemb3d(reference_points, self.in_dim))