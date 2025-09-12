import torch

def affine_transform(coords, cam_extrinsics):
    """
    Transforms 3D coordinates from world space to camera space or vice-versa using camera extrinsics.

    Args:
        coords (torch.Tensor): A tensor of shape (B, N, xyz, 3) representing 3D coordinates in world space. (xyz = number of voxel centroids in the scene)
        cam_extrinsics (torch.Tensor): A tensor of shape (B, N, 4, 4) representing camera extrinsic matrices. (N = number of views per batch)
    Returns:
        torch.Tensor: A tensor of shape (B, N, xyz, 3) representing the transformed 3D coordinates in camera space.
    """
    B, N, xyz, _ = coords.shape
    assert cam_extrinsics.shape == (B, N, 4, 4), f"Expected cam_extrinsics shape (B, N, 4, 4), but got {cam_extrinsics.shape}"
    assert coords.shape[-1] == 3, f"Expected coords shape (..., 3), but got {coords.shape}"

    # conversion of homogeneous coordinates lets the affine transformation be expressed as a single matrix multiplication
    coords_homogeneous = torch.cat([coords, torch.ones(B, N, xyz, 1, device=coords.device)], dim=-1) # (B, N, xyz, 4)
    coords_cam_homogeneous = torch.einsum('bnij, bnkj -> bnki', cam_extrinsics, coords_homogeneous) # (B, N, xyz, 4)

    coords_cam = coords_cam_homogeneous[..., :3]
    return coords_cam

def reproject(coords_cam, cam_intrinsics, spatial_dims):
    """
    Reprojects 3D coordinates in camera space to 2D pixel coordinates using camera intrinsics.

    Args:
        coords_cam (torch.Tensor): A tensor of shape (B, N, xyz, 3) representing 3D coordinates in camera space.
        cam_intrinsics (torch.Tensor): A tensor of shape (B, N, 3, 3) representing camera intrinsic matrices.
        spatial_dims (tuple): A tuple of two integers representing the spatial dimensions (width, height) of the image.
    Returns:
        torch.Tensor: A tensor of shape (B, N, xyz, 2) representing the reprojected 2D pixel coordinates.
    """ 
    # perform reprojection
    coords_img_homogeneous = torch.einsum('bnij, bnkj -> bnki', cam_intrinsics, coords_cam) # (B, N, xyz, 3)

    # convert non-homogeneous coordinates
    coords_img = coords_img_homogeneous[..., :2] / (coords_img_homogeneous[..., 2:] + 1e-30) # (B, N, xyz, 2)

    # apply validity mask
    valid_mask = check_valid(coords_img, spatial_dims)
    return coords_img, valid_mask
    
    # valid_coords_img = torch.stack([coords_img[b][valid_mask[b].bool()] for b in range(B)], dim=0)
    # invalid_coords_img = torch.stack([coords_img[b][~valid_mask[b].bool()] for b in range(B)], dim=0)
    # return valid_coords_img, invalid_coords_img

def check_valid(coords_img, spatial_dims):
    """
    Checks if 2D pixel coordinates are within the valid image bounds.

    Args:
        coords_img (torch.Tensor): A tensor of shape (B, N, xyz, 2) representing 2D pixel coordinates. (xyz = number of voxel centroids in the scene)
        spatial_dims (tuple): A tuple of two integers representing the spatial dimensions (width, height) of the image.
    Returns:
        torch.Tensor: A boolean tensor of shape (B, N) indicating whether each coordinate is valid (within image bounds).
    """ 
    width, height = spatial_dims
    valid_x = (coords_img[..., 0] >= 0) & (coords_img[..., 0] < width)
    valid_y = (coords_img[..., 1] >= 0) & (coords_img[..., 1] < height)

    valid = valid_x & valid_y # (B, N, xyz)
    return valid

def scale_intrinsics(cam_intrinsics, scale):
    """
    Scales camera intrinsic matrices based on the provided scale factors.

    Args:
        cam_intrinsics (torch.Tensor): A tensor of shape (..., 3, 3) representing camera intrinsic matrices. (B = batch size)
        scale (tuple): A tuple of two floats representing the scaling factors for width (sx) and height (sy) respectively.
    Returns:
        torch.Tensor: A tensor of shape (B, 3, 3) representing the scaled camera intrinsic matrices.
    """

    sx, sy = scale

    # scale focal length
    fx = cam_intrinsics[:, 0, 0] / sx
    fy = cam_intrinsics[:, 1, 0] / sy

    # adjust principal point
    cx = (cam_intrinsics[:, 0, 2] + 0.5) * sx - 0.5
    cy = (cam_intrinsics[:, 1, 2] + 0.5) * sy - 0.5

    # construct scaled intrinsics matrix
    scaled_intrinsics = cam_intrinsics.clone()
    scaled_intrinsics[:, 0, 0] = fx
    scaled_intrinsics[:, 1, 1] = fy
    scaled_intrinsics[:, 0, 2] = cx
    scaled_intrinsics[:, 1, 2] = cy
    
    return scaled_intrinsics