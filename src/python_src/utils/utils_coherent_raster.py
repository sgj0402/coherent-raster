# Copyright (c) 2026 POSTECH (Pohang University of Science and Technology) and ETRI (Electronics and Telecommunications Research Institute)
#
# This software is licensed under the MIT License.
# See the LICENSE file for details.
#
# ------------------------------
# PATENT NOTICE
# ------------------------------
# This software may implement technologies that are subject to patents
# owned by POSTECH and ETRI.
#
# NO EXPRESS OR IMPLIED LICENSES TO ANY PATENT RIGHTS ARE GRANTED
# UNDER THIS LICENSE.
#
# Commercial use of this software that requires the use of
# patented technologies may require a separate patent license
# from POSTECH and ETRI.
#
# For licensing inquiries, please contact POSTECH and ETRI.

import numpy as np

import torch
import torch.nn.functional as F


def get_viewpoint_index_matrix(path, device):
    viewpoint_index_matrix_np = np.load(path) # [H, W, C]
    viewpoint_index_matrix_np = viewpoint_index_matrix_np.astype(np.int32)
    viewpoint_index_matrix = torch.tensor(viewpoint_index_matrix_np, device=device)
    viewpoint_index_matrix = viewpoint_index_matrix.permute(2, 0, 1) # [C, H, W]

    return viewpoint_index_matrix.contiguous()


def generate_subpixel_coord_matrix(viewpoint_index_matrix):
    C, H, W = viewpoint_index_matrix.shape
    cc, hh, ww = torch.meshgrid(
        torch.arange(C, device=viewpoint_index_matrix.device, dtype=torch.int64),
        torch.arange(H, device=viewpoint_index_matrix.device, dtype=torch.int64),
        torch.arange(W, device=viewpoint_index_matrix.device, dtype=torch.int64),
        indexing='ij'
    )
    subpixel_coord_matrix = torch.stack([cc, hh, ww], dim=-1)  # [C, H, W, 3]

    return subpixel_coord_matrix.contiguous()


def pad_multiple_of(matrix, tile_size, value = 0):
    # matrix: [..., H, W]
    image_height = matrix.size(-2)
    image_width = matrix.size(-1)
    pad_h = (tile_size - (image_height % tile_size)) % tile_size
    pad_w = (tile_size - (image_width % tile_size)) % tile_size
    padding = (0, pad_w, 0, pad_h)
    matrix = F.pad(matrix, padding, "constant", value)

    return matrix.contiguous()


def patchify_image_shape_matrix(matrix, tile_size):
    # matrix: [3, h, w]
    assert matrix.size(1) % tile_size == 0 and matrix.size(2) % tile_size == 0

    n_tile_height = matrix.size(1) // tile_size
    n_tile_width = matrix.size(2) // tile_size
    matrix = matrix.reshape(3, n_tile_height, tile_size, n_tile_width, tile_size)
    matrix = matrix.permute(1, 3, 0, 2, 4)
    return matrix.contiguous()  # [n_tile_height, n_tile_width, 3, tile_size, tile_size]


def unpatchify_image_shape_matrix(matrix):
    # matrix : [n_tile_height, n_tile_width, 3, tile_size, tile_size]
    matrix = matrix.permute(2, 0, 3, 1, 4)  # -> [3, n_tile_height, tile_size, n_tile_width, tile_size]
    matrix = matrix.reshape(3, matrix.size(1)*matrix.size(2), matrix.size(3)*matrix.size(4))
    return matrix.contiguous()

def unpad(matrix, original_height, original_width):  # [..., H, W]
    matrix = matrix[..., :original_height, :original_width]
    return matrix


def patchify_coordinate_matrix(matrix, tile_size):
    # matrix: [3, h, w, 3]
    matrix = matrix.permute(0, 3, 1, 2)  # -> [3, 3, h, w]
    n_tile_height = matrix.size(2) // tile_size
    n_tile_width = matrix.size(3) // tile_size
    matrix = matrix.reshape(3, 3, n_tile_height, tile_size, n_tile_width, tile_size)
    matrix = matrix.permute(2, 4, 0, 3, 5, 1)
    return matrix.contiguous()  # [n_tile_height, n_tile_width, 3, tile_size, tile_size, 3]


def remap_subpixel_coord(viewpoint_index_matrix_patchified,
                         subpixel_coord_matrix_patchified):

    n_tile_h, n_tile_w, C, tile_size = viewpoint_index_matrix_patchified.shape[:4] # [n_tile_height, n_tile_width, 3, tile_size, tile_size]
    viewpoint_index_matrix_patchified_flat = viewpoint_index_matrix_patchified.flatten(2)  # -> [n_tile_h, n_tile_w, 3*tile_size*tile_size]
    subpixel_coord_matrix_patchified_flat = subpixel_coord_matrix_patchified.flatten(2, 4)  # -> [n_tile_h, n_tile_w, 3*tile_size*tile_size, 3]

    # get flattend
    remapped_viewpoint_index_matrix_patchified_flat, sort_indices = torch.sort(viewpoint_index_matrix_patchified_flat, dim=2)  # sort_indices: [n_tile_h, n_tile_w, 3*tile_size*tile_size]

    sort_indices = sort_indices.unsqueeze(-1).expand(-1, -1, -1, 3)  # -> (n_tile_h, n_tile_w, 3*tile_size*tile_size, 3)
    remapped_subpixel_coord_matrix_patchified_flat = torch.gather(subpixel_coord_matrix_patchified_flat, dim=2, index=sort_indices)
    
    # unflatten
    viewpoint_index_matrix_patchified = remapped_viewpoint_index_matrix_patchified_flat.reshape(n_tile_h, n_tile_w, C, tile_size, tile_size)
    subpixel_coord_matrix_patchified = remapped_subpixel_coord_matrix_patchified_flat.reshape(n_tile_h, n_tile_w, C, tile_size, tile_size, 3)

    return viewpoint_index_matrix_patchified, subpixel_coord_matrix_patchified



def get_image_data_from_loader(data_loader: torch.utils.data.DataLoader, using_data_idx, device):

    for i, data in enumerate(data_loader):
        camtoworld = data["camtoworld"].to(device).squeeze() # [4, 4]
        viewmat = torch.linalg.inv(camtoworld).squeeze() # [4, 4]
        K = data["K"].to(device).squeeze() # [3, 3]
        image = (data["image"].to(device) / 255.0).squeeze() # [H, W, C]
        image = image.permute(2, 0, 1) # [C, H, W]

        if i == using_data_idx:
            break

    return viewmat.contiguous(), K.contiguous(), image.contiguous()


def get_dynamic_orbit_center(c2w: torch.Tensor, distance: float, is_opengl: bool = False) -> torch.Tensor:
    
    camera_position = c2w[:3, 3]

    if is_opengl:
        forward_vector = -c2w[:3, 2]
    else:
        forward_vector = c2w[:3, 2]

    forward_vector = torch.nn.functional.normalize(forward_vector, dim=0)

    orbit_center = camera_position + (forward_vector * distance)

    return orbit_center



def calculate_scene_center(w2c_list_tensor, opencv_convention=False):
    """
    Calculates the optimal 3D scene center where the optical axes of multiple cameras intersect.
    
    This function uses the Least Squares method to find the 3D point that minimizes 
    the sum of squared distances to all camera viewing rays.

    Args:
        w2c_list_tensor (torch.Tensor): A tensor of shape [N, 4, 4] containing the 
            world-to-camera (w2c) transformation matrices for all cameras.
        opencv_convention (bool, optional): Determines the camera coordinate system convention.
            - True: OpenCV convention (Forward direction is +Z).
            - False: OpenGL/NeRF convention (Forward direction is -Z). Default is False.

    Returns:
        torch.Tensor: A tensor of shape [3] representing the (x, y, z) coordinates 
        of the calculated scene center.
    """
    device = w2c_list_tensor.device
    dtype = w2c_list_tensor.dtype

    # 1. Convert w2c to c2w (Requires camera positions and rotations)
    c2w_list = torch.linalg.inv(w2c_list_tensor)
    
    # 2. Extract camera positions (P) and forward viewing direction vectors (V)
    P = c2w_list[:, :3, 3]  # [N, 3] Camera positions
    R = c2w_list[:, :3, :3] # [N, 3, 3] Rotation matrices
    
    if opencv_convention:
        V = R[:, :, 2]  # [N, 3] Forward vector (+Z axis)
    else:
        V = -R[:, :, 2] # [N, 3] Forward vector (-Z axis)

    # 3. Formulate the least squares problem: A * center = b
    # For a single camera i, the distance to its ray is minimized when:
    # (I - V_i * V_i^T) * (Center - P_i) = 0
    # Summing this over all N cameras yields the linear system:
    # [Sum(I - V*V^T)] * Center = Sum((I - V*V^T) * P)
    
    # Calculate (I - V*V^T) for all cameras simultaneously
    # V shape: [N, 3] -> V.unsqueeze(2): [N, 3, 1], V.unsqueeze(1): [N, 1, 3]
    # Matmul yields [N, 3, 3]
    V_Vt = torch.matmul(V.unsqueeze(2), V.unsqueeze(1))
    I = torch.eye(3, device=device, dtype=dtype).unsqueeze(0) # [1, 3, 3]
    
    # M is the projection matrix onto the plane perpendicular to the viewing direction
    M = I - V_Vt # [N, 3, 3] 

    # Left-hand side matrix A: Sum of all M matrices -> [3, 3]
    A = torch.sum(M, dim=0)

    # Right-hand side vector b: Sum of (M @ P) -> [3]
    # Reshape P to [N, 3, 1] for batched matrix multiplication -> [N, 3, 1] -> sum -> [3]
    b = torch.sum(torch.matmul(M, P.unsqueeze(2)), dim=0).squeeze()

    # 4. Solve the linear system (A * Center = b)
    try:
        # torch.linalg.solve is preferred for well-conditioned square matrices
        center = torch.linalg.solve(A, b)
    except RuntimeError:
        # Fallback to Pseudo-inverse if A is a singular matrix 
        # (e.g., highly unlikely, but can happen if all cameras are perfectly parallel)
        center = torch.matmul(torch.linalg.pinv(A), b)

    return center


def get_scene_center(data_loader, device):

    viewmat_list = []

    for data in data_loader:
        camtoworld = data["camtoworld"].to(device).squeeze() # [4, 4]
        viewmat = torch.linalg.inv(camtoworld).squeeze() # [4, 4]

        viewmat_list.append(viewmat)

    all_w2c_tensor = torch.stack(viewmat_list)

    scene_center = calculate_scene_center(all_w2c_tensor)

    print(f"Calculated Scene Center: {scene_center}")

    return scene_center


def synthesize_orbit_viewmats(n_target_view,
                              viewmat,
                              view_degree,
                              orbit_direction,
                              center_point,
                              device
                              ):
    angles = get_orbit_angles(n_target_view, view_degree, orbit_direction, device)

    viewmats = get_orbit_w2c_with_center(viewmat, angles, center_point)

    return viewmats

def get_orbit_angles(n_target_view, view_degree, orbit_direction, device):
    if orbit_direction > 0:
        start_degree = -view_degree / 2
        end_degree = view_degree / 2
    else:
        start_degree = view_degree / 2
        end_degree = -view_degree / 2

    angles = torch.linspace(start_degree, end_degree, n_target_view, device=device)
    return angles


def get_orbit_w2c_with_center(viewmat, angles, center_point):
    """
    [GPU Native Version]
    Generates new world-to-camera (w2c) matrices by orbiting the camera around a specified 
    center point and forcing the camera to strictly look at that center.

    This function first translates the camera position along an orbit defined by the 
    given angles and a reference up-vector. Then, it recalculates the camera's rotation 
    matrix using a LookAt formulation to ensure the optical axis always points towards 
    (or away from, depending on the original convention) the center point.

    Args:
        viewmat (torch.Tensor): The original 4x4 view matrix (w2c) or a batch of them.
        angles (torch.Tensor or list): Rotation angles in degrees for the orbit.
        center_point (torch.Tensor): The 3D coordinates [3] of the scene center to orbit around.

    Returns:
        torch.Tensor: A batch of new 4x4 w2c matrices [N, 4, 4] for the orbiting cameras.
    """
    # 1. Dimension cleanup and device checking
    if viewmat.ndim == 3 and viewmat.shape[0] == 1:
        viewmat = viewmat.squeeze(0)
    
    device = viewmat.device
    dtype = viewmat.dtype

    # Unify data types and devices
    if center_point.device != device:
        center_point = center_point.to(device=device, dtype=dtype)
    
    if not isinstance(angles, torch.Tensor):
        angles = torch.tensor(angles, device=device, dtype=dtype)
    elif angles.device != device:
        angles = angles.to(device=device, dtype=dtype)

    # 2. Extract basic information (w2c -> c2w)
    c2w = torch.linalg.inv(viewmat)
    t = c2w[:3, 3]  # Original translation (camera position)
    R = c2w[:3, :3] # Original rotation matrix
    
    # Convert angles to radians
    angles_rad = torch.deg2rad(angles).view(-1)
    n_views = angles_rad.shape[0]
    
    # Reference Up Vector (Serves as the axis of rotation and the vertical reference for LookAt)
    ref_up = R[:, 1] 

    # -------------------------------------------------------------------------
    # STEP 1: Calculate New Translation (Position)
    # -------------------------------------------------------------------------
    
    # Rotate the position using Rodrigues' Rotation Formula
    u_x, u_y, u_z = ref_up[0], ref_up[1], ref_up[2]
    
    # Construct the cross-product matrix K for the rotation axis (ref_up)
    K = torch.zeros((3, 3), device=device, dtype=dtype)
    K[0, 1], K[0, 2] = -u_z, u_y
    K[1, 0], K[1, 2] = u_z, -u_x
    K[2, 0], K[2, 1] = -u_y, u_x
    
    I = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)
    
    sin_thetas = torch.sin(angles_rad).view(-1, 1, 1)
    cos_thetas = torch.cos(angles_rad).view(-1, 1, 1)
    
    K_unsqueezed = K.unsqueeze(0)
    K_sq = torch.matmul(K_unsqueezed, K_unsqueezed)
    
    # Batch of rotation matrices strictly for position translation
    R_rot_batch = I + sin_thetas * K_unsqueezed + (1 - cos_thetas) * K_sq

    # Translate position by rotating around the center point
    rel_pos = (t - center_point).view(1, 3, 1)
    new_rel_pos = torch.bmm(R_rot_batch, rel_pos.expand(n_views, -1, -1))
    new_t = center_point.view(1, 3) + new_rel_pos.squeeze(-1) # New camera positions [N, 3]


    # -------------------------------------------------------------------------
    # STEP 2: [CORE] Force LookAt (Recalculate Rotation)
    # -------------------------------------------------------------------------
    # Reconstruct the camera's local axes to ensure it looks at the center_point.
    
    # 1. New Forward (Z-axis)
    # Vector from the new camera position to the center point
    vec_cam_to_center = center_point.unsqueeze(0) - new_t # [N, 3]
    vec_cam_to_center = F.normalize(vec_cam_to_center, p=2, dim=1)
    
    # Check if the original Z-axis points towards (positive) or away from (negative) the center.
    # This maintains the original coordinate convention (e.g., OpenGL vs OpenCV).
    orig_cam_to_center = F.normalize(center_point - t, p=2, dim=0)
    orig_z_axis = R[:, 2]
    z_sign = torch.sign(torch.dot(orig_cam_to_center, orig_z_axis))
    
    # Finalize Z-axis while preserving the original convention
    new_z_axis = vec_cam_to_center * z_sign # [N, 3]

    # 2. New Right (X-axis)
    # Calculate the X-axis using the reference Up vector to keep the camera level
    ref_up_batch = ref_up.unsqueeze(0).expand(n_views, -1)
    
    # Find a geometrically orthogonal vector: X = Cross(Up, Z)
    new_x_raw = torch.linalg.cross(ref_up_batch, new_z_axis)
    new_x_axis = F.normalize(new_x_raw, p=2, dim=1)
    
    # Correct X-axis direction (sign): Align with the original camera's handedness convention.
    # Since orbiting 180 degrees might flip axes, we rely on the consistency of the cross product 
    # relationship rather than a direct comparison with R[:,0].
    
    # Determine the relationship between Cross(Up, Z) and the X-axis in the original matrix
    orig_cross_up_z = torch.linalg.cross(ref_up, orig_z_axis)
    # Check if the original X-axis aligns with or opposes the Cross(Up, Z) vector
    x_sign_check = torch.sign(torch.dot(orig_cross_up_z, R[:, 0]))
    
    # Apply the appropriate sign to the new X-axis
    new_x_axis = new_x_axis * x_sign_check

    # 3. New Up (Y-axis)
    # Recalculate the Y-axis to ensure strict orthogonality with Z and X: Y = Cross(Z, X)
    new_y_axis = torch.linalg.cross(new_z_axis, new_x_axis)
    
    # Correct Y-axis sign to align with the general direction of the reference Up vector
    y_sign_check = torch.sign(torch.sum(new_y_axis * ref_up_batch, dim=1, keepdim=True))
    new_y_axis = new_y_axis * y_sign_check

    # 4. Assemble the Rotation Matrix
    # Combine the orthogonal axes: [X, Y, Z] -> Shape: [N, 3, 3]
    new_R = torch.stack([new_x_axis, new_y_axis, new_z_axis], dim=2)

    # -------------------------------------------------------------------------
    # Final Assembly
    # -------------------------------------------------------------------------
    # Construct the new c2w matrices [N, 4, 4]
    new_c2ws = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(n_views, 1, 1)
    new_c2ws[:, :3, :3] = new_R
    new_c2ws[:, :3, 3] = new_t

    # Return the inverted matrices (w2c)
    return torch.linalg.inv(new_c2ws)


def pad_viewmats(viewmats, cluster_size):

    n_target_view = viewmats.size(0)

    n_pad = calculate_camera_n_pad(cluster_size, n_target_view)

    n_process_view = n_target_view + n_pad
    assert(n_process_view % cluster_size == 0)

    n_cluster = n_process_view // cluster_size

    viewmats_padded_list = [v for v in viewmats]
    for i in range(n_pad):
        viewmats_padded_list.append(viewmats_padded_list[-1])

    viewmats_padded = torch.stack(viewmats_padded_list)

    return viewmats_padded

def calculate_camera_n_pad(cluster_size, n_target_view):
    n_pad = (cluster_size - (n_target_view % cluster_size)) % cluster_size
    return n_pad

    





def scale_K(
    K_original, 
    original_height, original_width,
    target_height, target_width,
    crop_to_fill=True
):
    """
    Scales a camera intrinsic matrix (K) to match a new target resolution.
    
    The scaling behavior handles aspect ratio differences based on the 'crop_to_fill' flag.

    Args:
        K_original (torch.Tensor): The original 3x3 camera intrinsic matrix.
        original_width (int): Width of the original image.
        original_height (int): Height of the original image.
        target_width (int): Width of the target image.
        target_height (int): Height of the target image.
        crop_to_fill (bool, optional): Determines the scaling strategy. 
            - True (Default): 'Aspect Fill' (Crop). Scales the image to completely fill 
              the target resolution. The larger scale factor is used, meaning some 
              parts of the original scene may be cropped out of the FOV.
            - False: 'Aspect Fit' (Pad). Scales the image to fit entirely within 
              the target resolution. The smaller scale factor is used, meaning the 
              original FOV is preserved and new areas (padding) may become visible.

    Returns:
        torch.Tensor: The scaled 3x3 intrinsic matrix.
    """
    if K_original.shape != (3, 3):
        raise ValueError("K_original must be a 3x3 tensor.")
    
    fx_orig, fy_orig = K_original[0, 0], K_original[1, 1]

    # Calculate scale factors for width and height
    scale_w = target_width / original_width
    scale_h = target_height / original_height
    
    # Determine the final scale factor based on the chosen strategy
    if crop_to_fill:
        # Mode 1: Aspect Fill (Crop)
        # Use the maximum scale to ensure the target resolution is fully covered.
        scale_factor = max(scale_w, scale_h)
    else:
        # Mode 2: Aspect Fit (Pad)
        # Use the minimum scale to ensure the entire original image remains visible.
        scale_factor = min(scale_w, scale_h)

    # Scale the focal lengths to maintain the correct pixel density / magnification
    fx_new = fx_orig * scale_factor
    fy_new = fy_orig * scale_factor

    # Update the principal point (optical center) to the exact center of the new canvas
    cx_new = target_width / 2.0
    cy_new = target_height / 2.0

    # Construct the new intrinsic matrix
    K_new = torch.zeros_like(K_original)
    K_new[0, 0] = fx_new
    K_new[1, 1] = fy_new
    K_new[0, 2] = cx_new
    K_new[1, 2] = cy_new
    K_new[2, 2] = 1.0

    return K_new