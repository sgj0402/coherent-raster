/* Copyright (c) 2026 POSTECH (Pohang University of Science and Technology) and ETRI (Electronics and Telecommunications Research Institute)
*
* This software is licensed under the MIT License.
* See the LICENSE file for details.
*
* ------------------------------
* PATENT NOTICE
* ------------------------------
* This software may implement technologies that are subject to patents
* owned by POSTECH and ETRI.
*
* NO EXPRESS OR IMPLIED LICENSES TO ANY PATENT RIGHTS ARE GRANTED
* UNDER THIS LICENSE.
*
* Commercial use of this software that requires the use of
* patented technologies may require a separate patent license
* from POSTECH and ETRI.
*
* For licensing inquiries, please contact POSTECH and ETRI.
*/

#pragma once
#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/Atomic.cuh>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>

#include "Common.h"
#include "Utils.cuh"

#include <torch/extension.h>

namespace gsplat {

namespace cg = cooperative_groups;


inline __device__ void world_to_cam_coord(
    // inputs
    const float* __restrict__ point_in_world_coord,  // [3]
    const float* __restrict__ view_matrix,  // [4, 4]
    // outputs
    float* point_in_cam_coord  // [3]
) {
    // Function to transform from world coordinate system to camera coordinate system

    glm::vec3 p_w_vec = glm::vec3(
        point_in_world_coord[0], 
        point_in_world_coord[1], 
        point_in_world_coord[2]
    );

    // Note: glm uses column-major order, but the input is in row-major order
    glm::mat3 R = glm::mat3(
        view_matrix[0], view_matrix[4], view_matrix[8],  // 1st column
        view_matrix[1], view_matrix[5], view_matrix[9],  // 2nd column
        view_matrix[2], view_matrix[6], view_matrix[10]  // 3rd column
    );
    glm::vec3 t = glm::vec3(view_matrix[3], view_matrix[7], view_matrix[11]);

    glm::vec3 p_c_vec = R * p_w_vec + t;

    point_in_cam_coord[0] = p_c_vec.x;
    point_in_cam_coord[1] = p_c_vec.y;
    point_in_cam_coord[2] = p_c_vec.z;
}


inline __device__ void cam_to_image_coord(
    // inputs
    const float* __restrict__ point_in_cam_coord,  // [3]
    const float* __restrict__ K,  // [3, 3]
    // outputs
    float* point_in_image_coord // [2]
) {
    // Function to project from camera coordinates to 2D image coordinates

    // Extract camera intrinsic parameters
    float fx = K[0]; // Focal length (X-axis)
    float fy = K[4]; // Focal length (Y-axis)
    float cx = K[2]; // Principal point (X-coordinate)
    float cy = K[5]; // Principal point (Y-coordinate)

    // 3D point in camera coordinates
    float x_c = point_in_cam_coord[0];
    float y_c = point_in_cam_coord[1];
    float z_c = point_in_cam_coord[2];

    // Exception handling for when Zc is 0 is skipped for now
    // if (z_c == 0.0f) {
        
    // }

    // Project onto the normalized image plane (X_c / Z_c, Y_c / Z_c)
    float x_norm = x_c / z_c;
    float y_norm = y_c / z_c;

    // Calculate final pixel coordinates (pinhole camera model)
    glm::vec2 p_i_vec = vec2({fx * x_norm + cx, fy * y_norm + cy});

    point_in_image_coord[0] = p_i_vec.x;
    point_in_image_coord[1] = p_i_vec.y;
}


// AccuTile applied version
__global__ void intersect_tile_kernel_CR_AccuTile(
    const uint32_t n_view_group,
    const uint32_t n_gauss,
    const float *__restrict__ means3d,
    const float *__restrict__ reference_viewmat,
    const float *__restrict__ adjacent_viewmats,
    const float *__restrict__ K,
    const uint32_t view_group_size, 
    const float *__restrict__ means2d,
    const int32_t *__restrict__ radii,
    const float *__restrict__ depths,
    const int64_t *__restrict__ cum_tiles_per_gauss,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const uint32_t view_group_n_bits,
    int32_t *__restrict__ tiles_per_gauss,
    int64_t *__restrict__ isect_ids,
    int32_t *__restrict__ flatten_ids,
    float* __restrict__ translation_values
);


__global__ void rasterize_to_pixels_3dgs_fwd_kernel_CR(
    const torch::PackedTensorAccessor64<float, 3, torch::RestrictPtrTraits> means2d,  // [n_gauss, n_view_group, 2]
    const torch::PackedTensorAccessor64<float, 3, torch::RestrictPtrTraits> conics,  // [n_gauss, n_view_group, 3]
    const torch::PackedTensorAccessor64<float, 3, torch::RestrictPtrTraits> colors,  // [n_gauss, n_view_group, 3]
    const torch::PackedTensorAccessor64<float, 1, torch::RestrictPtrTraits> opacities,  // [n_gauss]
    const torch::PackedTensorAccessor64<float, 4, torch::RestrictPtrTraits> translation_values,  // [n_gauss, n_view_group, view_group_size, 2]
    const torch::PackedTensorAccessor64<uint32_t, 5, torch::RestrictPtrTraits> view_idx_matrix,  // [n_tile_height, n_tile_width, 3, tile_size, tile_size]
    const torch::PackedTensorAccessor64<uint32_t, 6, torch::RestrictPtrTraits> subpixel_coord_matrix,  // [n_tile_height, n_tile_width, 3, tile_size, tile_size, 3]
    const float *__restrict__ backgrounds, // [3]
    const torch::PackedTensorAccessor64<int32_t, 3, torch::RestrictPtrTraits> tile_offsets,  // [n_tile_height, n_tile_width, n_view_group]
    const torch::PackedTensorAccessor64<int32_t, 1, torch::RestrictPtrTraits> flatten_ids,  // [n_intersection]
    torch::PackedTensorAccessor64<float, 5, torch::RestrictPtrTraits> render_colors  // [n_tile_height, n_tile_width, 3, tile_size, tile_size]
);

} // namespace gsplat
