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

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/Atomic.cuh>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>

#include "Common.h"
#include "Utils.cuh"
#include "CoherentRaster_CR.cuh"

#include <torch/extension.h>

namespace gsplat {

namespace cg = cooperative_groups;


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
) {
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= n_view_group * n_gauss) return;

    uint32_t gauss_idx = (uint32_t)(idx % n_gauss);
    uint32_t view_group_idx = (uint32_t)(idx / n_gauss);
    bool first_pass = cum_tiles_per_gauss == nullptr;

    const float radius_x = radii[view_group_idx * n_gauss * 2 + gauss_idx * 2];
    const float radius_y = radii[view_group_idx * n_gauss * 2 + gauss_idx * 2 + 1];
    
    if (radius_x <= 0 || radius_y <= 0) {
        if (first_pass) tiles_per_gauss[idx] = 0;
        return;
    }

    vec2 mean2d = glm::make_vec2(means2d + uint32_t(n_view_group / 2) * n_gauss * 2 + 2 * gauss_idx);

    // Variables for Bounding Box calculation
    float min_translation_x = std::numeric_limits<float>::max();
    float max_translation_x = std::numeric_limits<float>::lowest();
    float min_translation_y = std::numeric_limits<float>::max();
    float max_translation_y = std::numeric_limits<float>::lowest();

    constexpr int MAX_VIEW_GROUP_SIZE = 32; 
    float local_trans_x[MAX_VIEW_GROUP_SIZE];
    float local_trans_y[MAX_VIEW_GROUP_SIZE];

    float adj_mean_cam[3];
    float adj_mean_image[2];

    #pragma unroll
    for (uint32_t i = 0; i < view_group_size; ++i) {
        const float* cur_mean3d = means3d + 3 * gauss_idx;
        const float* cur_adjacent_viewmat = adjacent_viewmats + view_group_size*16 * view_group_idx + 16*i;

        world_to_cam_coord(cur_mean3d, cur_adjacent_viewmat, adj_mean_cam);
        cam_to_image_coord(adj_mean_cam, K, adj_mean_image);

        float translation_x = adj_mean_image[0] - mean2d.x;
        float translation_y = adj_mean_image[1] - mean2d.y;

        if (i < MAX_VIEW_GROUP_SIZE) {
            local_trans_x[i] = translation_x;
            local_trans_y[i] = translation_y;
        }

        if (!first_pass) {
            translation_values[n_view_group*view_group_size*2 * gauss_idx + view_group_size*2 * view_group_idx + 2 * i + 0] = translation_x;
            translation_values[n_view_group*view_group_size*2 * gauss_idx + view_group_size*2 * view_group_idx + 2 * i + 1] = translation_y;
        }

        min_translation_x = fminf(min_translation_x, translation_x);
        max_translation_x = fmaxf(max_translation_x, translation_x);
        min_translation_y = fminf(min_translation_y, translation_y);
        max_translation_y = fmaxf(max_translation_y, translation_y);
    }

    float tile_radius_x = radius_x / static_cast<float>(tile_size);
    float tile_radius_y = radius_y / static_cast<float>(tile_size);
    float tile_x = mean2d.x / static_cast<float>(tile_size);
    float tile_y = mean2d.y / static_cast<float>(tile_size);

    // 1. Broad Phase: SnugBox calculation
    uint2 tile_min, tile_max;
    tile_min.x = min(max(0, (uint32_t)floor(tile_x - tile_radius_x + min_translation_x/static_cast<float>(tile_size))), tile_width);
    tile_min.y = min(max(0, (uint32_t)floor(tile_y - tile_radius_y + min_translation_y/static_cast<float>(tile_size))), tile_height);
    tile_max.x = min(max(0, (uint32_t)ceil(tile_x + tile_radius_x + max_translation_x/static_cast<float>(tile_size))), tile_width);
    tile_max.y = min(max(0, (uint32_t)ceil(tile_y + tile_radius_y + max_translation_y/static_cast<float>(tile_size))), tile_height);

    int64_t cur_idx = 0;
    int64_t depth_id_enc = 0;
    int64_t view_group_id = idx / n_gauss;
    
    if (!first_pass) {
        cur_idx = (idx == 0) ? 0 : cum_tiles_per_gauss[idx - 1];
        int32_t depth_i32 = *(int32_t *)&(depths[view_group_idx * n_gauss + gauss_idx]);
        depth_id_enc = static_cast<uint32_t>(depth_i32);
    }

    float inv_radius_x = 1.0f / radius_x;
    float inv_radius_y = 1.0f / radius_y;
    int32_t valid_tiles_count = 0;

    // ==========================================================
    // [Optimization] Axis Abstraction
    // Prevent Warp Divergence by changing only data mapping without branching
    // swap_axes = true (wider horizontally) -> Major: X, Minor: Y
    // swap_axes = false (taller vertically) -> Major: Y, Minor: X
    // ==========================================================
    bool swap_axes = (tile_max.x - tile_min.x) > (tile_max.y - tile_min.y);

    uint32_t t_min_major = swap_axes ? tile_min.x : tile_min.y;
    uint32_t t_max_major = swap_axes ? tile_max.x : tile_max.y;
    uint32_t t_min_minor = swap_axes ? tile_min.y : tile_min.x;
    uint32_t t_max_minor = swap_axes ? tile_max.y : tile_max.x;

    float radius_minor   = swap_axes ? radius_y : radius_x;
    float mean_major     = swap_axes ? mean2d.x : mean2d.y;
    float mean_minor     = swap_axes ? mean2d.y : mean2d.x;
    float inv_radius_major = swap_axes ? inv_radius_x : inv_radius_y;
    
    float t_base_major = t_min_major * tile_size;

    // Integrated loop: Maximize efficiency by using the Major Axis (longer axis) as the outer loop
    for (int32_t u = t_min_major; u < t_max_major; ++u) {
        float t_upper_major = t_base_major + tile_size;

        // [AccuTile] Narrow down the valid range (Start~End) of the Minor axis at the current Major position (u)
        int32_t minor_start = t_max_minor;
        int32_t minor_end   = t_min_minor;
        bool has_intersection = false;

        for (int k = 0; k < view_group_size; ++k) {
            // Processed with Conditional Move (CMOV) instructions, no branching
            float trans_major = swap_axes ? local_trans_x[k] : local_trans_y[k];
            float trans_minor = swap_axes ? local_trans_y[k] : local_trans_x[k];

            float c_major = mean_major + trans_major;
            
            // Major axis distance check
            float closest_major = fmaxf(t_base_major, fminf(c_major, t_upper_major));
            float dist_major = (closest_major - c_major) * inv_radius_major;
            float dist_sq_major = dist_major * dist_major;

            if (dist_sq_major > 1.0f) continue;

            // Minor axis Span calculation
            float span_ratio = sqrtf(1.0f - dist_sq_major);
            float half_width_minor = span_ratio * radius_minor;
            float c_minor = mean_minor + trans_minor;

            int32_t curr_min = static_cast<int32_t>(floor((c_minor - half_width_minor) / tile_size));
            int32_t curr_max = static_cast<int32_t>(ceil((c_minor + half_width_minor) / tile_size));

            minor_start = min(minor_start, curr_min);
            minor_end   = max(minor_end, curr_max);
            has_intersection = true;
        }

        if (!has_intersection) {
            t_base_major += tile_size;
            continue;
        }

        minor_start = max(minor_start, (int32_t)t_min_minor);
        minor_end   = min(minor_end, (int32_t)t_max_minor);

        float t_base_minor = minor_start * tile_size;

        // Inner Loop: Minor Axis (iterate through the narrowed range)
        for (int32_t v = minor_start; v < minor_end; ++v) {
            bool intersects = false;
            float t_upper_minor = t_base_minor + tile_size;

            // [Narrow Phase] Restore to actual physical coordinates for accurate intersection test
            float t_min_x_phys = swap_axes ? t_base_major : t_base_minor;
            float t_min_y_phys = swap_axes ? t_base_minor : t_base_major;
            float t_max_x_phys = swap_axes ? t_upper_major : t_upper_minor;
            float t_max_y_phys = swap_axes ? t_upper_minor : t_upper_major;

            int check_indices[] = {0, (int)view_group_size - 1, (int)view_group_size / 2};
            #pragma unroll
            for (int idx : check_indices) {
                if (idx < view_group_size) {
                    float cx = mean2d.x + local_trans_x[idx];
                    float cy = mean2d.y + local_trans_y[idx];

                    float px = fmaxf(t_min_x_phys, fminf(cx, t_max_x_phys));
                    float py = fmaxf(t_min_y_phys, fminf(cy, t_max_y_phys));

                    float dx = (px - cx) * inv_radius_x;
                    float dy = (py - cy) * inv_radius_y;

                    if (dx * dx + dy * dy <= 1.0f) {
                        intersects = true; break;
                    }
                }
            }

            if (!intersects) {
                for (int k = 1; k < view_group_size - 1; ++k) {
                    if (k == view_group_size / 2) continue;
                    float cx = mean2d.x + local_trans_x[k];
                    float cy = mean2d.y + local_trans_y[k];

                    float px = fmaxf(t_min_x_phys, fminf(cx, t_max_x_phys));
                    float py = fmaxf(t_min_y_phys, fminf(cy, t_max_y_phys));

                    float dx = (px - cx) * inv_radius_x;
                    float dy = (py - cy) * inv_radius_y;

                    if (dx * dx + dy * dy <= 1.0f) {
                        intersects = true; break;
                    }
                }
            }

            if (intersects) {
                if (first_pass) {
                    valid_tiles_count++;
                } else {
                    // Store results: Restore index to the physical coordinate system
                    int32_t final_x = swap_axes ? u : v;
                    int32_t final_y = swap_axes ? v : u;

                    int64_t tile_id = final_y * tile_width + final_x;
                    int64_t tile_id_enc = tile_id << (32 + view_group_n_bits);
                    isect_ids[cur_idx] = tile_id_enc | (view_group_id << 32) | depth_id_enc;
                    flatten_ids[cur_idx] = static_cast<int32_t>(gauss_idx);
                    ++cur_idx;
                }
            }
            t_base_minor += tile_size;
        }
        t_base_major += tile_size;
    }

    if (first_pass) {
        tiles_per_gauss[idx] = valid_tiles_count;
    }
}


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
) {
    // constant
    const uint32_t n_view_group = conics.size(1);
    const uint32_t view_group_size = translation_values.size(2);
    const uint32_t n_gauss = conics.size(0);
    const uint32_t n_isects = flatten_ids.size(0);
    const uint32_t n_tile_height = tile_offsets.size(0);
    const uint32_t n_tile_width = tile_offsets.size(1);
    const uint32_t tile_size = view_idx_matrix.size(3);

    // indices
    auto block = cg::this_thread_block();

    int32_t tile_id = block.group_index().y * n_tile_width + block.group_index().x;

    uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
    uint32_t j = block.group_index().x * tile_size + block.thread_index().x;
    uint32_t k = block.group_index().z * 3 + block.thread_index().z;

    uint32_t original_x = subpixel_coord_matrix[block.group_index().y][block.group_index().x][block.thread_index().z][block.thread_index().y][block.thread_index().x][2];
    uint32_t original_y = subpixel_coord_matrix[block.group_index().y][block.group_index().x][block.thread_index().z][block.thread_index().y][block.thread_index().x][1];
    uint32_t original_z = subpixel_coord_matrix[block.group_index().y][block.group_index().x][block.thread_index().z][block.thread_index().y][block.thread_index().x][0];
    
    float px = (float)original_x + 0.5f;
    float py = (float)original_y + 0.5f;
    
    // intersection range
    uint32_t view_number = view_idx_matrix[block.group_index().y][block.group_index().x][block.thread_index().z][block.thread_index().y][block.thread_index().x];
    uint32_t view_group_idx = uint32_t(view_number / view_group_size);
    uint32_t view_idx_in_group = view_number % view_group_size;
    
    int32_t range_start = tile_offsets[block.group_index().y][block.group_index().x][view_group_idx];
    int32_t range_end =
        (view_group_idx == n_view_group - 1) && (block.group_index().y == n_tile_height - 1) && (block.group_index().x == n_tile_width - 1)
        ? n_isects
        : tile_offsets.data()[block.group_index().y * (n_tile_width * n_view_group) + block.group_index().x * n_view_group + view_group_idx + 1];
    
    // iterate intersection range
    float T = 1.0f;  // current transmission left to render
    float next_T = 0.f;
    float pix_out = 0.f;  // rendered subpixel value

    for (uint32_t intersection_idx = range_start; intersection_idx < range_end; ++intersection_idx) {
        int32_t g = flatten_ids[intersection_idx];

        const float2 xy = make_float2(means2d[g][uint32_t(n_view_group / 2)][0], means2d[g][uint32_t(n_view_group / 2)][1]);
        const float opac = opacities[g];
        const float4 conic = make_float4(conics[g][view_group_idx][0], conics[g][view_group_idx][1], conics[g][view_group_idx][2], 0);

        const float2 translation_value = make_float2(
            translation_values[g][view_group_idx][view_idx_in_group][0],
            translation_values[g][view_group_idx][view_idx_in_group][1]
        );

        const vec2 delta = {(xy.x + translation_value.x - px), (xy.y + translation_value.y - py)};
        const float sigma = 0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) + conic.y * delta.x * delta.y;
        float alpha = min(0.999f, opac * __expf(-sigma));
        if (sigma < 0.f || alpha < ALPHA_THRESHOLD) {
            continue;
        }

        next_T = T * (1.0f - alpha);
        if (next_T <= 1e-4f) { // this subpixel is done: exclusive
            continue;
        }

        const float vis = alpha * T;
        pix_out += colors[g][view_group_idx][original_z] * vis;
        T = next_T;
    }
    
    // save subpixel value
    render_colors[block.group_index().y][block.group_index().x][original_z % 3][original_y % tile_size][original_x % tile_size] =
        (backgrounds == nullptr) ? pix_out : (pix_out + T * backgrounds[original_z]);
}


} // namespace gsplat
