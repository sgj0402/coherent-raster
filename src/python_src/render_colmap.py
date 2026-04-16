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

import os
import subprocess
import sys
import glob

# ==========================================
# 1. Variable Configuration
# ==========================================

# Dataset
SCENE_DIR = "../../datasets/360_v2"
SCENE_LIST = ["garden", "bicycle", "stump", "bonsai", "counter", "kitchen", "room"] # (excluded: treehill, flowers)

DATASET_CATEGORY = "colmap"
TRAIN_RESULT_DIR = "train_results/MipNeRF360_MCMC500000"
RENDER_RESULT_DIR = "render_results/MipNeRF360_MCMC500000"

VIEWPOINT_INDEX_MATRIX_PATH = "../../datasets/viewpoint_index_matrix_4K.npy" 

# Orbiting Camera Synthesize
USING_DATA_IDX = "0"
ORBIT_DIRECTION = "-1"
CROP_TO_FILL = "1"
VIEW_DEGREE = "53"

# CoherentRaster hyperparameters
CLUSTER_SIZE = "8"
USE_REMAPPING = "1"

SAVE_EACH_VIEW_IMAGE = "1"
SAVE_INTERLACED_IMAGE = "1"

# Set CUDA_VISIBLE_DEVICES=0 environment variable dynamically
env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = "0"

# ==========================================
# 2. Render Loop
# ==========================================
for scene in SCENE_LIST:
    # Set DATA_FACTOR dynamically based on the scene name
    if scene in ["bonsai", "counter", "kitchen", "room"]:
        data_factor = "2"
    else:
        data_factor = "4"
    
    print(f"Running {scene}")

    # Combine paths dynamically based on the OS (handles Windows '\' and Linux '/')
    data_dir = os.path.join(SCENE_DIR, scene)
    result_dir_scene = os.path.join(RENDER_RESULT_DIR, scene)
    
    # Match specific checkpoint (e.g., step 29999)
    ckpt_pattern = os.path.join(TRAIN_RESULT_DIR, scene, "ckpts", "*29999*.pt")
    
    # Run evaluation and render
    for ckpt in glob.glob(ckpt_pattern):
        render_cmd = [
            sys.executable, "renderer.py", "default",
            "--disable_viewer",
            "--data_factor", data_factor,
            "--data_dir", data_dir,
            "--result_dir", result_dir_scene,
            "--cluster_size", CLUSTER_SIZE,
            "--use_remapping", USE_REMAPPING,
            "--viewpoint_index_matrix_path", VIEWPOINT_INDEX_MATRIX_PATH,
            "--view_degree", VIEW_DEGREE,
            "--orbit_direction", ORBIT_DIRECTION,
            "--dataset_category", DATASET_CATEGORY,
            "--crop_to_fill", CROP_TO_FILL,
            "--using_data_idx", USING_DATA_IDX,
            "--save_each_view_image", SAVE_EACH_VIEW_IMAGE,
            "--save_interlaced_image", SAVE_INTERLACED_IMAGE,
            "--ckpt", ckpt
        ]
        subprocess.run(render_cmd, env=env)