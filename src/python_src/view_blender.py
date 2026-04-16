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
import glob
import subprocess
import sys

# ==========================================
# 1. Variable Configuration
# ==========================================

# Dataset
SCENE_DIR = "../../datasets/nerf_synthetic"
SCENE = "chair" # Set a single scene to render ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"]

DATASET_CATEGORY = "blender"
TRAIN_RESULT_DIR = "train_results/blender"
RENDER_RESULT_DIR = "render_results/blender"

VIEWPOINT_INDEX_MATRIX_PATH = "../../datasets/viewpoint_index_matrix_4K.npy"

# Orbiting Camera Synthesize
USING_DATA_IDX = "0"
ORBIT_DIRECTION = "-1"
CROP_TO_FILL = "0"
VIEW_DEGREE = "53"

# CoherentRaster hyperparameters
CLUSTER_SIZE = "8"
USE_REMAPPING = "1"

# Set CUDA_VISIBLE_DEVICES=0 environment variable
env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = "0"

# ==========================================
# 2. Render Execution
# ==========================================
data_factor = "1"

print(f"\nRunning {SCENE}...")

ckpt_pattern = os.path.join(TRAIN_RESULT_DIR, SCENE, "ckpts", "*29999*.pt")
ckpt_files = glob.glob(ckpt_pattern)

if not ckpt_files:
    print(f"  -> [Warning] Checkpoint not found for {SCENE}: {ckpt_pattern}")
else:
    for ckpt in ckpt_files:
        cmd = [
            sys.executable, "renderer.py", "default",
            "--data_factor", data_factor,
            "--data_dir", os.path.join(SCENE_DIR, SCENE),
            "--result_dir", os.path.join(RENDER_RESULT_DIR, SCENE),
            "--cluster_size", CLUSTER_SIZE,
            "--use_remapping", USE_REMAPPING,
            "--viewpoint_index_matrix_path", VIEWPOINT_INDEX_MATRIX_PATH,
            "--view_degree", VIEW_DEGREE,
            "--orbit_direction", ORBIT_DIRECTION,
            "--dataset_category", DATASET_CATEGORY,
            "--crop_to_fill", CROP_TO_FILL,
            "--using_data_idx", USING_DATA_IDX,
            "--ckpt", ckpt
        ]

        subprocess.run(cmd, env=env)