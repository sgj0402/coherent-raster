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
SCENE_DIR = "../../datasets/360_v2"
SCENE_LIST = ["garden", "bicycle", "stump", "bonsai", "counter", "kitchen", "room"] # Excluded scenes: treehill, flowers

DATASET_CATEGORY = "colmap"
RESULT_DIR = "train_results/MipNeRF360"

# Set CUDA_VISIBLE_DEVICES=0 environment variable dynamically
env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = "0"

# ==========================================
# 2. Train and Eval Loop
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
    result_dir_scene = os.path.join(RESULT_DIR, scene)

    # [Train without eval]
    train_cmd = [
        sys.executable, "trainer.py", "default",
        "--eval_steps", "-1",
        "--disable_viewer",
        "--data_factor", data_factor,
        "--data_dir", data_dir,
        "--result_dir", result_dir_scene,
        "--dataset_category", DATASET_CATEGORY
    ]
    subprocess.run(train_cmd, env=env)

    # [Run eval and render]
    ckpt_pattern = os.path.join(result_dir_scene, "ckpts", "*")
    
    for ckpt in glob.glob(ckpt_pattern):
        eval_cmd = [
            sys.executable, "trainer.py", "default",
            "--disable_viewer",
            "--data_factor", data_factor,
            "--data_dir", data_dir,
            "--result_dir", result_dir_scene,
            "--ckpt", ckpt,
            "--dataset_category", DATASET_CATEGORY
        ]
        subprocess.run(eval_cmd, env=env)

# ==========================================
# 3. Print Stats Loop
# ==========================================
for scene in SCENE_LIST:
    print("\n=== Eval Stats ===")
    result_dir_scene = os.path.join(RESULT_DIR, scene)
    
    val_pattern = os.path.join(result_dir_scene, "stats", "val*.json")
    for stats_file in glob.glob(val_pattern):
        print(stats_file)
        # Read and print the file content, equivalent to the 'cat' command in bash
        with open(stats_file, 'r', encoding='utf-8') as f:
            print(f.read())
        print() # Add an empty line for readability

    print("=== Train Stats ===")
    train_pattern = os.path.join(result_dir_scene, "stats", "train*_rank0.json")
    for stats_file in glob.glob(train_pattern):
        print(stats_file)
        with open(stats_file, 'r', encoding='utf-8') as f:
            print(f.read())
        print()