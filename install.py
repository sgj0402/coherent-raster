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
from pathlib import Path
import subprocess
import shutil

# --- USER CONFIGURATION ---
# 1. Target Git repository path
TARGET_REPO_DIR = Path("../gsplat").resolve()
# 2. Patch file path (Set to None if not using)
PATCH_FILE_PATH = Path("./patches/coherent_raster.patch").resolve()

# 3. Files/Folders to copy
# Format: ("Source path", "Destination path")
COPY_TASKS = [
    ("./src/csrc", "../gsplat/gsplat/cuda/csrc"),
    ("./src/python_src", "../gsplat/coherent_raster")
]
# --------------------------

def apply_patch():
    if not PATCH_FILE_PATH:
        return

    print(f"\n[1] Applying Git Patch: {PATCH_FILE_PATH}")
    if not os.path.exists(PATCH_FILE_PATH):
        print(f"  [Error] Patch file not found: '{PATCH_FILE_PATH}'")
        return

    try:
        abs_patch_path = os.path.abspath(PATCH_FILE_PATH)
        subprocess.run(
            ['git', 'apply', abs_patch_path],
            cwd=TARGET_REPO_DIR,
            capture_output=True,
            text=True,
            check=True
        )
        print("  [Success] Git patch applied.")
    except subprocess.CalledProcessError as e:
        print(f"  [Error] Failed to apply patch:\n{e.stderr}")
    except FileNotFoundError:
        print("  [Error] 'git' command not found.")

def copy_files():
    if not COPY_TASKS:
        return

    print("\n[2] Copying Files...")
    for src, dest in COPY_TASKS:
        if not os.path.exists(src):
            print(f"  [Warning] Source not found '{src}'. Skipping.")
            continue

        dest_dir = os.path.dirname(dest)
        if dest_dir:
            os.makedirs(dest_dir, exist_ok=True)

        try:
            if os.path.isdir(src):
                shutil.copytree(src, dest, dirs_exist_ok=True)
                print(f"  [Copied Folder] {src} -> {dest}")
            else:
                shutil.copy2(src, dest)
                print(f"  [Copied File] {src} -> {dest}")
        except Exception as e:
            print(f"  [Error] Failed to copy '{src}': {e}")

def main():
    if not os.path.exists(TARGET_REPO_DIR):
        print(f"[Error] Target repository not found: '{TARGET_REPO_DIR}'")
        return

    apply_patch()
    copy_files()
    
    print("\nDone.")

if __name__ == "__main__":
    main()
    input("\nPress Enter to exit...")