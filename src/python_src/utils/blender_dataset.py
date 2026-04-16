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

import json
import os
from typing import Any, Dict, List, Optional

import cv2
import imageio.v2 as imageio
import numpy as np
import torch


class BlenderParser:
    """Blender (NeRF Synthetic) Dataset Parser."""

    def __init__(
        self,
        data_dir: str,
        factor: int = 1,
        normalize: bool = False,
    ):
        self.data_dir = data_dir
        self.factor = factor
        self.normalize = normalize

        # Blender datasets have explicitly separated train/val/test splits
        self.split_indices = {"train": [], "val": [], "test": []}
        
        self.image_paths = []
        self.camtoworlds = []
        self.Ks_dict = {}
        self.params_dict = {}
        self.imsize_dict = {}
        self.camera_ids = []
        
        # Load all splits
        files = [
            ("train", "transforms_train.json"),
            ("val", "transforms_val.json"),
            ("test", "transforms_test.json")
        ]

        global_idx = 0
        for split, filename in files:
            filepath = os.path.join(data_dir, filename)
            if not os.path.exists(filepath):
                continue
                
            with open(filepath, 'r') as f:
                meta = json.load(f)

            camera_angle_x = float(meta['camera_angle_x'])
            frames = meta['frames']

            # [Modified] Sort by filename to ensure deterministic order
            # Sort alphabetically by file path using lambda x: x['file_path']
            frames = sorted(frames, key=lambda x: natural_sort_key(x["file_path"]))

            # Print progress
            print(f"[Parser] Loading {split} split from {filename}...")

            for frame in frames:
                # 1. Image Path
                fname = frame['file_path']
                
                # [Modified 2] Filter out depth or normal images
                # Skip if 'depth' or 'normal' is included in the file path
                if "depth" in fname or "normal" in fname:
                    continue


                # Handle cases with and without extensions
                if fname.endswith(".png"): 
                    image_path = os.path.join(data_dir, fname)
                else:
                    image_path = os.path.join(data_dir, fname + ".png")
                
                self.image_paths.append(image_path)

                # 2. Camera to World Matrix (Blender uses OpenGL convention)
                # Need to convert to OpenCV convention (Right, Down, Forward)
                c2w = np.array(frame['transform_matrix']).astype(np.float32)
                
                # Blender (Right, Up, Back) -> OpenCV (Right, Down, Forward)
                # Flip Y and Z axes
                c2w[0:3, 1:3] *= -1
                self.camtoworlds.append(c2w)

                # 3. Intrinsics (Assume one camera for simplicity, or per-image)
                # To be accurate, we need to read the image to check H and W. Since they are missing in the metadata, 
                # we could process this later or assume a fixed size (e.g., 800x800).
                # For accuracy, we determine K by reading the first image.
                
                # Calculate Focal Length: f = 0.5 * W / tan(0.5 * angle_x)
                # Temporarily save only the angle here and calculate later
                self.camera_ids.append(global_idx) # Treat each image as a unique camera (simplification)
                
                # Distortion parameters (Blender is perfect pinhole)
                self.params_dict[global_idx] = np.empty(0, dtype=np.float32)

                # Save split indices
                self.split_indices[split].append(global_idx)
                
                # Temporarily save for K calculation (requires image size)
                # Blender datasets usually have the same resolution for all images
                self.temp_camera_angle_x = camera_angle_x
                
                global_idx += 1

        self.camtoworlds = np.stack(self.camtoworlds, axis=0)

        # 4. Check image size and determine the K matrix
        if len(self.image_paths) > 0:
            # Read the first image to check the resolution
            sample_img = imageio.imread(self.image_paths[0])
            H, W = sample_img.shape[:2]
            
            # Apply downscale factor
            H = H // factor
            W = W // factor
            
            focal = 0.5 * W / np.tan(0.5 * self.temp_camera_angle_x)
            
            K = np.array([
                [focal, 0, W / 2],
                [0, focal, H / 2],
                [0, 0, 1]
            ], dtype=np.float32)

            for i in range(len(self.image_paths)):
                self.Ks_dict[i] = K
                self.imsize_dict[i] = (W, H)

        # 5. Scene Normalization (Optional)
        # Blender data is usually near the origin, but we can adjust the scale
        self.scene_scale = 1.0
        if normalize:
            # Calculate the distance of all camera positions for scaling
            cam_centers = self.camtoworlds[:, :3, 3]
            dist = np.linalg.norm(cam_centers, axis=1)
            self.scene_scale = np.max(dist) 
            # Normalizing between 1.0 and 1.5 is usually recommended. Here, we will divide points by scene_scale later.

        # 6. Initialize Point Cloud
        # Blender datasets lack a sparse point cloud, so we perform random initialization
        # Generate random points within the bounding box [-1.5, 1.5] (based on Blender standard scale)
        num_points = 100_000
        print(f"[Parser] Initializing {num_points} random points...")
        pts = (np.random.rand(num_points, 3) - 0.5) * 3.0 # [-1.5, 1.5]
        
        self.points = pts.astype(np.float32)
        self.points_rgb = (np.random.rand(num_points, 3) * 255).astype(np.uint8) # Random colors
        
        # Attributes for COLMAP parser compatibility
        self.image_names = [os.path.basename(p) for p in self.image_paths]

        self.transform = np.eye(4) # Scene 정규화 변환 매트릭스
        self.points_err = np.zeros(num_points, dtype=np.float32)
        
        self.point_indices = {name: np.array([], dtype=np.int32) for name in self.image_names}
        
        self.exposure_values = [None] * len(self.image_paths)
        
        self.mask_dict = {cid: None for cid in self.camera_ids}

        unique_camera_ids = sorted(set(self.camera_ids))
        self.camera_id_to_idx = {cid: idx for idx, cid in enumerate(unique_camera_ids)}
        self.camera_indices = [self.camera_id_to_idx[cid] for cid in self.camera_ids]
        
        self.num_cameras = len(unique_camera_ids)




class BlenderDataset:
    """Blender Dataset compatible with the Parser."""

    def __init__(
        self,
        parser: BlenderParser,
        split: str = "train",
        patch_size: int | None = None,
        white_background: bool = True, # Blender datasets have transparent backgrounds, requiring background compositing
        load_depths: bool = False,     # Synthetic datasets usually lack sparse depth data, so False is recommended
    ):
        self.parser = parser
        self.split = split
        self.patch_size = patch_size
        self.white_background = white_background
        self.load_depths = load_depths

        if split not in self.parser.split_indices:
            raise ValueError(f"Unknown split {split}. Available: {list(self.parser.split_indices.keys())}")
        
        self.indices = np.array(self.parser.split_indices[split])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        index = self.indices[item]
        
        # 1. Load Image
        image_path = self.parser.image_paths[index]
        image = imageio.imread(image_path) # [H, W, 4] uint8
        
        # Handle Resize
        H, W = self.parser.imsize_dict[index]
        if image.shape[0] != H or image.shape[1] != W:
            image = cv2.resize(image, (W, H), interpolation=cv2.INTER_AREA)

        # 2. Handle Alpha Channel & Background
        # Since the training code performs "/ 255.0" later, we must maintain the 0~255 (uint8) range here.
        if image.shape[-1] == 4:
            # Convert to float for blending calculations
            image_f = image.astype(np.float32) / 255.0
            rgb = image_f[..., :3]
            alpha = image_f[..., 3:4]
            
            bg_color = np.array([1., 1., 1.]) if self.white_background else np.array([0., 0., 0.])
            
            # Alpha blending
            image_blended = rgb * alpha + bg_color * (1 - alpha)
            
            # Restore back to 0~255 uint8! (Important)
            image = (image_blended * 255).astype(np.uint8)
            
            # Create mask (Alpha > 0.5)
            mask = (alpha > 0.5).reshape(H, W)
        else:
            image = image[..., :3] # Use as is if it's already RGB
            mask = np.ones((H, W), dtype=bool)

        camera_id = self.parser.camera_ids[index]
        K = self.parser.Ks_dict[camera_id].copy()
        camtoworld = self.parser.camtoworlds[index]

        # 3. Apply Patch / Crop
        if self.patch_size is not None:
            h, w = image.shape[:2]
            x = int(np.random.randint(0, max(w - self.patch_size, 1)))
            y = int(np.random.randint(0, max(h - self.patch_size, 1)))
            
            image = image[y : y + self.patch_size, x : x + self.patch_size]
            mask = mask[y : y + self.patch_size, x : x + self.patch_size]
            K[0, 2] -= x
            K[1, 2] -= y

        # 4. Return Data
        data = {
            "K": torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(camtoworld).float(),
            "image": torch.from_numpy(image).float(),
            "image_id": item,
            "camera_idx": self.parser.camera_indices[index],
            "mask": torch.from_numpy(mask).bool()
        }

        exposure = self.parser.exposure_values[index]
        if exposure is not None:
            data["exposure"] = torch.tensor(exposure, dtype=torch.float32)

        # [Note] Blender Synthetic datasets do not have Sparse Point Clouds like COLMAP,
        # making it difficult to provide "points" and "depths" natively. 
        # Therefore, even if load_depths=True, we should return empty tensors or raise an error here.
        # Based on the gsplat code structure, it is highly recommended to turn off depth_loss entirely.
        if self.load_depths:
            # We could add dummy data or a warning here, but disabling the setting in the config is recommended.
            pass

        return data
    


import re

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]