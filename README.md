# CoherentRaster

This repository provides an official implementation for *CoherentRaster: Efficient 3D Gaussian Splatting for Light Field Displays*

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://sgj0402.github.io/coherent-raster-project-page/)
[![arXiv](https://img.shields.io/badge/arXiv-2401.12345-b31b1b.svg?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2605.04509v1)

## 📖 Overview

* Training and rendering were tested in a **WSL environment**.
* The **Windows environment is only intended for the interactive viewer**, due to CUDA Interop limitations in WSL.


## 🛠️ Environment Setup

The experiments in the paper were conducted on a single RTX 5090 GPU.

### 1. Create Conda Environment

```bash
conda create -n coherent_raster python=3.11
conda activate coherent_raster
```

### 2. Install Pytorch

- Install PyTorch according to your GPU and CUDA version.


### 3. Clone Repository

Result file tree of repositories

```
your_folder
├── coherent-raster/
└── gsplat/
```

1. Clone CoherentRaster repository.

```bash
git clone https://github.com/sgj0402/coherent-raster.git
```

2. Clone [gsplat](https://github.com/nerfstudio-project/gsplat) repository.

```bash
git clone --recursive https://github.com/nerfstudio-project/gsplat.git
cd gsplat
git checkout a8d88d387f6e554b18153d309f5536696882de5c
git checkout -b cr
```


### 4. Patch

Install patch.

```
cd your_folder/coherent-raster
python install.py
```


## 🐧 Linux / WSL Installation

In `your_folder/gsplat`:

```bash
pip install --no-build-isolation -r coherent_raster/requirements.txt
pip install --no-build-isolation -e .
```


## 🪟 Windows Installation (Viewer Only)

> ⚠️ Training on Windows is **not recommended** due to package compatibility issues (PyTorch 2.11.0, CUDA 12.8).
> Use Windows only for visualization.

### 1. Setup Build Environment

You need Microsoft Visual Studio to build and activate build environment by:

```bash
\vcvar64.bat
set DISTUTILS_USE_SDK=1
```

`vcvar64.bat` path example:

`C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build`

### 2. Install Dependencies

In `your_folder\gsplat`:

Edit `coherent_raster\requirements.txt` and comment out or remove the following packages:

```
...
# git+https://github.com/rahul-goel/fused-ssim@...
# git+https://github.com/harry7557558/fused-bilagrid@...
# ppisp @ git+https://github.com/nv-tlabs/ppisp@v1.0.0
...
```

In `your_folder\gsplat`:

```bash
pip install --no-build-isolation -r coherent_raster\requirements.txt
```

### 3. Fix PyTorch Issue

Edit the following file: `...\site-packages\torch\include\torch\csrc\dynamo\compiled_autograd.h`

Comment out the block starting around line 1143.

```cpp
// #if defined(_WIN32) && (defined(USE_CUDA) || defined(USE_ROCM))
//     // NB: the if-constexpr usage triggers compilation errors on Windows
//     // with certain compiler settings
//     // (see https://github.com/pytorch/pytorch/pull/144707 for examples).
//     // It's not clear what the problem is, so we're going to ignore it for now.
// ...
//       TORCH_CHECK_NOT_IMPLEMENTED(
//           false, "IValuePacker not implemented for type");
//       return at::NoneType::get();
//     }
// #endif
```

### 4. Install

In `your_folder\gsplat`:

```bash
pip install --no-build-isolation -e .
```

### 5. Fix pycolmap Issue

Edit the following file: `...\site-packages\pycolmap\scene_manager.py`

Replace all occurrences of:

* `L` → `Q` in `struct.unpack`

Examples:

```python
struct.unpack('L', ...)   -> struct.unpack('Q', ...)
struct.unpack('<L', ...)  -> struct.unpack('<Q', ...)
struct.unpack('idLLL', ...)  -> struct.unpack('idQQQ', ...)
```


## 🗂️ Dataset Structure

### Viewpoint Index Matrix

* H x W x 3, start from 0 index (.npy)

We provide the viewpoint index matrix used in our experiments; however, the provided matrix may not be suitable for other displays due to display-specific calibration differences. [[Google Drive]](https://drive.google.com/drive/folders/1HiwoMv0j5GT74Oe3Ad9j7wSBA2kXFlsi?usp=sharing) 

### NeRF Synthetic

```
nerf_synthetic/
├── chair/
│   ├── train/
│   │   ├── r_0.png
│   │   ├── r_1.png
│   │   └── ...
│   ├── val/
│   │   ├── r_0.png
│   │   └── ...
│   ├── test/
│   │   ├── r_0.png
│   │   └── ...
│   ├── transforms_train.json
│   ├── transforms_val.json
│   └── transforms_test.json
├── drums/
├── ...
```

### Mip-NeRF 360 / COLMAP Format

```
360_v2/
├── bicycle/
│   ├── images/
│   │   ├── _DSC8733.JPG
│   │   ├── _DSC8734.JPG
│   │   └── ...
│   ├── images_2/
│   ├── images_4/
│   ├── images_8/
│   └── sparse/
│       └── 0/
│           ├── cameras.bin
│           ├── images.bin
│           └── points3D.bin
├── bonsai/
├── ...
```


## 🚀 Training

In `your_folder/gsplat/coherent_raster`:

Parameters are configured directly inside the wrapper scripts. Before running, open the script (e.g., `train_blender.py`) and modify the variables to match your environment:

```bash
python train_blender.py
python train_colmap_mcmc.py
python train_colmap.py
```

- `SCENE_DIR` : Root directory of the dataset
- `SCENE_LIST` : List of scenes to train
- `DATASET_CATEGORY` : Dataset type (`blender` or `colmap`)
- `RESULT_DIR` : Directory to save training outputs
- `CUDA_VISIBLE_DEVICES` : GPU index to use

- `CAP_MAX` : Maximum number of Gaussians for the MCMC (Markov Chain Monte Carlo) strategy (used in `train_colmap_mcmc.py`)

## 🖼️ Rendering

In `your_folder/gsplat/coherent_raster`:

Parameters are configured directly inside the wrapper scripts. Before running, open the script (e.g., `render_blender.py`) and modify the variables to match your environment:

```bash
python render_blender.py
python render_colmap.py
```

- `SCENE_DIR` : Root directory of the dataset
- `SCENE_LIST` : List of scenes to render
- `TRAIN_RESULT_DIR` : Directory containing trained checkpoints
- `RENDER_RESULT_DIR` : Directory to save rendered outputs

- `VIEWPOINT_INDEX_MATRIX_PATH` : Path to viewpoint index matrix (`.npy`) (H x W x 3, start from 0 index)

- `USING_DATA_IDX` : Index of the input view from dataset for synthesizing orbit cameras
- `ORBIT_DIRECTION` : Camera orbit direction (-1 or 1)
- `CROP_TO_FILL` : Enable crop-to-fill rendering (0 or 1)
- `VIEW_DEGREE` : Angular range of the orbiting camera around the scene center

- `CLUSTER_SIZE` : Number of views per cluster
- `USE_REMAPPING` : Enable View-Coherent Remapping (0 or 1)

- `SAVE_EACH_VIEW_IMAGE` : Save per-view images (0 or 1)
- `SAVE_INTERLACED_IMAGE` : Save interlaced image (0 or 1)

- `CUDA_VISIBLE_DEVICES` : GPU index to use

## 🎮 Interactive Viewer

> CUDA Interop is required, which is not supported in WSL. \
> Ensure your display settings are configured correctly:
> * Set LFD as the main display to enable Interop.
> * Set LFD display scale to 100%

In `your_folder/gsplat/coherent_raster`:

Parameters are configured directly inside the wrapper scripts. Before running, open the script (e.g., `view_blender.py`) and modify the variables to match your environment:

```bash
python view_blender.py
python view_colmap.py
```

- `SCENE_DIR` : Root directory of the dataset
- `SCENE` : Target scene for visualization
- `TRAIN_RESULT_DIR` : Directory containing trained checkpoints
- `RENDER_RESULT_DIR` : Directory to save rendered outputs

- `VIEWPOINT_INDEX_MATRIX_PATH` : Path to viewpoint index matrix (`.npy`) (H x W x 3, start from 0 index)

- `USING_DATA_IDX` : Index of the input view from dataset for synthesizing orbit cameras
- `ORBIT_DIRECTION` : Camera orbit direction (-1 or 1)
- `CROP_TO_FILL` : Enable crop-to-fill rendering (0 or 1)
- `VIEW_DEGREE` : Angular range of the orbiting camera around the scene center

- `CLUSTER_SIZE` : Number of views per cluster
- `USE_REMAPPING` : Enable View-Coherent Remapping (0 or 1)

- `CUDA_VISIBLE_DEVICES` : GPU index to use


Then open the **Viser web page** (ex: `http://localhost:8080` or `ws://localhost:8080`):

* Use the Viser page to control the camera
* Rendering output appears in a separate window
* Press 'F11' to toggle full-screen mode. Press 'P' to toggle Profiling Mode.


## 📜 License

This repository contains multiple components with different licenses:

- `src/`  
  Licensed under the MIT License.  
  Note: No patent rights are granted. See `PATENT_NOTICE.md` for details.

- `patches/`  
  Contains patch files for software originally licensed under the Apache License 2.0.  
  Portions of the patch files may include Apache-licensed code.  
  See `patches/LICENSE` and `patches/README.md` for details.


## ⚖️ Patent Notice

Commercial use involving patented technologies may require a separate license.

See `PATENT_NOTICE.md` for full details.


## ✒️ Citation

```
To be updated
```


## 🙏 Acknowledgements

* [gsplat](https://github.com/nerfstudio-project/gsplat)
