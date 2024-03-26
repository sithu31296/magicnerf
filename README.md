# MagicNeRF

MagicNeRF is a user-friendly and high-performance implementation of neural radiance fields ([NeRF](http://www.matthewtancik.com/nerf)), a cutting-edge technique for synthesizing photorealistic 3D scenes from 2D images.

Built with simplicity and efficiency in mind, MagicNeRF offers a seamless experience for researchers and developers looking to explore NeRF-based rendering and scene reconstruction.

If you are a seasoned researcher to NeRF or want a highly robust NeRF framework, checkout [nerfstudio](https://github.com/nerfstudio-project/nerfstudio) or [nerfacc](https://github.com/nerfstudio-project/nerfacc).

## Introduction

### Explicit Representations
Normally, a 3D scene is represented with point clouds, voxel grids or meshes.

### Implicit Representations


## Additional References to Read

* Fourier Features let Networks Learn High Frequency Functions in Low Dimensional Domains. (NeurIPS 2020)
* VolSDF: Volume Render of Neural Implicit Surfaces (NeurIPS 2021)
* MonoSDF: Exploring Monocular Geometric Cues for Neural Implicit Surface Reconstruction (NeurIPS 2022)
* Omnidata: A Scalable Pipeline for Making Multi-Task Mid-Level Vision Datasets from 3D Scans (ICCV 2021)

## Design

* Ease of use
* Performance
    * Fast and efficient training and rendering, enabling rapid experimentation and prototyping.
* Flexibility
    * Adapt the model to diverse applications and datasets.

## Installation

Clone the repository and install with the following:

```bash
pip install -e .
```

```bash
sudo apt install colmap
```


<!-- ## Workflow

1. Capture Images or Video
2. Estimate camera intrinsics and extrinsics via Structure-from-Motion
    * Colmap (Open-source)
    * Reality Capture (Commercial, much faster)
3. Convert to suitable input for Nerfstudio, InstantNGP, etc.
4. Start optimizing. -->

## Usage

### Download the Data

```bash
bash scripts/download_data.sh
```

### Optimization

```bash
bash run.py
```


### What is not implemented?

* Hierarchical Volume Sampling



## References
* [official_tensorflow_implementation](https://github.com/bmild/nerf)

This code is based on the following great repositories:
* [nerf_pytorch](https://github.com/yenchenlin/nerf-pytorch)
* [nerf_pl](https://github.com/kwea123/nerf_pl)

