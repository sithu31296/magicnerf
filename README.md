# MagicNeRF

MagicNeRF is a user-friendly and high-performance implementation of neural radiance fields ([NeRF](http://www.matthewtancik.com/nerf)), a cutting-edge technique for synthesizing photorealistic 3D scenes from 2D images.

Built with simplicity and efficiency in mind, MagicNeRF offers a seamless experience for researchers and developers looking to explore NeRF-based rendering and scene reconstruction.

If you are a seasoned researcher to NeRF or want a highly robust NeRF framework, checkout [nerfstudio](https://github.com/nerfstudio-project/nerfstudio) or [nerfacc](https://github.com/nerfstudio-project/nerfacc).

## Introduction

Previously, a 3D scene is represented with point clouds, voxel grids or meshes.

NeRF uses an implicit function (MLP) to represent the 3D scene conditioned on the 5D coordinate inputs (spatial location (x, y, z) and viewing direction ($\theta$, $\phi$)).
The output is the volume density $\sigma$ and color $c$ at that spatial location.
The implicit function is optimized with classifical volume rendering equation with the gradients backpropagated from the photometric loss calculated between the ground-truth image and the rendered image.

> Notes: The term "optimization" is typically used instead of "training" in the neural radiance field methods since the model needs to be learned per scene. 

How can I use the NeRF technique?
* 3D reconstruction.
* Novel view synthesis (generating novel views from the learned implicit function.)
* Occupancy prediction (free-space modelling)



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
python run.py
```

### Results

> Notes: The results are tested with 8 novel views.

Playing with Implicit Function (MLP):
MLP dim | num_layers| Image Size | num_samples | num_rays | PSNR  | Memory
---     | ---       | ---        | ---         | ---      | ---   | ---
32      | 6         | 400x400    | 64          | 400*400  | 18.78 | 29GB
64      | 6         | 400x400    | 64          | 400*400  | 19.33 | 46GB
128     | 6         | 400x400    | 64          | 400*400  | 20.02 | 76GB
||
64      | 4         | 400x400    | 64          | 400*400  | 19.07 | 41GB
64      | 6         | 400x400    | 64          | 400*400  | 19.33 | 46GB
64      | 8         | 400x400    | 64          | 400*400  | 19.29 | 51GB

Playing with Ray Samplers:
MLP dim | num_layers| Image Size | num_samples | num_rays | PSNR  | Memory
---     | ---       | ---        | ---         | ---      | ---   | ---
64      | 6         | 400x400    | 32          | 400*400  | 18.45 | 23GB
64      | 6         | 400x400    | 64          | 400*400  | 19.33 | 46GB
64      | 6         | 400x400    | 128         | 400*400  | -     | OOM

Playing with Number of Rays in Training:
MLP dim | num_layers| Image Size | num_samples | num_rays | PSNR  | Memory
---     | ---       | ---        | ---         | ---      | ---   | ---
64      | 6         | 400x400    | 64          | 10*10    | 23.12 | 500MB
64      | 6         | 400x400    | 64          | 50*50    | 22.22 | 1GB
64      | 6         | 400x400    | 64          | 100*100  | 21.11 | 3GB
64      | 6         | 400x400    | 64          | 200*200  | 20.40 | 12GB
64      | 6         | 400x400    | 64          | 400*400  | 19.33 | 46GB
64      | 6         | 400x400    | 64          | 600*600  | -     | OOM

Optimal Model
MLP dim | num_layers| Image Size | num_samples | num_rays | PSNR  | Memory
---     | ---       | ---        | ---         | ---      | ---   | ---
32 (no skip) | 4    | 400x400    | 64          | 100*100  | 19.89 | 1.7GB
32      | 4         | 400x400    | 64          | 100*100  | 20.17 | 2.2GB
32 (w/o hidden encoder) | 4 | 400x400 | 64     | 100*100  | 20.40 | 2.1GB

Common Parameters:
* Epochs = 16
* LR = 5e-4

Tips
* Make a reasonable small MLP model. (small dim and number of layers)
* Increasing number of layers take much longer to train, despite a small improvement in quality.
* Sampling algorithm makes a big impact. (comes with an increased memory cost)
* Interestingly, smaller batch size achieves better quality. (but training time will be longer)



## References
* [official_tensorflow_implementation](https://github.com/bmild/nerf)

This code is based on the following great repositories:
* [nerf_pytorch](https://github.com/yenchenlin/nerf-pytorch)
* [nerf_pl](https://github.com/kwea123/nerf_pl)

