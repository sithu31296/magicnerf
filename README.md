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


<!-- ```bash
sudo apt-get install \
    git \
    cmake \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-regex-dev \
    libboost-system-dev \
    libboost-test-dev \
    libeigen3-dev \
    libsuitesparse-dev \
    libfreeimage-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libcgal-qt5-dev \
    libatlas-base-dev \
    libsuitesparse-dev \
    libflann-dev \
    libsqlite3-dev \
    libmetis-dev \
```

Install Ceres-solver

```bash
git clone https://ceres-solver.googlesource.com/ceres-solver
cd ceres-solver
git checkout $(git describe --tags) # Checkout the latest release
mkdir build
cd build
cmake .. -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF
make
sudo make install
```

Install ColMap

```bash
git clone https://github.com/colmap/colmap
cd colmap
git checkout dev
mkdir build
cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=native
make
sudo make install
CC=/usr/bin/gcc-6 CXX=/usr/bin/g++-6 cmake ..
``` -->

## Workflow

1. Capture Images or Video
2. Estimate camera intrinsics and extrinsics via Structure-from-Motion
    * Colmap (Open-source)
    * Reality Capture (Commercial, much faster)
3. Convert to suitable input for Nerfstudio, InstantNGP, etc.
4. Start optimizing.

## Usage

### Download the Data

```bash
bash scripts/download_data.sh
```

### Optimization

```bash
bash run.py --config configs/lego.yaml
```

After training for 100k iterations (~4 hours on a single 2080ti), you can find the following video at.


### What is not implemented?

* Hierarchical Volume Sampling



## References
* [official_tensorflow_implementation](https://github.com/bmild/nerf)

This code is based on the following great repositories:
* [nerf_pytorch](https://github.com/yenchenlin/nerf-pytorch)
* [nerf_pl](https://github.com/kwea123/nerf_pl)

