


#### SDF (Signed Distance Function)
* The further away from the surface -> + or larger,
inside the surface -> - or smaller.
* How?
    * Store signed distance values in voxel grid.
    * Geometry/mesh can be obtained by extracting level set for distance 0 (on the surface). 
* Issues?
    * Memory intensive (because of voxel grids)
    * Assume Lambertian surfaces

#### NeRF (Neural Radiance Fields) (ECCV 2020)
* Compact
* Learn the volume representation
* Represent a scene with a MLP given a 3D point (x, y, z) and a viewing direction (theta and phi), it will output the color and volume density.
* Volume sampling
    * Near plane
    * Far plane
    * Subdivide into equally sized intervals
    * Uniform sampling inside intervals -> continuous sampling of the volume


Hierarchical Volume Sampling
* Two networks to sample corase and fine features

Input encoding
* Fourier encoding
* Fourier Features let Networks Learn High Frequency Functions in Low Dimensional Domains. (NeurIPS 2020)

Disadvantages
* Very slow. For each 3D point, we need to evaluate the network.
* Training takes 1-2 days on a single GPU.
* Rendering image takes seconds to minutes.

Do we really need a neural network?
* Plenoxels: Radiance Feilds without Neural Networks (CVPR 2022)
* Rather than optimizing the weights of a neural network, optimize the weights of spherical harmonics.
* Plus a regularization term 
* Trilinear interpolation of volume density -> grid needs to be fine enough.
* Reducing training time at cost of higher memory consumption (because of voxels)

Combining Voxel Grids with Networks
* InstantNGP: Instant Neural Graphics Primitives (SIGGRAPH 2022)
* Each voxel represents a trainable feature vectors
* Concatenated trilinearly interpolated features with additional information (viewing direction, etc.)
* Trilinearly interpolated features are passed to a shallow neural network
* Jointly train features and neural network.
* Decreases the quality because of multi-resolution grid.
* Multi-resolution Hash Encoding

NeRF in the Wild
* NeRF in the Wild: Neural Radiance Fields for Unconstrained Photo Collections (CVPR 20221)
* Illumination changes
* Transient objects (Pedestrians)
* Low-dimensional vector encoding appearance (learned)
* Learned per image to account for transient objects (not needed at testing time)
* Then Learn two outputs for static (reconstructed object) and transient objects.
* You can render the scene with different learned appearance embeddings to simlate different illumination conditions.

Scaling to Large Scene
* Block-NeRF: Scalable Large Scene Neural View Synthesis (CVPR 2022)

Relaxing Constraints
* RegNeRF; Regularizing Neural Radiance Fields for View Synthesis from Sparse Inputs (CVPR 2022)

Getting More Constraints from Monocular Cues
* Depth map
* Normal map
* Measurements are noisy (and up to scale), but provide priors


### Extracting 3D Geometry from Neural Radiance Fields

* What is the right density threshold to use?
    It is not clear what is the right density value.
* Volume density does necessarily model 3D geometry
* Estimate SDF, model volume density as learnable function of SDF.


### Details

Volume bounds
* We have to scale the scene so that the continuous 5D coordinates along camera rays lies within a cube of side length 2 centered at the origin, and only query the representation within this bounding volume.
* While the dataset with real images contains content that can exist anywhere between the closest point and infinity, so you must use Normalized Device Coordinates (NDC) to map the depth range of these points into [-1, 1].
* This shifts all the ray origins to the near plane of the scene, maps the perspective rays of the camera to parallel rays in the transformed volume, and uses disparity (inverse depth) instead of metric depth, so all coordinates are now bounded.


### Training Tricks
* For real scene data, they regularize the network by adding random Gaussian noise with zero mean and unit variance to the output $/sigma$ values (before passing them through the ReLU) during optimization, finding that this slightly improves visual performance for rendering novel views.
* To render new views at test time, they sample 64 points per ray through the coarse network and 64+128=192 points per ray through the fine network, for a total of 256 network queries per ray.




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
