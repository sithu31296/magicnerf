


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

