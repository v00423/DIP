# Assignment 4 - Implement Simplified 3D Gaussian Splatting

## This repositories is WeiLin Liu's implement of of Assignment_04 of DIP.



# Requirements
To install requirements:

```bash
python -m pip install -r requirements.txt
```

# Running

## To finish Task 1 (Structure-from-Motion with COLMAP),run:

```bash
python mvs_with_colmap.py --data_dir data/lego

python debug_mvs_by_projecting_pts.py --data_dir data/lego
```

## To finish Task 2 (Simplified 3D Gaussian Splatting),run:

```bash
python train.py --colmap_dir data/chair --checkpoint_dir data/lego/checkpoints

python render_3dgs_mv.py \
    --colmap_dir data/lego \
    --checkpoint data/chair/checkpoints/checkpoint_000060.pt \
    --num_frames 240 --fps 30
```

## To finish Task 3 (Official 3D Gaussian Splatting) please check this website.[官方 3DGS](https://github.com/graphdeco-inria/gaussian-splatting)

## Data
```
data/
└──lego/images/    # 100 张 multi-view 渲染图像
```

# Results and Discusion

## Task 1: Structure-from-Motion with COLMAP

### Projection and Debug figs

<p align="center">
<img src="./data/lego/projections/r_1.png" alt="Projection1"> <img src="./data/lego/projections/r_16.png" alt="Projection2"><img src="./data/lego/projections/r_59.png" alt="Projection3">
</p>

## Task 2: Simplified 3D Gaussian Splatting

### Simplified 3DGS Results

<p align="center">
<img src="./data/lego/checkpoints/debug_images/epoch_0000.png" alt="Debug1">

<img src="./data/lego/checkpoints/debug_images/epoch_0062.png" alt="Debug1">

<img src="./data/lego/checkpoints/debug_images/epoch_0192.png" alt="Debug1">
</p>

<p align="center">
<a href="./data/lego/checkpoints/debug_rendering.mp4">Click here to watch Debug video</a>
</p>

As you can see, as the number of epochs increases (the number of training iterations increases), the clarity of the rendered image gradually increases, becoming more and more faithful to the original image.

We also can have the following files:
```
lego/
├── sparse/
│   └── 0
│       ├──cameras.bin
│       ├──frames.bin
│       └──...
└──databse.db     
```
which is pretty convenient for the next task.

## Render a Multi-view Video
<p align="center">
<a href="./data/lego/render_mv.mp4">Click here to watch rendered video</a>
</p>

### Task 1 and Task 2 are run based on the following Python environment:
```
    Python 3.11.15(Windows 11)
    Pytorch 2.11.0
    CUDA 12.8
    Device_CPU = " AMD Ryzen 7500F 6-Core Processor "
    Device_GPU = " NVIDIA RTX 5060Ti 8G "
```

## Task 3: Compare with the Official 3DGS Implementation

### This task is run based on the following Python environment:
```
    Python  3.8(ubuntu)
    PyTorch  2.0.0
    CUDA  11.8
    Device_CPU = " AMD EPYC 7T83 64-Core Processor "
    Device_GPU = " NVIDIA RTX 4080 32G "
```
### Output:point cloud

<p align="center">
<img src="./official3DGS/output/lego/point_cloud/pointcloud1.png" alt="Pointcloud1" width="40%">

<img src="./official3DGS/output/lego/point_cloud/pointcloud2.png" alt="Pointcloud2" width="40%">
</p>

### Rendered figs

The leftside fig is original figs,the rightside fig is rendered figs.

<p align="center">
<img src="./official3DGS/output/lego/figs/test/ours_30000/gt/00000.png" alt="original figs1" width="25%" > <img src="./official3DGS/output/lego/figs/test/ours_30000/renders/00000.png" alt="rendered figs1" width="25%" >
</p>

<p align="center">
<img src="./official3DGS/output/lego/figs/test/ours_30000/gt/00010.png" alt="original figs2" width="25%" > <img src="./official3DGS/output/lego/figs/test/ours_30000/renders/00010.png" alt="rendered figs2" width="25%" >
</p>

As can be seen, the point cloud and images rendered using the official 3DGS are of extremely high quality, perfectly displaying numerous details of the original image (such as the bumps on the LEGO brick version). Its performance is much higher than the simplified 3DGS implementation in Task 2.

# Code performance comparison

## I. Rendering quality

<p align="center">
<img src="./data/lego/checkpoints/debug_images/epoch_0192.png" alt="Debug1" width="50%" ><img src="./official3DGS/output/lego/figs/test/ours_30000/renders/00010.png" alt="rendered figs2" width="25%" >
</p>

Clearly, the rendering quality of the official 3DGS is much higher than that of the simplified 3DGS. It is more detailed in its reproduction of details and its processing of high-resolution images, and it can be trained without cropping.

## II. Training speed

Simplified 3DGS training requires about an hour to iterate through all images in the dataset, while according to the saved training log, the official 3DGS only takes about three minutes to complete all training and provide high-quality results.

<p align="center">
check[official3DGSTraninglog](official3DGS/log.txt)
</p>


## III. Memory usage

<p align="center">
<img src="./Figs/simplified.png" alt="simplified" >
</p>

When training with a simplified version of 3DGS, an RTX 5060 Ti with 8GB of VRAM used approximately 3GB of VRAM. 

<p align="center">
<img src="./Figs/official.png" alt="official" >
</p>

However, when training with the official 3DGS, an RTX 4080 with 32GB of VRAM only used about 1GB of VRAM.

## IV. Discussion on different performance characteristics

### Rendering quality

**(1)Differences in Color Representation Capabilities**

The simplified version only learns RGB (gaussian_model.py uses 3-channel logit parameters for color). 
The official version uses SH and progressively increases the degree (sh_degree=3, and oneupSHdegree() during training).

**(2)Differences in Loss Functions**

The simplified version only uses L1.
The official version uses L1 + SSIM (can be fused_ssim) + optional depthwise regularization.

**(3)Resolution/Sampling Differences**

The simplified version defaults to downsample_factor=8, and training images are scaled down first.
The official version loads according to camera resolution, only performing controlled scaling on large images.

**(4)Exposure/Color Compensation Differences**

The official version supports the per-image exposure parameter in training and rendering.
The simplified version has no corresponding mechanism.

**(5)Conclusion**

The official approach systematically improves reconstruction quality through stronger parameterization (SH), stronger supervision (SSIM/Depth), imaging consistency (Exposure), and higher effective resolution, rather than simply relying on "more training."

### Training speed

**(1)Differences in Rendering Computation Paths**

The simplified version explicitly constructs the (N,H,W) Gaussian response in PyTorch, performing inverse/det/einsum/cumprod.
The official version places both forward and reverse directions into the CUDA op.

**(2)Differences in Rasterization Strategy**

The simplified version uses a "full Gaussian × full pixel" calculation.
The official version uses tile/binning + sorting + parallel blending per tile.

**(3)Visibility and Early Termination**

The official documentation includes frustum culling, radius filtering, and early alpha termination.
The simplified version only uses a depth threshold mask, but still calculates a large number of invalid values ​​first.

**(4)Sparse Optimization and Fusion Loss**

The official documentation offers SparseGaussianAdam (updated by visibility points) , and supports fused SSIM . The simplified version is a standard Adam full-parameter update. 
Tile + sort makes memory access more contiguous and thread utilization higher; forward and backward shared buffers (geom/binning/img buffer) reduce the overhead of repeated construction.

**(5)Conclusion**

The essence of the official speedup is: algorithm-level pruning (visibility/bugging) + engineering-level fusion (CUDA forward and backward) + optimizer-level sparse updates, rather than simply adjusting the learning rate.

### Memory usage

**(1)Intermediate Tensor Shape Differences**

The simplified version explicitly preserves large tensors such as dx(N,H,W,2), gaussian(N,H,W), and alphas(N,H,W).
The official version processes tensors by tile, uses shared memory for mini-batch caching, and does not construct a global N×H×W.

**(2)Parameter Storage Compression and Visibility-Driven**

The official version uses a 6-parameter symmetric form for internal covariance.
The official training process prune the points. Points can be sparsely updated based on visibility.

**(3)Reverse caching strategy**

The official version only saves necessary buffers (geom/binning/img) for backward.
The simplified version relies on general autograd to track a large number of dense intermediate values.

**(4)Conclusion**

The key to saving VRAM in the official version is: avoiding fully expanded intermediate tensors + tiled local working set + sparse visibility updates + dynamic point control (densify/prune).

