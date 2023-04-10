'''
An example script for using the camera model objects to recover ray vectors from pixel coordinates.
Note the following reference frames:
    - The camera frame is z-forward, x-right, y-down. And the rays will be specified in this frame.
    - The robot frame is NED (x-forward, y-right, z-down). And the poses will be specified in this frame.

    We need to take this into account when moving between camera-frame rays to world-frame points to remain consistent, I believe.
'''

# General imports.
import os
import sys
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.spatial.transform import Rotation

# Local imports.
sys.path.append('..')
from image_resampling.mvs_utils.camera_models import LinearSphere
from image_resampling.mvs_utils.shape_struct import ShapeStruct
from utils.dataset import TartanAirDataset


# Create the camera model. These are the intrinsics. of the dataset.
linsphere_camera_model = LinearSphere(
                    fov_degree = 195, 
                    shape_struct = ShapeStruct(256, 256),
                    in_to_tensor=False, 
                    out_to_numpy=False)

# Get some images.
tartanair_data_root = '../tartanair'
traj_data_root = os.path.join(tartanair_data_root, 'Sewerage', 'Data_hard', 'P000')
dataset = TartanAirDataset(traj_data_root=traj_data_root)

# Create the dataloader.
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x: x)

# Some geometry that we need for conversions.
R_ned_cam = torch.tensor([[0, 0, 1, 0],
                            [1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 0, 1]]).view(4, 4).float()

# Check that the transformation is correct.
# z-axis in the camera frame should point forward along the x-axis in the NED frame.
z_cam = torch.tensor([0, 0, 1, 0]).view(4, 1).float()
z_axis_ned = R_ned_cam @ z_cam
print("z-axis in the camera frame should point forward along the x-axis in the NED frame:")
print(z_axis_ned.T)

# x-axis in the camera frame should point forward along the y-axis in the NED frame.
x_cam = torch.tensor([1, 0, 0, 0]).view(4, 1).float()
x_axis_ned = R_ned_cam @ x_cam
print("x-axis in the camera frame should point forward along the y-axis in the NED frame:")
print(x_axis_ned.T)

# y-axis in the camera frame should point forward along the z-axis in the NED frame.
y_cam = torch.tensor([0, 1, 0, 0]).view(4, 1).float()
y_axis_ned = R_ned_cam @ y_cam
print("y-axis in the camera frame should point forward along the z-axis in the NED frame:")
print(y_axis_ned.T)


# Iterate over the dataset.
for i, data in enumerate(dataloader):
    # Get the data.
    img, pose_gt = data[0]

    # Compute rays from the camera model.

    # Tensor of pixels coordinates. Shape: (2, 256*256). This is where we can randomly choose pixels to create NeRF training batches.
    pixel_coords = torch.stack(torch.meshgrid(torch.arange(256), torch.arange(256))).view(2, -1)
    
    # Can also just choose one row.
    pixel_coords = torch.stack(torch.meshgrid(torch.arange(256), torch.tensor([128]))).view(2, -1)

    X_cam_rays, valid_mask = linsphere_camera_model.pixel_2_ray(pixel_coords)
    X_cam_rays = torch.stack([X_cam_rays[0], X_cam_rays[1], X_cam_rays[2], torch.ones_like(X_cam_rays[0])], dim=0)

    # Rays in the base frame NED.
    X_base_rays = R_ned_cam @ X_cam_rays

    # Verify that the middle ray points forward along the z-axis.
    # print("Pixel (128, 128) corresponds to ray:")
    # print(X_cam_rays[:, 256*128 + 128])

    # Convert the rays to world-frame points.
    # First, convert the pose to a transformation matrix.
    X_world_base = torch.eye(4)
    X_world_base[:3, 3] = torch.tensor(pose_gt[:3])
    X_world_base[:3, :3] = torch.tensor(Rotation.from_quat(pose_gt[3:]).as_matrix())

    # Convert sample on rays to world-frame points.
    sample_dists = torch.linspace(0, 100, 10)

    # Let's loop over the distances (vectorize later).
    # Visualize the points.
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for d in sample_dists:
        # Get the world-frame points.
        X_base_rays_scaled = X_base_rays.clone()
        X_base_rays_scaled[:3, :] = X_base_rays_scaled[:3, :] * d
        samples_world = X_world_base @ (X_base_rays_scaled)

        # Subsample the points.
        samples_world = samples_world[:, ::10]


        ax.scatter(samples_world[0, :], samples_world[1, :], samples_world[2, :])
    plt.show()



