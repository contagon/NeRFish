import sys
from scipy.spatial.transform import Rotation

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from fish_nerf.ray import get_pixels_from_image  # noqa: E402
from fish_nerf.ray import get_rays_from_pixels  # noqa: E402
from .dataset import trivial_collate
from image_resampling.mvs_utils.camera_models import ShapeStruct, Pinhole
import pypose as pp

import tqdm


def render_images(model, camera, translation, num_images):
    # TODO: Make this work for both regular / fisheye cameras
    # (would be cool to see renders for both!)
    """
    Render a list of images from the given viewpoints.

    """
    all_images = []
    device = list(model.parameters())[0].device
    camera_model = camera.model


    # Rotate around the origin of the camera. Aka, assign rotations to the input translation.
    for theta_ix, theta in enumerate(np.linspace(0, 2 * np.pi, num_images + 1)[:-1]):
        quat = Rotation.from_euler('z', theta, degrees=False).as_quat()
        pose = np.array([*translation, *quat])

        pixel_coords, pixel_xys = get_pixels_from_image(
            camera_model, filter_valid=True
        )

        # A ray bundle is a collection of rays. RayBundle Object includes origins, directions, sample_points, sample_lengths. Origins are tensor (N, 3) in NED world frame, directions are tensor (N, 3) of unit vectors our of the camera origin defined in its own NED origin, sample_points are tensor (N, S, 3), sample_lengths are tensor (N, S - 1) of the lengths of the segments between sample_points.
        ray_bundle = get_rays_from_pixels(pixel_coords, camera_model, model.X_ned_cam, pose, debug=False)
        
        ray_bundle.origins = ray_bundle.origins.to(dtype=torch.float32)
        ray_bundle.directions = ray_bundle.directions.to(dtype=torch.float32)

        # Run model forward
        out = model(ray_bundle)

        # Return rendered features (colors)
        image = np.zeros((camera_model.ss.W, camera_model.ss.H, 3))
        image[camera_model.get_valid_mask() == 1, :] = out["feature"].cpu().detach().numpy()
        all_images.append(image)

    return all_images


def render_images_in_poses(model, camera, pose_model, dataset, num_images = -1, save_traj=True, fix_heading=False):
    """
    Render a list of images from the given viewpoints.

    """
    all_images = []

    # A dataloader for the images and poses.
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1, # One image at a time, and the batches are of ray samples.
        shuffle=False,
        num_workers=0,
        collate_fn=trivial_collate,
    )
    
    fish_camera_model = camera.model
    pinhole_camera_model = Pinhole(128, 
                               128, 
                               128, 
                               128, 
                               ShapeStruct(256,256),
                            in_to_tensor=False, 
                            out_to_numpy=False)
    pinhole_camera_model.device = 'cuda'

    num_images = num_images if num_images != -1 else len(dataloader)
    t_range = tqdm.tqdm(enumerate(dataloader), total=num_images)

    # Rotate around the origin of the camera. Aka, assign rotations to the input translation.
    for iter, batch in t_range:

        if num_images > 0 and iter >= num_images:
            break

        # Get the batch contents.
        idx, image_gt, pose_gt = batch[0]
        pose = pose_model(idx)

        # Fix the heading, if required.
        if fix_heading:
            pose.rotation().identity_()

        # ------------------------- Render fisheye ------------------------- #
        pixel_coords, pixel_xys = get_pixels_from_image(
            fish_camera_model, filter_valid=True
        )

        # Render
        ray_bundle = get_rays_from_pixels(pixel_coords, fish_camera_model, model.X_ned_cam, pose, debug=False)
 
        # Run model forward
        out = model(ray_bundle)

        # Return rendered features (colors)
        mask = fish_camera_model.get_valid_mask().cpu()
        image_fish = np.zeros((fish_camera_model.ss.W, fish_camera_model.ss.H, 3))
        image_fish[mask == 1, :] = out["feature"].cpu().detach().numpy()

        # ------------------------- Render projective ------------------------- #        
        pixel_coords, pixel_xys = get_pixels_from_image(
            pinhole_camera_model, filter_valid=True
        )

        # Render
        ray_bundle = get_rays_from_pixels(pixel_coords, pinhole_camera_model, model.X_ned_cam, pose, debug=False)
 
        # Run model forward
        out = model(ray_bundle)

        image_proj = out["feature"].view(256,256,3).detach().cpu().numpy()

        # ------------------------- Save & Return ------------------------- #
        # Concatenate the original images and the rendered images.
        image_gt_viewed = image_gt.squeeze().permute(1,2,0).cpu()
        image = np.concatenate((image_gt_viewed, image_fish, image_proj), axis=1)

        all_images.append(image)


    # ------------------------- Save trajectory as well ------------------------- #
    if save_traj:
        pose_est = (pose_model.init_c2w @ pose_model.delta.Exp()).translation().cpu()
        poses_gt = dataset.poses_gt.translation().cpu()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(pose_est[:,0], pose_est[:,1], marker='o')
        ax.plot(poses_gt[:num_images,0].cpu(), poses_gt[:num_images,1].cpu(), marker='o')
    else:
        fig = None


    return all_images, fig
