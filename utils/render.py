import sys
import cv2
from scipy.spatial.transform import Rotation

sys.path.append("../fish_nerf")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from pytorch3d.renderer import PerspectiveCameras  # noqa: E402
from pytorch3d.renderer import look_at_view_transform  # noqa: E402

from fish_nerf.ray import get_pixels_from_image  # noqa: E402
from fish_nerf.ray import get_rays_from_pixels  # noqa: E402
from .dataset import trivial_collate
from image_resampling.mvs_utils.camera_models import ShapeStruct, Pinhole

import tqdm

def create_surround_cameras(radius, n_poses=20, up=(0.0, 1.0, 0.0), focal_length=1.0):
    """
    Spiral cameras looking at the origin
    """
    cameras = []

    for theta in np.linspace(0, 2 * np.pi, n_poses + 1)[:-1]:
        if np.abs(up[1]) > 0:
            eye = [
                np.cos(theta + np.pi / 2) * radius,
                0,
                -np.sin(theta + np.pi / 2) * radius,
            ]
        else:
            eye = [
                np.cos(theta + np.pi / 2) * radius,
                np.sin(theta + np.pi / 2) * radius,
                2.0,
            ]

        R, T = look_at_view_transform(
            eye=(eye,),
            at=([0.0, 0.0, 0.0],),
            up=(up,),
        )

        cameras.append(
            PerspectiveCameras(
                focal_length=torch.tensor([focal_length])[None],
                principal_point=torch.tensor([0.0, 0.0])[None],
                R=R,
                T=T,
            )
        )

    return cameras


def render_images(model, translation, num_images, save=False, file_prefix=""):
    # TODO: Make this work for both regular / fisheye cameras
    # (would be cool to see renders for both!)
    """
    Render a list of images from the given viewpoints.

    """
    all_images = []
    device = list(model.parameters())[0].device


    # Rotate around the origin of the camera. Aka, assign rotations to the input translation.
    for theta_ix, theta in enumerate(np.linspace(0, 2 * np.pi, num_images + 1)[:-1]):
        quat = Rotation.from_euler('z', theta, degrees=False).as_quat()
        pose = np.array([*translation, *quat])

        pixel_coords, pixel_xys = get_pixels_from_image(
            model.camera_model, filter_valid=True
        )

        # A ray bundle is a collection of rays. RayBundle Object includes origins, directions, sample_points, sample_lengths. Origins are tensor (N, 3) in NED world frame, directions are tensor (N, 3) of unit vectors our of the camera origin defined in its own NED origin, sample_points are tensor (N, S, 3), sample_lengths are tensor (N, S - 1) of the lengths of the segments between sample_points.
        ray_bundle = get_rays_from_pixels(pixel_coords, model.camera_model, model.X_ned_cam, pose, debug=False)
        
        ray_bundle.origins = ray_bundle.origins.to(dtype=torch.float32)
        ray_bundle.directions = ray_bundle.directions.to(dtype=torch.float32)

        # Run model forward
        out = model(ray_bundle)

        # Return rendered features (colors)
        image = np.zeros((model.camera_model.ss.W, model.camera_model.ss.H, 3))
        image[model.camera_model.get_valid_mask() == 1, :] = out["feature"].cpu().detach().numpy()
        all_images.append(image)

        # Save
        if save:
            plt.imsave(f"{file_prefix}_{theta}.png", image)

    return all_images


def render_images_in_poses(model, dataset, num_images = -1, save=False, file_prefix="", fix_heading=False):
    # TODO: Make this work for both regular / fisheye cameras
    # (would be cool to see renders for both!)
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
        image_gt, pose = batch[0]

        # Fix the heading, if required.
        if fix_heading:
            pose[:3,:3] = torch.eye(3)

        # ------------------------- Render fisheye ------------------------- #
        pixel_coords, pixel_xys = get_pixels_from_image(
            model.camera_model, filter_valid=True
        )

        # Render
        ray_bundle = get_rays_from_pixels(pixel_coords, model.camera_model, model.X_ned_cam, pose, debug=False)
 
        # Run model forward
        out = model(ray_bundle)

        # Return rendered features (colors)
        image_fish = np.zeros((model.camera_model.ss.W, model.camera_model.ss.H, 3))
        image_fish[model.camera_model.get_valid_mask() == 1, :] = out["feature"].cpu().detach().numpy()

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

        # Save
        if save:
            plt.imsave(f"{file_prefix}_traj.png", image)

    return all_images
