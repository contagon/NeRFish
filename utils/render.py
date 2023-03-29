import sys

sys.path.append("../fish_nerf")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from pytorch3d.renderer import PerspectiveCameras  # noqa: E402
from pytorch3d.renderer import look_at_view_transform  # noqa: E402

from fish_nerf.ray import get_pixels_from_image  # noqa: E402
from fish_nerf.ray import get_rays_from_pixels  # noqa: E402


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


def render_images(model, cameras, image_size, save=False, file_prefix=""):
    # TODO: Make this work for both regular / fisheye cameras
    # (would be cool to see renders for both!)
    """
    Render a list of images from the given viewpoints.

    """
    all_images = []
    device = list(model.parameters())[0].device

    for cam_idx, camera in enumerate(cameras):
        print(f"Rendering image {cam_idx}")

        torch.cuda.empty_cache()
        camera = camera.to(device)
        xy_grid = get_pixels_from_image(image_size, camera)
        ray_bundle = get_rays_from_pixels(xy_grid, image_size, camera)

        # Implement rendering in renderer.py
        out = model(ray_bundle)

        # Return rendered features (colors)
        image = np.array(
            out["feature"].view(image_size[1], image_size[0], 3).detach().cpu()
        )
        all_images.append(image)

        # Visualize depth
        if cam_idx == 2 and file_prefix == "":
            depth = out["depth"].view(image_size[1], image_size[0]).detach().cpu()
            plt.imsave("data/depth.png", depth)

        # Save
        if save:
            plt.imsave(f"{file_prefix}_{cam_idx}.png", image)

    return all_images
