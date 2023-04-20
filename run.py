# General imports.
import os
import random
import hydra
import imageio
import numpy as np
import torch
import tqdm
from omegaconf import DictConfig
import matplotlib.pyplot as plt

import pypose as pp
from fish_nerf.ray import (
    get_random_pixels_from_image,
    get_rays_from_pixels,
)

from fish_nerf.create import create_model
from utils.dataset import get_dataset, trivial_collate
from utils.render import render_images, render_images_in_poses
from utils.datasaver import IterationState
from utils.plot import animate_pose, plot_params
np.set_printoptions(suppress=True, precision=3, linewidth=100)


def train(cfg):
    torch.manual_seed(cfg.seed)
    for_plotting = IterationState(cfg.data.traj_data_root)


    # Load the training/validation data.
    train_dataset, val_dataset = get_dataset(
        traj_data_root=cfg.data.traj_data_root,
        image_shape=[cfg.data.image_shape[1], cfg.data.image_shape[0]],
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,  # One image at a time, and the batches are of ray samples.
        shuffle=True,
        num_workers=0,
        collate_fn=trivial_collate,
    )

    # Create model
    var_t = cfg.data.var_t if cfg.training.train_t else 0
    var_R = cfg.data.var_R if cfg.training.train_R else 0
    noise = pp.randn_SE3(train_dataset.num_frames, sigma=[var_t, var_R]).cuda()
    poses_est = train_dataset.poses_gt@noise
    model, camera, pose_model, optimizer, lr_scheduler, start_epoch, checkpoint_path = create_model(cfg, train_dataset.num_frames, poses_est)

    # Keep track of the camera poses (NED) seen so far (to sample from for validation).
    seen_camera_poses = set()

    # Run the main training loop.
    for epoch in range(start_epoch, cfg.training.num_epochs):
        t_range = tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader))

        pose_est = pose_model()
        pose_error = (train_dataset.poses_gt.Inv()@pose_est).Log().norm() # torch.linalg.inv(pose_gt)@pose

        for iteration, batch in t_range:
            idx, image, pose_gt = batch[0] # Batches are not collated, so `batch` is a list of samples. Take the first one only. NOTE(yoraish): This means that a batch size larger than 1 passed to the torch.utils.data.DataLoader will be a waste of work, and the first sample in the batch will be used (and incorreectly so, since we'll try to index into the tensor and things will probably break).
            pose = pose_model(idx)
            seen_camera_poses.add(idx)

            # Sample rays. The xy grid is of shape (2, N), where N is the number of rays. The first row is the x (column) coordinates, and the second row is the y (row) coordinates. By convention, the image origin is top left, and the x axis is to the right, and the y axis is down.
            pixel_coords, pixel_xys = get_random_pixels_from_image(
                cfg.training.batch_size, camera.model
            )

            if cfg.debug:
                # Show the valid mask.
                image_np = camera.model.get_valid_mask().cpu().numpy()
                plt.imshow(image_np)
                plt.show()

                # Show the sampled pixels.
                image_np = image.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 0
                image_np[pixel_coords.cpu()[1, :], pixel_coords.cpu()[0, :]] = 1
                plt.imshow(image_np)
                plt.show()

            # A ray bundle is a collection of rays. RayBundle Object includes origins, directions, sample_points, sample_lengths. Origins are tensor (N, 3) in NED world frame, directions are tensor (N, 3) of unit vectors our of the camera origin defined in its own NED origin, sample_points are tensor (N, S, 3), sample_lengths are tensor (N, S - 1) of the lengths of the segments between sample_points.
            ray_bundle = get_rays_from_pixels(pixel_coords, camera.model, model.X_ned_cam, pose, debug=cfg.debug)
          
            # Sample the image at the sampled pixels. rgb_gt is of shape (N, 3), where N is the number of rays.
            rgb_gt = image[:, :, pixel_coords[1, :].long(), pixel_coords[0, :].long()].squeeze(0).transpose(0, 1)

            # Run model forward
            out = model(ray_bundle)

            # Calculate loss
            loss = torch.nn.functional.mse_loss(out["feature"], rgb_gt)

            # Take the training step.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t_range.set_description(f"Epoch: {epoch:04d}, Loss: {loss:.06f}, FOV: {camera.forward():.03f}, Pose: {pose_error:.04f}")
            t_range.refresh()

            for_plotting.append(loss=loss.item(), fov=camera.forward().item())

        # Adjust the learning rate.
        lr_scheduler.step()

        pose_model.apply_delta()

        for_plotting.end_epoch(pose=pose_model().tensor().detach().cpu().numpy())

        # Checkpoint.
        if (
            epoch % cfg.training.checkpoint_interval == 0
            and len(cfg.training.checkpoint_path) > 0
            and epoch > 0
        ):
            print(f"Storing checkpoint {checkpoint_path}.")

            data_to_store = {
                "model": model.state_dict(),
                "camera": camera.state_dict(),
                "pose": pose_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }

            file = f"{checkpoint_path}_data.npz"
            for_plotting.save(file)

            torch.save(data_to_store, checkpoint_path)

        # Render
        if epoch % cfg.training.render_interval == 0 and epoch > 0:
            with torch.no_grad():
                # We can rednder images in a given pose, outputting a list of images showing the camera rotating around its own axis.
                # Choose a random camera pose.
                if cfg.vis_style == "random_pose":
                    idx = random.sample(seen_camera_poses, 1)[0]
                    random_pose = pose_model(idx)
                    print(f"Rendering at pose {random_pose}.")
                    test_images = render_images(
                        model,
                        camera,
                        translation = random_pose[:3,3],
                        num_images=20,
                    )
                
                # We can also use a torch dataset to render images. The poses from the dataset are used as input to the model and the output is rendered, concaternated with the ground truth image, and returned. Note that we can also optionally fix the heading of the camera to xyzw = 0001.
                if cfg.vis_style == "trajectory":
                    print("Rendering Trajectory")
                    test_images, fig = render_images_in_poses(
                        model,
                        camera,
                        pose_model,
                        train_dataset,
                        num_images=cfg.training.render_length,
                        fix_heading = False
                    )
                    fig.savefig(f'results/training_{epoch}_traj.png')
                    fig.clf()

                imageio.mimsave(
                    f"results/training_{epoch}.gif",
                    [np.uint8(im * 255) for im in test_images],
                )


def render(cfg):
    # Load the training/validation data.
    train_dataset, val_dataset = get_dataset(
        traj_data_root=cfg.data.traj_data_root,
        image_shape=[cfg.data.image_shape[1], cfg.data.image_shape[0]],
    )

    # Create model
    model, camera, pose_model, optimizer, lr_scheduler, start_epoch, checkpoint_path = create_model(cfg, train_dataset.num_frames, train_dataset.poses_gt)

    # Figure out our save folder
    data_type = cfg.data.traj_data_root.split(os.sep)[1]
    out_folder = os.path.join('media', data_type)
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    # Load saved data
    data = np.load(checkpoint_path + "_data.npz")

    # Render image
    print("Rendering Trajectory")
    with torch.no_grad():
        test_images, fig = render_images_in_poses(
            model,
            camera,
            pose_model,
            train_dataset,
            num_images=cfg.training.render_length,
            fix_heading = False
        )
        fig.clf()
        imageio.mimsave(
            os.path.join(out_folder, "render.gif"),
            [np.uint8(im * 255) for im in test_images],
        )

    # Animate poses
    print("Animating poses")
    animate_pose(
        data["pose"], 
        train_dataset.poses_gt.cpu().tensor(), 
        os.path.join(out_folder, "animate_poses.gif"), 
        cfg.plot.num_animate
    )
    plt.clf()

    # Epoch / Loss
    print("Animating poses")
    plot_params(
        data["loss"], 
        data["fov"], 
        os.path.join(out_folder, "loss_fov.png")
    )


@hydra.main(config_path="./configs", config_name="main", version_base=None)
def main(cfg: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())

    if cfg.type == "render":
        render(cfg)
    elif cfg.type == "train":
        train(cfg)


if __name__ == "__main__":
    main()
