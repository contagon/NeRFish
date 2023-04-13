# General imports.
import os

import hydra
import imageio
import numpy as np
import torch
import tqdm
from omegaconf import DictConfig

from fish_nerf.models import volume_dict
from fish_nerf.ray import (
    get_random_pixels_from_image,
    get_rays_from_pixels,
    sample_images_at_xy,
)
from fish_nerf.renderer import renderer_dict
from fish_nerf.sampler import sampler_dict
from image_resampling.mvs_utils.camera_models import LinearSphere
from image_resampling.mvs_utils.shape_struct import ShapeStruct
from utils.dataset import get_dataset, trivial_collate
from utils.render import create_surround_cameras, render_images

np.set_printoptions(suppress=True, precision=3)

# Some geometry that we need for conversions.
R_ned_cam = (
    torch.tensor([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    .view(4, 4)
    .float()
)


# Model class containing:
#   1) Implicit volume defining the scene
#   2) Sampling scheme which generates sample points along rays
#   3) Renderer which can render an implicit volume given a sampling scheme


class Model(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # Create the camera model. These are the intrinsics. of the dataset.
        LinearSphere(
            fov_degree=cfg.fov_degree,
            shape_struct=ShapeStruct(256, 256),
            in_to_tensor=False,
            out_to_numpy=False,
        )

        # Get implicit function from config
        self.implicit_fn = volume_dict[cfg.implicit_function.type](
            cfg.implicit_function
        )

        # Point sampling (raymarching) scheme
        self.sampler = sampler_dict[cfg.sampler.type](cfg.sampler)

        # Initialize volume renderer
        self.renderer = renderer_dict[cfg.renderer.type](cfg.renderer)

    def forward(self, ray_bundle):
        # Call renderer with
        #  a) Implicit volume
        #  b) Sampling routine

        return self.renderer(self.sampler, self.implicit_fn, ray_bundle)


def create_model(cfg):
    # Create model
    model = Model(cfg)
    model.cuda()
    model.train()

    # Load checkpoints
    optimizer_state_dict = None
    start_epoch = 0

    checkpoint_path = os.path.join(
        hydra.utils.get_original_cwd(), cfg.training.checkpoint_path
    )

    if len(cfg.training.checkpoint_path) > 0:
        # Make the root of the experiment directory.
        checkpoint_dir = os.path.split(checkpoint_path)[0]
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Resume training if requested.
        if cfg.training.resume and os.path.isfile(checkpoint_path):
            print(f"Resuming from checkpoint {checkpoint_path}.")
            loaded_data = torch.load(checkpoint_path)
            model.load_state_dict(loaded_data["model"])
            start_epoch = loaded_data["epoch"]

            print(f"   => resuming from epoch {start_epoch}.")
            optimizer_state_dict = loaded_data["optimizer"]

    # Initialize the optimizer.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.training.lr,
    )

    # Load the optimizer state dict in case we are resuming.
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
        optimizer.last_epoch = start_epoch

    # The learning rate scheduling is implemented with LambdaLR PyTorch scheduler.
    def lr_lambda(epoch):
        return cfg.training.lr_scheduler_gamma ** (
            epoch / cfg.training.lr_scheduler_step_size
        )

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda, last_epoch=start_epoch - 1, verbose=False
    )

    return model, optimizer, lr_scheduler, start_epoch, checkpoint_path


def train(cfg):
    # Image shape and size. Be convention,
    # the image shape is [H, W] and the image size is [W, H].
    image_size = [cfg.data.image_shape[1], cfg.data.image_shape[0]]

    # Create model
    model, optimizer, lr_scheduler, start_epoch, checkpoint_path = create_model(cfg)

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

    # Run the main training loop.
    for epoch in range(start_epoch, cfg.training.num_epochs):
        t_range = tqdm.tqdm(enumerate(train_dataloader))

        for iteration, batch in t_range:
            # Batches are not collated, so `batch` is a list of samples.
            # Take the first one only. NOTE(yoraish): This means that a batch size
            # larger than 1 passed to the torch.utils.data.DataLoader will be a
            # waste of work, and the first sample in the batch will be used
            # (and incorreectly so, since we'll try to index into the tensor and
            # things will probably break).
            image, pose = batch[0]
            image = image.cuda().unsqueeze(0)
            # TODO Setup camera
            camera = None
            camera = camera.cuda()

            # Sample rays
            xy_grid = get_random_pixels_from_image(
                cfg.training.batch_size, image_size, camera
            )
            ray_bundle = get_rays_from_pixels(xy_grid, image_size, camera)
            rgb_gt = sample_images_at_xy(image, xy_grid)

            # Run model forward
            out = model(ray_bundle)

            # Calculate loss
            loss = torch.nn.functional.mse_loss(out["feature"], rgb_gt)

            # Take the training step.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t_range.set_description(f"Epoch: {epoch:04d}, Loss: {loss:.06f}")
            t_range.refresh()

        # Adjust the learning rate.
        lr_scheduler.step()

        # Checkpoint.
        if (
            epoch % cfg.training.checkpoint_interval == 0
            and len(cfg.training.checkpoint_path) > 0
            and epoch > 0
        ):
            print(f"Storing checkpoint {checkpoint_path}.")

            data_to_store = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }

            torch.save(data_to_store, checkpoint_path)

        # Render
        if epoch % cfg.training.render_interval == 0 and epoch > 0:
            with torch.no_grad():
                test_images = render_images(
                    model,
                    create_surround_cameras(
                        4.0, n_poses=20, up=(0.0, 0.0, 1.0), focal_length=2.0
                    ),
                    image_size,
                    file_prefix="nerf",
                )
                imageio.mimsave(
                    f"results/training_{epoch}.gif",
                    [np.uint8(im * 255) for im in test_images],
                )


def render(
    cfg,
):
    # Create model
    model = Model(cfg)
    model = model.cuda()
    model.eval()

    # Render spiral
    cameras = create_surround_cameras(3.0, n_poses=20)
    all_images = render_images(model, cameras, cfg.data.image_shape)
    imageio.mimsave("results/3d_revolve.gif", [np.uint8(im * 255) for im in all_images])


@hydra.main(config_path="./configs", config_name="main", version_base=None)
def main(cfg: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())

    if cfg.type == "render":
        render(cfg)
    elif cfg.type == "train":
        train(cfg)


if __name__ == "__main__":
    main()
