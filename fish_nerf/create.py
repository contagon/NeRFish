import os
import torch
import hydra

from fish_nerf.renderer import renderer_dict
from fish_nerf.sampler import sampler_dict
from fish_nerf.models import volume_dict, LinearSphereModel, PoseModel, Model

def create_model(cfg, num_cams, poses_est=None):
    # Create models
    model = Model(cfg)
    model.cuda()
    model.train()

    camera = LinearSphereModel(cfg.data.fov_degree, req_grad=cfg.training.train_intrinsics)
    camera.cuda()
    camera.train()

    pose_model = PoseModel(num_cams, cfg.training.train_R, cfg.training.train_t, poses_est)
    pose_model.cuda()
    pose_model.train()

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
            camera.load_state_dict(loaded_data["camera"])
            pose_model.load_state_dict(loaded_data["pose"])
            start_epoch = loaded_data["epoch"]

            print(f"   => resuming from epoch {start_epoch}.")
            optimizer_state_dict = loaded_data["optimizer"]

    # Initialize the optimizer.
    optimizer = torch.optim.Adam(
        [
            {"params": model.parameters()},
            {"params": camera.parameters(), "lr": cfg.training.lr},
            {"params": pose_model.parameters(), "lr": cfg.training.lr},
        ],
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

    return model, camera, pose_model, optimizer, lr_scheduler, start_epoch, checkpoint_path