import torch


# Sampler which implements stratified (uniform) point sampling along rays
class StratifiedRaysampler(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.n_pts_per_ray = cfg.n_pts_per_ray
        self.min_depth = cfg.min_depth
        self.max_depth = cfg.max_depth

    def forward(
        self,
        ray_bundle,
    ):
        # Compute z values for self.n_pts_per_ray points
        # uniformly sampled between [near, far]
        z_vals = torch.linspace(
            self.min_depth, self.max_depth, self.n_pts_per_ray
        ).cuda()[:, None, None]

        # Sample points from z values
        ray_bundle.origins = ray_bundle.origins.cuda()
        ray_bundle.directions = ray_bundle.directions.cuda()
        sample_points = ray_bundle.origins + ray_bundle.directions * z_vals
        sample_points = sample_points.transpose(0, 1).contiguous()

        # Return
        return ray_bundle._replace(
            sample_points=sample_points,
            sample_lengths=z_vals.view(1, self.n_pts_per_ray, 1)
            * torch.ones_like(sample_points[..., :1]),
        )


sampler_dict = {"stratified": StratifiedRaysampler}
