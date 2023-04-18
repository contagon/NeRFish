import torch
import torch.nn as nn
import torch.nn.functional as F

from fish_nerf.ray import RayBundle
from utils.lie_groups import make_c2w

import pypose as pp
from image_resampling.mvs_utils.camera_models import LinearSphere
from image_resampling.mvs_utils.shape_struct import ShapeStruct

# ------------------------- Helper Modules ------------------------- #


class HarmonicEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        n_harmonic_functions: int = 6,
        omega0: float = 1.0,
        logspace: bool = True,
        include_input: bool = True,
    ) -> None:
        super().__init__()

        if logspace:
            frequencies = 2.0 ** torch.arange(
                n_harmonic_functions,
                dtype=torch.float32,
            )
        else:
            frequencies = torch.linspace(
                1.0,
                2.0 ** (n_harmonic_functions - 1),
                n_harmonic_functions,
                dtype=torch.float32,
            )

        self.register_buffer("_frequencies", omega0 * frequencies, persistent=False)
        self.include_input = include_input
        self.output_dim = n_harmonic_functions * 2 * in_channels

        if self.include_input:
            self.output_dim += in_channels

    def forward(self, x: torch.Tensor):
        embed = (x[..., None] * self._frequencies).view(*x.shape[:-1], -1)

        if self.include_input:
            return torch.cat((embed.sin(), embed.cos(), x), dim=-1)
        else:
            return torch.cat((embed.sin(), embed.cos()), dim=-1)


class LinearWithRepeat(torch.nn.Linear):
    def forward(self, input):
        n1 = input[0].shape[-1]
        rays = input[1].shape[0]
        output1 = F.linear(input[0], self.weight[:, :n1], self.bias)
        output2 = F.linear(input[1], self.weight[:, n1:], None)
        out_size = output1.shape[1]
        tog = output1.view(-1, rays, out_size) + output2.unsqueeze(0)
        return tog.view(-1, out_size)


class MLPWithInputSkips(nn.Module):
    def __init__(
        self,
        n_layers: int,
        input_dim: int,
        output_dim: int,
        skip_dim: int,
        hidden_dim: int,
        input_skips,
    ):
        super().__init__()

        layers = []

        for layeri in range(n_layers):
            if layeri == 0:
                dimin = input_dim
                dimout = hidden_dim
            elif layeri in input_skips:
                dimin = hidden_dim + skip_dim
                dimout = hidden_dim
            else:
                dimin = hidden_dim
                dimout = hidden_dim

            linear = torch.nn.Linear(dimin, dimout)
            do_xavier(linear)
            layers.append(torch.nn.Sequential(linear, torch.nn.ReLU(True)))

        self.mlp = nn.ModuleList(layers)
        self._input_skips = set(input_skips)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        y = x

        for li, layer in enumerate(self.mlp):
            if li in self._input_skips:
                y = torch.cat((y, z), dim=-1)

            y = layer(y)

        return y


def do_xavier(layer):
    torch.nn.init.xavier_uniform_(layer.weight.data)


# ------------------------- Actual Models ------------------------- #


class NeuralRadianceField(nn.Module):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()

        self.harmonic_embedding_xyz = HarmonicEmbedding(3, cfg.n_harmonic_functions_xyz)
        self.harmonic_embedding_dir = HarmonicEmbedding(3, cfg.n_harmonic_functions_dir)

        embedding_dim_xyz = self.harmonic_embedding_xyz.output_dim
        embedding_dim_dir = self.harmonic_embedding_dir.output_dim

        # First network
        self.start_layers = MLPWithInputSkips(
            cfg.n_layers_xyz,
            embedding_dim_xyz,
            None,
            embedding_dim_xyz,
            cfg.n_hidden_neurons_xyz,
            [cfg.append_xyz],
        )

        # Mini network just for density
        self.density_layers = torch.nn.Sequential(
            torch.nn.Linear(cfg.n_hidden_neurons_xyz, 1),
            torch.nn.ReLU(inplace=True),
        )
        self.density_layers[0].bias.data[:] = 0.0
        do_xavier(self.density_layers[0])

        # Networks for color
        self.color1 = torch.nn.Linear(
            cfg.n_hidden_neurons_xyz, cfg.n_hidden_neurons_xyz
        )
        self.color2 = torch.nn.Sequential(
            LinearWithRepeat(
                cfg.n_hidden_neurons_xyz + embedding_dim_dir, cfg.n_hidden_neurons_dir
            ),
            torch.nn.ReLU(True),
            torch.nn.Linear(cfg.n_hidden_neurons_dir, 3),
            torch.nn.Sigmoid(),
        )
        do_xavier(self.color1)
        do_xavier(self.color2[0])
        do_xavier(self.color2[2])

    def forward(self, ray_bundle: RayBundle):
        # Get points
        points = ray_bundle.sample_points.view(-1, 3)

        # Embed them
        harm_xyz = self.harmonic_embedding_xyz(points)
        harm_dir = self.harmonic_embedding_dir(ray_bundle.directions)

        # Pass them through our networks
        first = self.start_layers(harm_xyz, harm_xyz)
        density = self.density_layers(first)
        features = self.color2((self.color1(first), harm_dir))

        # clean up!
        final = {
            "density": density,
            "feature": features,
        }

        return final


# https://github.com/ActiveVisionLab/nerfmm/blob/main/models/poses.py
class PoseModel(nn.Module):
    def __init__(self, num_cams, train_R, train_t, init_c2w=None):
        """
        :param num_cams:
        :param learn_R:  True/False
        :param learn_t:  True/False
        :param init_c2w: (N, 4, 4) torch tensor
        """
        super(PoseModel, self).__init__()
        self.num_cams = num_cams
        self.init_c2w = None
        if init_c2w is not None:
            self.init_c2w = pp.Parameter(init_c2w, requires_grad=False)

        self.delta = pp.Parameter(
            pp.identity_se3(num_cams), requires_grad=train_R
        )  # (N, 6)

    def forward(self, cam_id):
        pose = self.delta[cam_id].Exp()

        # learn a delta pose between init pose and target pose,
        # if a init pose is provided
        if self.init_c2w is not None:
            pose = self.init_c2w[cam_id] @ pose

        return pose
    
    def apply_delta(self):
        if self.init_c2w is not None:
            with torch.no_grad():
                self.init_c2w[:] = self.init_c2w @ self.delta.Exp()
                self.delta[:] = pp.identity_se3(self.num_cams)


class LinearSphereModel(nn.Module):
    def __init__(self, fov_degree, req_grad):

        super(LinearSphereModel, self).__init__()
        
        self.fov = nn.Parameter(torch.tensor(fov_degree), requires_grad=False)
        self.delta = nn.Parameter(
            torch.tensor(1.0, dtype=torch.float32), requires_grad=req_grad
        )  # (1, )


    def forward(self):
        # Rather than use additive delta here, NeRF-- uses a scale squared
        return self.fov * self.delta**2

    @property
    def model(self):
        fish = LinearSphere(
            fov_degree = self.forward(), 
            shape_struct = ShapeStruct(256, 256),
            in_to_tensor=False, 
            out_to_numpy=False)
        fish.device = self.delta.device
        return fish

volume_dict = {
    "nerf": NeuralRadianceField,
}
