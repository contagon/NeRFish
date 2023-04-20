import torch
import torch.nn.functional as F

# Convenience class wrapping several ray inputs:
#   1) Origins -- ray origins
#   2) Directions -- ray directions
#   3) Sample points -- sample points along ray direction from ray origin
#   4) Sample lengths -- distance of sample points from ray origin


class RayBundle(object):
    def __init__(
        self,
        origins,
        directions,
        sample_points,
        sample_lengths,
    ):
        self.origins = origins
        self.directions = directions
        self.sample_points = sample_points
        self.sample_lengths = sample_lengths

    def __getitem__(self, idx):
        return RayBundle(
            self.origins[idx],
            self.directions[idx],
            self.sample_points[idx],
            self.sample_lengths[idx],
        )

    @property
    def shape(self):
        return self.origins.shape[:-1]

    @property
    def sample_shape(self):
        return self.sample_points.shape[:-1]

    def reshape(self, *args):
        return RayBundle(
            self.origins.reshape(*args, 3),
            self.directions.reshape(*args, 3),
            self.sample_points.reshape(*args, self.sample_points.shape[-2], 3),
            self.sample_lengths.reshape(*args, self.sample_lengths.shape[-2], 1),
        )

    def view(self, *args):
        return RayBundle(
            self.origins.view(*args, 3),
            self.directions.view(*args, 3),
            self.sample_points.view(*args, self.sample_points.shape[-2], 3),
            self.sample_lengths.view(*args, self.sample_lengths.shape[-2], 1),
        )

    def _replace(self, **kwargs):
        for key in kwargs.keys():
            setattr(self, key, kwargs[key])

        return self


# Sample image colors from pixel values
def sample_images_at_xy(
    images: torch.Tensor,
    xy_grid: torch.Tensor,
):
    batch_size = images.shape[0]
    images.shape[1:-1]

    xy_grid = -xy_grid.view(batch_size, -1, 1, 2)

    images_sampled = torch.nn.functional.grid_sample(
        images.permute(0, 3, 1, 2),
        xy_grid,
        align_corners=True,
        mode="bilinear",
    )

    return images_sampled.permute(0, 2, 3, 1).view(-1, images.shape[-1])


# Generate pixel coordinates from in NDC space (from [-1, 1])
def get_pixels_from_image(camera, filter_valid=True):
    # W, H = image_size[0], image_size[1]

    # # Generate pixel coordinates from [0, W] in x and [0, H] in y

    # # Convert to the range [-1, 1] in both x and y
    # x = torch.linspace(-1, 1, W)
    # y = torch.linspace(-1, 1, H)

    # # Create grid of coordinates
    # xy_grid = torch.stack(
    #     tuple(reversed(torch.meshgrid(y, x))),
    #     dim=-1,
    # ).view(W * H, 2)

    # return -xy_grid
    valid_mask = camera.get_valid_mask()

    pixel_coords = camera.pixel_coordinates(shift=0, normalized=False, flatten=False)
    pixel_coords =  pixel_coords.to(dtype=torch.uint8)

    # Same but normalized to [-1, 1] range. Shape: (2, W*H)
    xy_coords = camera.pixel_coordinates(shift=0, normalized=True, flatten=False)

    # Get valid pixels.
    if filter_valid:
        valid_pixel_coords = pixel_coords[:, valid_mask == 1]
        valid_xy_coords = xy_coords[:, valid_mask == 1]
    else:
        valid_pixel_coords = pixel_coords
        valid_xy_coords = xy_coords

    valid_pixel_coords = valid_pixel_coords.view(2, -1).cuda()
    valid_xy_coords = valid_xy_coords.view(2, -1).cuda()

    return valid_pixel_coords, valid_xy_coords



# Random subsampling of pixels from an image
def get_random_pixels_from_image(n_pixels, image_size, camera):
    # xy_grid = get_pixels_from_image(image_size, camera)

    # # Random subsampling of pixel coordinates
    # rand_idx = torch.randperm(xy_grid.shape[0])
    # xy_grid_sub = xy_grid[rand_idx[:n_pixels]].cuda()

    # # Return
    # return xy_grid_sub.reshape(-1, 2)[:n_pixels]
    valid_mask = camera.get_valid_mask()

    # Tensor of pixels coordinates (x, y), of shape (2, W, H). Shape: (2, W*H). This is where we can randomly choose pixels to create NeRF training batches.
    pixel_coords = camera.pixel_coordinates(shift=0, normalized=False, flatten=False)
    pixel_coords =  pixel_coords.to(dtype=torch.uint8)

    # Same but normalized to [-1, 1] range. Shape: (2, W*H)
    xy_coords = camera.pixel_coordinates(shift=0, normalized=True, flatten=False)

    # Get valid pixels.
    valid_pixel_coords = pixel_coords[:, valid_mask == 1]
    valid_xy_coords = xy_coords[:, valid_mask == 1]

    # Random subsampling of pixel coordinates.
    rand_idx = torch.randperm(valid_pixel_coords.shape[1])
    xy_grid_sub = valid_xy_coords[:, rand_idx[:n_pixels]].cuda()
    coords_sub = valid_pixel_coords[:, rand_idx[:n_pixels]].cuda()

    # Return
    return coords_sub, xy_grid_sub # Shapes are (2, n_pixels) and (2, n_pixels)


    


# Get rays from pixel values.
def get_rays_from_pixels(pixel_coords, camera, X_ned_cam, camera_pose_ned, debug=False):
    """Get rays from pixel values.

    This function takes in pixel coordinates in range ([0, W], [0, H]), organized in the shape (2, N), where N is the number of pixels. It returns a RayBundle object, which contains the following attributes:
        - origins: (N, 3) tensor of ray origins
        - directions: (N, 3) tensor of ray directions
        - sample_points: (N, S, 3) tensor of sample points along the ray. The value here is zeros, as no samples are made yet.
        - sample_lengths: (N, S, 1) tensor of sample lengths along the ray. The value here is zeros, as no samples are made yet.

    """

    # Map pixels to 3D points on the unit sphere (otherwise known as unit vectors.) These are in the camera's coordinate system (z forward, x right, y down). The output is of shape (3, N), where N is the number of pixels.
    x_cam_rays, valid_mask = camera.pixel_2_ray(pixel_coords)

    # Transform the rays from the camera's coordinate system to the base coordinate system, which is NED.
    # Rays in the base frame NED.
    X_ned_cam = X_ned_cam.to(x_cam_rays.device)
    x_base_rays = X_ned_cam @ x_cam_rays.T
    x_base_rays = x_base_rays.to(device=x_cam_rays.device)
       
    # Get ray origins from camera center.
    rays_o = camera_pose_ned.translation()
    rays_o = rays_o.repeat(x_base_rays.shape[0], 1)

    # Rotate the rays by the base rotation, which would yield their directions in the world frame (yes it is still NED). 
    # First, convert the pose to a transformation matrix.
    X_world_base = camera_pose_ned
    R_world_base = X_world_base.rotation()
    rays_d = R_world_base @ x_base_rays

    if debug:
        # Visualize the rays.   
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x_cam_rays.cpu()[0], x_cam_rays.cpu()[1], x_cam_rays.cpu()[2], c='r', marker='o')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        # Create axes at the camera center.
        ax.quiver(0, 0, 0, 1, 0, 0, color='r', arrow_length_ratio=0.1, pivot='tail', linewidth=5)
        ax.quiver(0, 0, 0, 0, 1, 0, color='g', arrow_length_ratio=0.1, pivot='tail', linewidth=5)
        ax.quiver(0, 0, 0, 0, 0, 1, color='b', arrow_length_ratio=0.1, pivot='tail', linewidth=5)
        ax.title.set_text('Rays in the camera frame (Z forward, X right, Y down)')
        # Visualize the rays.   
        fig2 = plt.figure()
        ax = fig2.add_subplot(111, projection='3d')
        ax.scatter(x_base_rays.cpu()[0], x_base_rays.cpu()[1], x_base_rays.cpu()[2], c='r', marker='o')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        # Create axes at the camera center.
        ax.quiver(0, 0, 0, 1, 0, 0, color='r', arrow_length_ratio=0.1, pivot='tail', linewidth=5)
        ax.quiver(0, 0, 0, 0, 1, 0, color='g', arrow_length_ratio=0.1, pivot='tail', linewidth=5)
        ax.quiver(0, 0, 0, 0, 0, 1, color='b', arrow_length_ratio=0.1, pivot='tail', linewidth=5)
        ax.title.set_text('Rays in the base frame (NED)')
        # Visualize the rays.
        fig3 = plt.figure()
        ax = fig3.add_subplot(111, projection='3d')
        ax.scatter(rays_d.cpu()[:,0], rays_d.cpu()[:,1], rays_d.cpu()[:,2], c='r', marker='o')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        # Create axes at the camera center.
        ax.quiver(0, 0, 0, 1, 0, 0, color='r', arrow_length_ratio=0.1, pivot='tail', linewidth=5)
        ax.quiver(0, 0, 0, 0, 1, 0, color='g', arrow_length_ratio=0.1, pivot='tail', linewidth=5)
        ax.quiver(0, 0, 0, 0, 0, 1, color='b', arrow_length_ratio=0.1, pivot='tail', linewidth=5)
        ax.title.set_text('Rays in the world frame (NED). For illustration purposes only, the world origin coincides in translation with the base origin.')
        plt.show()

    # Create and return RayBundle
    return RayBundle(
        rays_o,
        rays_d,
        torch.zeros_like(rays_o).unsqueeze(1),
        torch.zeros_like(rays_o).unsqueeze(1),
    )
