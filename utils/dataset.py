# General imports.
import os
import sys
import numpy as np
import torch
from torchvision import io
from torchvision import transforms, io
from colorama import Fore, Style
from scipy.spatial.transform import Rotation

'''
A Torch dataset for TartanAir-style data. It is a very simple implementation that can load only one trajectory at a time.
'''
class TartanAirDataset(torch.utils.data.Dataset):
    def __init__(self, traj_data_root, image_shape=[256, 256], device='cuda'):

        # Set member variables.
        self.num_frames = len(os.listdir(os.path.join(traj_data_root, 'image_lcam_fish')))
        print(Fore.GREEN + 'Found {} frames.'.format(self.num_frames) + Style.RESET_ALL)
        
        # Dataset root.
        self.traj_data_root = traj_data_root

        # Image shape.
        self.image_shape = image_shape

        # Device to return everything on
        self.device = device
        
        # Get all the image poses up to memory for easy lookup.
        # Note(yoraish): poses are in the form of [x, y, z, qx, qy, qz, qw]. Those are in the robot frame, which is NED (x-forward, y-right, z-down). Confusingly, the camera frame is z-forward, x-right, y-down. I am unsure if we need to make a distinction here, but I am leaving this note here for now.
        poses_np = np.loadtxt(os.path.join(traj_data_root, 'pose_lcam_fish.txt'))
        self.poses_gt = torch.zeros((poses_np.shape[0],4,4)).to(self.device)
        self.poses_gt[:,-1,-1] = 1
        for i, pose in enumerate(poses_np):
            self.poses_gt[i,:3, 3] = torch.tensor(pose[:3])
            self.poses_gt[i,:3, :3] = torch.tensor(Rotation.from_quat(pose[3:]).as_matrix())


    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx):
        # Get the image.
        img_path = os.path.join(self.traj_data_root, 'image_lcam_fish', '{:06d}.png'.format(idx))
        # print(Fore.GREEN + 'Loading image from {}'.format(os.path.abspath(img_path)) + Style.RESET_ALL)
        img = transforms.functional.convert_image_dtype(
            io.read_image(img_path, io.ImageReadMode.RGB).unsqueeze(0),
            torch.float32
        )

        # Resize the image.


        # Get the pose.
        pose_gt = self.poses_gt[idx]

        return idx, img.to(self.device), pose_gt

def get_dataset(traj_data_root, image_shape):
    '''
    Returns two datasets, one for training and one for validation.
    '''
    dataset = TartanAirDataset(traj_data_root=traj_data_root, image_shape=image_shape)

    # TODO(yoraish): Split into train and validation.
    return dataset, dataset


def trivial_collate(batch):
    """
    A trivial collate function that merely returns the uncollated batch.
    """
    return batch
