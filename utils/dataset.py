# General imports.
import os
import sys
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from colorama import Fore, Style

'''
A Torch dataset for TartanAir-style data. It is a very simple implementation that can load only one trajectory at a time.
'''
class TartanAirDataset(torch.utils.data.Dataset):
    def __init__(self, traj_data_root):
        # Set member variables.
        self.num_frames = len(os.listdir(os.path.join(traj_data_root, 'image_lcam_fish')))
        print(Fore.GREEN + 'Found {} frames.'.format(self.num_frames) + Style.RESET_ALL)
        self.traj_data_root = traj_data_root

        # Get all the image poses up to memory for easy lookup.
        # Note(yoraish): poses are in the form of [x, y, z, qx, qy, qz, qw]. Those are in the robot frame, which is NED (x-forward, y-right, z-down). Confusingly, the camera frame is z-forward, x-right, y-down. I am unsure if we need to make a distinction here, but I am leaving this note here for now.
        self.poses_gt = [] 
        with open(os.path.join(traj_data_root, 'pose_lcam_fish.txt')) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.poses_gt.append(np.array([float(x) for x in line.split(' ')]))


    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx):
        # Get the image.
        img_path = os.path.join(self.traj_data_root, 'image_lcam_fish', '{:06d}.png'.format(idx))
        print(Fore.GREEN + 'Loading image from {}'.format(os.path.abspath(img_path)) + Style.RESET_ALL)
        img = cv2.imread(img_path)

        # Get the pose.
        pose_gt = self.poses_gt[idx]

        return img, pose_gt



# TODO: Dataset loading code
def trivial_collate(batch):
    """
    A trivial collate function that merely returns the uncollated batch.
    """
    return batch
