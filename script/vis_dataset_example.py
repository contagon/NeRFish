'''
A quick script for iterating-over and visualizing the dataset.
'''

# General imports.
import os
import sys
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from colorama import Fore, Style

# Local imports.
sys.path.append('..')
from utils.dataset import TartanAirDataset

if __name__ == '__main__':
    # Create the dataset.
    tartanair_data_root = '../tartanair'
    traj_data_root = os.path.join(tartanair_data_root, 'Sewerage', 'Data_hard', 'P000')
    dataset = TartanAirDataset(traj_data_root=traj_data_root)

    # Create the dataloader.
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x: x)

    # Iterate over the dataset.
    for i, data in enumerate(dataloader):
        # Get the data.
        img, pose_gt = data[0]

        # Print the data.
        print('Image shape: {}'.format(img.shape))
        print('Pose: {}'.format(pose_gt))

        # Show the image.
        cv2.imshow('img', img)
        cv2.waitKey(0)

        # Break if we've reached the end.
        if i == 10:
            break