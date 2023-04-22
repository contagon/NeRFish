'''
A quick script for visualizing the evolution of the FoV estimation in the nerf pipeline.

The produced figure has two curves:
1. Horizontal line marking the FoV of the ground truth camera model.
2. The estimated FoV of the nerf pipeline.
'''

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Local imports.
from figure_maker import FigureMaker

class FoVEvolutionFigureMaker(FigureMaker):
    def __init__(self):
        super().__init__()

    def make_figure(self, experiment_dir, save = True):

        # Path to the FoV estimation file.
        fov_estimation_file = os.path.join(experiment_dir, 'fov_est.npy')

        # Load the FoV estimation.
        fov_est = np.load(fov_estimation_file)

        # Plot the FoV estimation.
        x = fov_est[:, 0]
        y = fov_est[:, 1]

        # Plot smoothed y with std shadow.
        y_smooth = np.convolve(y, np.ones((300,))/300, mode='valid')
        
        # Running window of 10 for std.
        y_std = np.zeros_like(y_smooth)
        for i in range(len(y_smooth)):
            y_std[i] = np.std(y[i:i+5000])

        # Set the figure size.
        plt.figure(figsize=(10, 4))
        plt.plot(x[:len(y_smooth)], y_smooth, label='Estimated FoV')
        # plt.fill_between(x[:len(y_smooth)], y_smooth - y_std, y_smooth + y_std, alpha=0.2)
        plt.plot(x[:len(y_smooth)], y[:len(y_smooth)], alpha=0.52, label='Estimated FoV (raw)')

        # Plot the ground truth FoV.
        plt.axhline(y=195, linestyle='--', color='black', label='Ground truth FoV')

        # Set the legend.
        plt.legend()

        # Set the labels.
        plt.xlabel('Iteration')
        plt.ylabel('FoV (degrees)')
        plt.title('FoV evolution')

        # Save the figure.
        if save:
            plt.savefig(os.path.join(experiment_dir, 'fov_evolution.png'))
        else:
            plt.show()

if __name__ == '__main__':
    # Create the figure maker.
    fov_figure_maker = FoVEvolutionFigureMaker()

    # Get some arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_dir', '-e', type=str, required=True)
    parser.add_argument('--save', '-s', action='store_true')
    args = parser.parse_args()

    # Make the figure.
    fov_figure_maker.make_figure(args.experiment_dir, args.save)
    