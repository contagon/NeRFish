import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
import pypose as pp
import sys

sys.path.append('/home/contagon/Classes/nerfish')
from utils.dataset import get_dataset

def setup_plot():
    sns.set_style("whitegrid")
    sns.set_palette('deep')
    sns.set_context("notebook")
    return sns.color_palette('deep')


def plot_params(loss, fov, file):
    c = setup_plot()

    if file is None:
        file = "media/loss_fov.png"

    fig, ax = plt.subplots(1,2, layout="constrained", figsize=(8,3))
    true_fov = 195

    ax[0].plot(loss)
    ax[1].plot(fov, label="Learned")
    ax[1].plot(np.full_like(fov, true_fov), c=c[7], label="GT")
    ax[1].legend()

    ax[0].set_xlabel("Epochs")
    ax[1].set_xlabel("Epochs")

    ax[0].set_title("Loss")
    ax[1].set_title("Field of View (degrees)")

    plt.savefig(file)

def animate_pose(pose, gt, file, n):
    c = setup_plot()

    if file is None:
        file = "media/animate_poses.gif"
    if n == 0:
        n = pose.shape[0]

    fig, ax = plt.subplots(1, 1, layout="constrained")
    
    # Plot ground truth
    trans_gt = gt[:,:3]
    ax.plot(trans_gt[:,0], trans_gt[:,1], marker='.', c=c[7], label="GT")

    # Plot our estimate
    trans_est = pose[:n,:,:3]
    line_est, = ax.plot(trans_est[0,:,0], trans_est[0,:,1], marker='.', label="Learned")

    # Clean up axes
    ax.set_title("Frame Poses")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.axis('equal')
    ax.legend()

    def animate(i):
        line_est.set_data(trans_est[i,:,0], trans_est[i,:,1])

    ani = animation.FuncAnimation(fig, animate, frames=trans_est.shape[0], interval=10, repeat=True)
    ani.save(file, writer=animation.PillowWriter())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--data', type=str, default='checkpoints_data.npz')
    parser.add_argument('-o', '--outfile', type=str, default=None)
    parser.add_argument('--kind', type=str, help='Either pose or params')
    parser.add_argument('-n', '--num', type=int, default=0, help='How many epochs to render')

    args = parser.parse_args()

    data = np.load(args.data)

    if args.kind == "params":
        plot_params(data["loss"], data["fov"], args.outfile)
    
    if args.kind == "pose":
        gt = get_dataset(str(data["dataset"]))[0].poses_gt.cpu().tensor()
        animate_pose(data["pose"], gt, args.outfile, args.num)