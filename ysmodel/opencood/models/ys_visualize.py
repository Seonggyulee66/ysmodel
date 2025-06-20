import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

class Visualization(object):
    def __init__(self):
        """
        Initialize the visualization with means and covariances.
        Args:
            means: [B, H, W, 3] tensor
            covs: [B, H, W, 3, 3] tensor
            skip: sampling stride (to avoid overplotting)
            batch_index: which batch to draw
            scale: scale for ellipse size (1-sigma)
        """
        

    def draw_gaussian_ellipses_2d(self, means, covs, skip=1, batch_index=0, scale=1):
        """
        Visualize 2D projection of 3D Gaussians as ellipses on XY plane.
        Args:
            means: [B, H, W, 3] tensor
            covs: [B, H, W, 3, 3] tensor
            skip: sampling stride (to avoid overplotting)
            batch_index: which batch to draw
            scale: scale for ellipse size (1-sigma)
        """
        B, H, W, _ = means.shape
        mu = means[batch_index].detach().cpu().numpy()     # [H, W, 3]
        cov = covs[batch_index].detach().cpu().numpy()     # [H, W, 3, 3]

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_title(f"2D Gaussian Ellipses (Batch {batch_index})")
        ax.set_aspect('equal')

        for i in range(0, H, skip):
            for j in range(0, W, skip):
                x, y = mu[i, j, 0], mu[i, j, 1]
                cov_2d = cov[i, j, :2, :2]  # extract 2×2 xy submatrix
                try:
                    # Eigen-decomposition for ellipse axes
                    vals, vecs = np.linalg.eigh(cov_2d)
                    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
                    width, height = 2 * scale * np.sqrt(np.maximum(vals, 0))
                    ellipse = Ellipse((x, y), width, height, angle, edgecolor='blue', facecolor='none', lw=1)
                    ax.add_patch(ellipse)
                except np.linalg.LinAlgError:
                    continue

        ax.set_xlim(mu[:, :, 0].min(), mu[:, :, 0].max())
        ax.set_ylim(mu[:, :, 1].min(), mu[:, :, 1].max())
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        plt.savefig(f"/home/rina/SG/savefigs/gaussian_ellipses_batch_{batch_index}.png")

