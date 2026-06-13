## dependencies
import numpy as np

## custom dependencies
from Min3D.core.framework.point_cloud import PointCloud

__all__ = ["add_gaussian_noise"]


def add_gaussian_noise(
    point_cloud: PointCloud,
    sigma: float,
) -> PointCloud:
    # Get the points from the PointCloud object
    points = point_cloud.get_points()

    # Add Gaussian noise to the points
    noise = np.random.normal(loc=0, scale=sigma, size=points.shape)
    noisy_points = points + noise

    # Create a new PointCloud object with the noisy points
    return PointCloud.from_numpy(noisy_points)
