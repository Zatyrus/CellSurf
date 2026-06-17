## dependencies
import numpy as np

## custom dependencies
from cellnav.core.framework.point_cloud import PointCloud

__all__ = ["add_gaussian_noise"]


def add_gaussian_noise(
    point_cloud: PointCloud,
    sigma: float,
) -> PointCloud:
    """
    Add Gaussian noise to a given point cloud object.
    The noise is generated with a mean of 0 and a standard deviation of sigma, and is added to each point in the point cloud.
    This is similar to pushing the points in the point cloud in random directions, with the magnitude of the push determined by the standard deviation of the noise.

    Args:
        point_cloud (PointCloud): The point cloud to which to add noise.
        sigma (float): The standard deviation of the Gaussian noise.

    Raises:
        ValueError: If sigma is negative.

    Returns:
        PointCloud: A new point cloud with Gaussian noise added.
    """

    if sigma < 0:
        raise ValueError("sigma must be non-negative")

    # Get the points from the PointCloud object
    points = point_cloud.get_points()

    # Add Gaussian noise to the points
    noise = np.random.normal(loc=0, scale=sigma, size=points.shape)
    noisy_points = points + noise

    # Create a new PointCloud object with the noisy points
    return PointCloud.from_numpy(noisy_points)
