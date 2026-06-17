## dependencies
import numpy as np
from typing import Tuple, Union, List

## custom dependencies
from cellnav.core.framework.point_cloud import PointCloud

__all__ = ["generate_gaussian_blob"]


def generate_gaussian_blob(
    center: Union[np.ndarray, Tuple[float, float, float], List[float]],
    sigma: float,
    num_points: int = 1000,
    scale_factor: float = 1.0,
) -> PointCloud:
    """Generate a point cloud representing a Gaussian blob, which is created by sampling points from a Gaussian distribution centered at a given point.

    Args:
        center (Union[np.ndarray, Tuple[float, float, float], List[float]]): The center of the Gaussian distribution.
        sigma (float): The standard deviation of the Gaussian distribution.
        num_points (int, optional): The number of points to generate. Defaults to 1000.
        scale_factor (float, optional): The factor by which to scale the generated points. Defaults to 1.0.

    Raises:
        ValueError: If sigma is negative.

    Returns:
        PointCloud: A point cloud representing the Gaussian blob.
    """

    if sigma < 0:
        raise ValueError("sigma must be non-negative")

    # Generate random points around the center
    points = np.random.normal(loc=center, scale=sigma, size=(num_points, 3))

    # Scale the points
    points *= scale_factor

    # Create a PointCloud object
    point_cloud = PointCloud.from_numpy(points)

    return point_cloud
