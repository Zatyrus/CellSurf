## dependencies
import numpy as np
from typing import Tuple, Union, List

## custom dependencies
from cellnav.core.framework.point_cloud import PointCloud
from cellnav.generators.sphere import generate_sphere

__all__ = ["generate_gaussian_corona"]


def generate_gaussian_corona(
    center: Union[np.ndarray, Tuple[float, float, float], List[float]],
    radius: float,
    sigma: float,
    num_points: int = 1000,
) -> PointCloud:
    """
    Generate a point cloud representing a Gaussian corona, which is created by adding Gaussian noise to points on the surface of a sphere.
    The sphere is defined by a center and a radius, and the noise is defined by a standard deviation (sigma).
    The number of points in the corona can be specified.

    Args:
        center (Union[np.ndarray, Tuple[float, float, float], List[float]]): The center of the sphere from which to generate the corona.
        radius (float): The radius of the sphere from which to generate the corona.
        sigma (float): The standard deviation of the Gaussian noise to add.
        num_points (int, optional): The number of points to generate in the corona. Defaults to 1000.

    Raises:
        ValueError: If sigma is negative.
        ValueError: If radius is not positive.

    Returns:
        PointCloud: A point cloud representing the Gaussian corona.
    """

    if sigma < 0:
        raise ValueError("sigma must be non-negative")

    if radius <= 0:
        raise ValueError("radius must be positive")

    # generate points on the surface of a sphere
    sphere_points = generate_sphere(center, radius, num_points).get_points()

    # add Gaussian noise to the points
    noise = np.random.normal(loc=0, scale=sigma, size=sphere_points.shape)
    corona_points = sphere_points + noise

    # return a PointCloud object
    return PointCloud.from_numpy(corona_points)
