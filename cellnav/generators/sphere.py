## depedencies
import numpy as np
from typing import Tuple, Union, List

## custom dependencies
from cellnav.core.framework.point_cloud import PointCloud

__all__ = ["generate_sphere"]


def generate_sphere(
    center: Union[np.ndarray, Tuple[float, float, float], List[float]],
    radius: float,
    num_points: int = 1000,
) -> PointCloud:
    """Generate a point cloud representing a sphere with a given center, radius, and number of points.

    Args:
        center (Union[np.ndarray, Tuple[float, float, float], List[float]]): The center of the sphere.
        radius (float): The radius of the sphere.
        num_points (int, optional): The number of points to generate on the sphere's surface. Defaults to 1000.

    Returns:
        PointCloud: A point cloud representing the sphere.
    """

    # Generate random points on the surface of a sphere
    phi = np.random.uniform(0, 2 * np.pi, num_points)

    # we do this, instead of sampling theta uniformly, to ensure an even distribution of points on the sphere's surface
    # otherswise, we would get a higher density of points near the poles (theta = 0 and theta = pi)
    arc_cos = np.random.uniform(
        -1, 1, num_points
    )  # Uniform distribution for cos(theta)
    theta = np.arccos(arc_cos)  # Convert to theta using arcc

    x = radius * np.sin(theta) * np.cos(phi) + center[0]
    y = radius * np.sin(theta) * np.sin(phi) + center[1]
    z = radius * np.cos(theta) + center[2]

    points = np.stack((x, y, z), axis=1)
    return PointCloud.from_numpy(points)
