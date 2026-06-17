## dependencies
import numpy as np
from typing import Tuple, Union, List

## custom dependencies
from cellnav.core.framework.point_cloud import PointCloud

__all__ = ["generate_hollow_cylinder"]


def generate_hollow_cylinder(
    center: Union[np.ndarray, Tuple[float, float, float], List[float]],
    inner_radius: float,
    outer_radius: float,
    height: float,
    num_points: int = 1000,
) -> PointCloud:
    """Generate a point cloud representing a hollow cylinder with a given center, inner radius, outer radius, height, and number of points.

    Args:
        center (Union[np.ndarray, Tuple[float, float, float], List[float]]): The center of the cylinder.
        inner_radius (float): The inner radius of the cylinder.
        outer_radius (float): The outer radius of the cylinder.
        height (float): The height of the cylinder.
        num_points (int, optional): The number of points to generate on the cylinder's surface. Defaults to 1000.

    Raises:
        ValueError: If inner_radius or outer_radius is not positive.
        ValueError: If inner_radius is greater than or equal to outer_radius.

    Returns:
        PointCloud: A point cloud representing the hollow cylinder.
    """
    if inner_radius <= 0 or outer_radius <= 0:
        raise ValueError("inner_radius and outer_radius must be positive")

    if inner_radius >= outer_radius:
        raise ValueError("inner_radius must be less than outer_radius")

    # Generate random points on the surface of a hollow cylinder
    phi = np.random.uniform(0, 2 * np.pi, num_points)
    z = np.random.uniform(-height / 2, height / 2, num_points)

    # Randomly choose between inner and outer radius for each point
    radii = np.where(np.random.rand(num_points) < 0.5, inner_radius, outer_radius)

    x = radii * np.cos(phi) + center[0]
    y = radii * np.sin(phi) + center[1]
    z = z + center[2]

    points = np.stack((x, y, z), axis=1)
    return PointCloud.from_numpy(points)
