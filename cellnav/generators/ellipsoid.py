## dependencies
import numpy as np
from typing import Tuple, Union, List

## custom dependencies
from cellnav.core.framework.point_cloud import PointCloud

__all__ = ["generate_ellipsoid"]


def generate_ellipsoid(
    center: Union[np.ndarray, Tuple[float, float, float], List[float]],
    radii: Union[np.ndarray, Tuple[float, float, float], List[float]],
    num_points: int = 1000,
) -> PointCloud:
    """
    Generate a point cloud representing an ellipsoid, which is created by sampling points from the surface of an
    ellipsoid defined by a center and radii along the x, y, and z axes. The number of points in the ellipsoid can be specified.

    Args:
        center (Union[np.ndarray, Tuple[float, float, float], List[float]]): The center of the ellipsoid.
        radii (Union[np.ndarray, Tuple[float, float, float], List[float]]): The radii of the ellipsoid along the x, y, and z axes.
        num_points (int, optional): The number of points to generate on the surface of the ellipsoid. Defaults to 1000.

    Raises:
        ValueError: If any of the radii are not positive.

    Returns:
        PointCloud: A point cloud representing the ellipsoid.
    """

    if any(r <= 0 for r in radii):
        raise ValueError("All radii must be positive")

    # Generate random points on the surface of an ellipsoid
    phi = np.random.uniform(0, 2 * np.pi, num_points)
    arc_cos = np.random.uniform(
        -1, 1, num_points
    )  # Uniform distribution for cos(theta)
    theta = np.arccos(arc_cos)  # Convert to theta using arccos

    x = radii[0] * np.sin(theta) * np.cos(phi) + center[0]
    y = radii[1] * np.sin(theta) * np.sin(phi) + center[1]
    z = radii[2] * np.cos(theta) + center[2]

    points = np.stack((x, y, z), axis=1)
    return PointCloud.from_numpy(points)
