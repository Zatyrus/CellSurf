## dependencies
import numpy as np
from typing import Tuple, Union, List

## custom dependencies
from cellnav.core.framework.point_cloud import PointCloud

__all__ = ["generate_circle"]


def generate_circle(
    center: Union[np.ndarray, Tuple[float, float, float], List[float]],
    radius: float,
    num_points: int = 1000,
) -> PointCloud:
    """Generate a point cloud representing a circle with a given center, radius, and number of points.

    Raises:
        ValueError: If the radius is not positive.

    Returns:
        _type_: A point cloud representing the circle.
    """

    if radius <= 0:
        raise ValueError("radius must be positive")

    # Generate random points on the circumference of a circle
    phi = np.random.uniform(0, 2 * np.pi, num_points)

    x = radius * np.cos(phi) + center[0]
    y = radius * np.sin(phi) + center[1]
    z = np.full_like(x, center[2])  # All points have the same z-coordinate

    points = np.stack((x, y, z), axis=1)
    return PointCloud.from_numpy(points)
