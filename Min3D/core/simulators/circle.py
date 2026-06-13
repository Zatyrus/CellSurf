## dependencies
import numpy as np
from typing import Tuple, Union, List

## custom dependencies
from Min3D.core.framework.point_cloud import PointCloud

__all__ = ["generate_circle"]


def generate_circle(
    center: Union[np.ndarray, Tuple[float, float, float], List[float]],
    radius: float,
    num_points: int = 1000,
) -> PointCloud:
    # Generate random points on the circumference of a circle
    phi = np.random.uniform(0, 2 * np.pi, num_points)

    x = radius * np.cos(phi) + center[0]
    y = radius * np.sin(phi) + center[1]
    z = np.full_like(x, center[2])  # All points have the same z-coordinate

    points = np.stack((x, y, z), axis=1)
    return PointCloud.from_numpy(points)
