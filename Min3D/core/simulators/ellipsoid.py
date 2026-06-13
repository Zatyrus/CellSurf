## dependencies
import numpy as np
from typing import Tuple, Union, List

## custom dependencies
from Min3D.core.framework.point_cloud import PointCloud

__all__ = ["generate_ellipsoid"]


def generate_ellipsoid(
    center: Union[np.ndarray, Tuple[float, float, float], List[float]],
    radii: Union[np.ndarray, Tuple[float, float, float], List[float]],
    num_points: int = 1000,
) -> PointCloud:
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
