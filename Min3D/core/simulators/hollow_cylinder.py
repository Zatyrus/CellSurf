## dependencies
import numpy as np
from typing import Tuple, Union, List

## custom dependencies
from Min3D.core.framework.point_cloud import PointCloud

__all__ = ["generate_hollow_cylinder"]


def generate_hollow_cylinder(
    center: Union[np.ndarray, Tuple[float, float, float], List[float]],
    inner_radius: float,
    outer_radius: float,
    height: float,
    num_points: int = 1000,
) -> PointCloud:
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
