## dependencies
import numpy as np
from typing import Tuple, Union, List

## custom dependencies
from Min3D.core.framework.point_cloud import PointCloud

__all__ = ["generate_gaussian_blob"]


def generate_gaussian_blob(
    center: Union[np.ndarray, Tuple[float, float, float], List[float]],
    sigma: float,
    num_points: int = 1000,
    scale_factor: float = 1.0,
) -> PointCloud:
    # Generate random points around the center
    points = np.random.normal(loc=center, scale=sigma, size=(num_points, 3))

    # Scale the points
    points *= scale_factor

    # Create a PointCloud object
    point_cloud = PointCloud.from_numpy(points)

    return point_cloud
