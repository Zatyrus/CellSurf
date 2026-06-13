## dependencies
import numpy as np
from typing import Tuple, Union, List

## custom dependencies
from Min3D.core.framework.point_cloud import PointCloud

__all_ = ["distance_from_point"]


def distance_from_point(
    point_cloud: PointCloud,
    point: Union[Tuple[float, float, float], np.ndarray, List[float]],
) -> np.ndarray:
    """
    Calculate the distance from a given point to each point in the point cloud.

    Parameters:
    - point_cloud: A PointCloud object containing the points.
    - point: A 3D point (x, y, z) to which distances will be calculated.

    Returns:
    - distances: A numpy array of distances from the given point to each point in the point cloud.
    """
    # Get the points from the PointCloud object
    points = point_cloud.get_points()

    # Calculate the distance from the given point to each point in the point cloud
    distances = np.linalg.norm(points - point, axis=1)

    return distances
