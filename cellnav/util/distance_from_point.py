## dependencies
import numpy as np
from typing import Tuple, Union, List

## custom dependencies
from cellnav.core.framework.point_cloud import PointCloud

__all__= ["distance_from_point"]


def distance_from_point(
    point_cloud: PointCloud,
    point: Union[Tuple[float, float, float], np.ndarray, List[float]],
) -> np.ndarray:
    """Calculate the distance from a given point to each point in a point cloud.

    Args:
        point_cloud (PointCloud): The point cloud from which to calculate distances.
        point (Union[Tuple[float, float, float], np.ndarray, List[float]]): A point (x, y, z) to which to calculate distances.

    Returns:
        np.ndarray: A numpy array of distances from the given point to each point in the point cloud.
    """

    # Get the points from the PointCloud object
    points = point_cloud.get_points()

    # Calculate the distance from the given point to each point in the point cloud
    distances = np.linalg.norm(points - point, axis=1)

    return distances
