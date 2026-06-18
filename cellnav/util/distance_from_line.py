## dependencies
import numpy as np
from typing import Tuple, Union, List

## custom dependencies
from cellnav.core.framework.point_cloud import PointCloud

__all__ = ["distance_from_line"]


## get distance from central axis
def distance_from_line(
    point_cloud: PointCloud,
    axis_point: Union[Tuple[float, float, float], np.ndarray, List[float]] = (0, 0, 0),
    axis_direction: Union[Tuple[float, float, float], np.ndarray, List[float]] = (
        0,
        0,
        1,
    ),
) -> np.ndarray:
    """
    Calculate the distance from a given point cloud to a line defined by a point and a direction vector.
    The line is defined by an axis point and an axis direction vector.
    The distance is calculated as the shortest distance from each point in the point cloud to the line.

    Args:
        point_cloud (PointCloud): The point cloud from which to calculate distances.
        axis_point (Union[Tuple[float, float, float], np.ndarray, List[float]], optional): A point on the line. Defaults to (0, 0, 0).
        axis_direction (Union[Tuple[float, float, float], np.ndarray, List[float]], optional): The direction vector of the line. Defaults to (0, 0, 1).

    Returns:
        np.ndarray: An array of distances from each point in the point cloud to the line.
    """

    # Convert inputs to numpy arrays
    points = point_cloud.get_points()
    axis_point = np.array(axis_point)
    axis_direction = np.array(axis_direction)

    # Normalize the axis direction vector
    axis_direction = axis_direction / np.linalg.norm(axis_direction)

    # Vector from axis point to each point
    vec_to_points = points - axis_point

    # Project vec_to_points onto the axis direction
    projection_length = np.dot(vec_to_points, axis_direction)
    projection_vector = np.outer(projection_length, axis_direction)

    # Calculate the perpendicular vector from each point to the axis
    perpendicular_vector = vec_to_points - projection_vector

    # Calculate distances as the norm of the perpendicular vectors
    distances = np.linalg.norm(perpendicular_vector, axis=1)

    return distances
