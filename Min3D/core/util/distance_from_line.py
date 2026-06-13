## dependencies
import numpy as np
from typing import Tuple, Union, List

## custom dependencies
from Min3D.core.framework.point_cloud import PointCloud

__all_ = ["distance_from_axis"]


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
    Calculate the distance of each point from a specified axis defined by a point and a direction vector.

    Parameters:
    - point_cloud: A PointCloud object containing the points.
    - axis_point: A tuple representing a point on the axis (default is the origin).
    - axis_direction: A tuple representing the direction vector of the axis (default is along the z-axis).

    Returns:
    - distances: A numpy array of shape (N,) containing the distance of each point from the axis.
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
