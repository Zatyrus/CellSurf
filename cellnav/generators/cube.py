## dependencies
import numpy as np
from typing import Tuple, Union, List

## custom dependencies
from cellnav.core.framework.point_cloud import PointCloud

__all__ = ["generate_cube"]


def generate_cube(
    center: Union[np.ndarray, Tuple[float, float, float], List[float]],
    side_length: float,
    num_points: int = 1000,
) -> PointCloud:
    """
    Generate a point cloud representing a cube, which is created by sampling points from the surface of a cube defined by a center and a side length.
    The number of points in the cube can be specified.

    Args:
        center (Union[np.ndarray, Tuple[float, float, float], List[float]]): The center of the cube.
        side_length (float): The length of each side of the cube.
        num_points (int, optional): The number of points to generate. Defaults to 1000.

    Raises:
        ValueError: If the side_length is not positive.

    Returns:
        PointCloud: A point cloud representing the cube.
    """

    if side_length <= 0:
        raise ValueError("side_length must be positive")

    # Generate random points on the surface of a cube
    half_side = side_length / 2
    points = []

    for _ in range(num_points):
        face = np.random.randint(6)  # Randomly select one of the 6 faces
        match face:
            case 0:  # Front face
                x = np.random.uniform(-half_side, half_side)
                y = np.random.uniform(-half_side, half_side)
                z = half_side
            case 1:  # Back face
                x = np.random.uniform(-half_side, half_side)
                y = np.random.uniform(-half_side, half_side)
                z = -half_side
            case 2:  # Left face
                x = -half_side
                y = np.random.uniform(-half_side, half_side)
                z = np.random.uniform(-half_side, half_side)
            case 3:  # Right face
                x = half_side
                y = np.random.uniform(-half_side, half_side)
                z = np.random.uniform(-half_side, half_side)
            case 4:  # Top face
                x = np.random.uniform(-half_side, half_side)
                y = half_side
                z = np.random.uniform(-half_side, half_side)
            case 5:  # Bottom face
                x = np.random.uniform(-half_side, half_side)
                y = -half_side
                z = np.random.uniform(-half_side, half_side)
            case _:  # This should never happen
                raise ValueError("Invalid face index")

        points.append([x + center[0], y + center[1], z + center[2]])

    return PointCloud.from_numpy(np.array(points))
