## dependencies
import numpy as np
from typing import Tuple, Union, List

## custom dependencies
from cellnav.core.framework.point_cloud import PointCloud

__all__ = ["generate_cylinder"]


def generate_cylinder(
    center: Union[np.ndarray, Tuple[float, float, float], List[float]],
    radius: float,
    height: float,
    num_points: int = 1000,
    no_faces: bool = False,
    face_to_surface_ratio: Tuple[float, float] = (0.30, 0.70),
) -> PointCloud:
    """
    Generate a point cloud representing a cylinder with a given center, radius, height, and number of points.
    The cylinder can be generated with or without faces, and the ratio of points on the faces to points

    Args:
        center (Union[np.ndarray, Tuple[float, float, float], List[float]]): The center of the cylinder.
        radius (float): The radius of the cylinder.
        height (float): The height of the cylinder.
        num_points (int, optional): The number of points to generate. Defaults to 1000.
        no_faces (bool, optional): Whether to generate points only on the curved surface. Defaults to False.
        face_to_surface_ratio (Tuple[float, float], optional): The ratio of points on the faces to points on the surface. Defaults to (0.30, 0.70).

    Raises:
        ValueError: If the radius is not positive.
        ValueError: If the height is not positive.
        ValueError: If the face_to_surface_ratio does not sum to 1.0.

    Returns:
        PointCloud: A point cloud representing the cylinder.
    """
    if radius <= 0:
        raise ValueError("radius must be positive")

    if height <= 0:
        raise ValueError("height must be positive")

    # Generate random points on the surface of a cylinder
    if no_faces:
        phi = np.random.uniform(0, 2 * np.pi, num_points)
        z = np.random.uniform(-height / 2, height / 2, num_points)

        x = radius * np.cos(phi) + center[0]
        y = radius * np.sin(phi) + center[1]
        z = z + center[2]

        return PointCloud.from_numpy(np.stack((x, y, z), axis=1))

    else:
        # make sure the face_to_surface_ratio sums to 1.0
        assert sum(face_to_surface_ratio) == 1.0, (
            "face_to_surface_ratio must sum to 1.0"
        )

        points = []
        while len(points) < num_points:
            state = np.random.choice(
                [0, 1, 2],
                p=[
                    face_to_surface_ratio[1],
                    face_to_surface_ratio[0] / 2,
                    face_to_surface_ratio[0] / 2,
                ],
            )  # 70% curved surface, 15% top face, 15% bottom face
            match state:
                case 0:  # Curved surface
                    phi = np.random.uniform(0, 2 * np.pi)
                    z = np.random.uniform(-height / 2, height / 2)

                    x = radius * np.cos(phi) + center[0]
                    y = radius * np.sin(phi) + center[1]
                    z = z + center[2]

                case 1:  # Top face
                    xy = np.random.uniform(-radius, radius, size=2)
                    x = xy[0] + center[0]
                    y = xy[1] + center[1]
                    z = height / 2 + center[2]

                    if (
                        not np.sqrt(xy[0] ** 2 + xy[1] ** 2) <= radius
                    ):  # Ensure the point is within the circular face
                        continue

                case 2:  # Bottom face
                    xy = np.random.uniform(-radius, radius, size=2)
                    x = xy[0] + center[0]
                    y = xy[1] + center[1]
                    z = -height / 2 + center[2]

                    if (
                        not np.sqrt(xy[0] ** 2 + xy[1] ** 2) <= radius
                    ):  # Ensure the point is within the circular face
                        continue
                case _:  # This should never happen
                    raise ValueError("Invalid state index")

            points.append([x, y, z])

    return PointCloud.from_numpy(np.array(points))
