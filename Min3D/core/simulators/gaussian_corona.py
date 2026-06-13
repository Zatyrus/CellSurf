## dependencies
import numpy as np
from typing import Tuple, Union, List

## custom dependencies
from Min3D.core.framework.point_cloud import PointCloud
from Min3D.core.simulators.sphere import generate_sphere

__all__ = ["generate_gaussian_corona"]


def generate_gaussian_corona(
    center: Union[np.ndarray, Tuple[float, float, float], List[float]],
    corona_radius: float,
    sigma: float,
    num_points: int = 1000,
) -> PointCloud:
    # generate points on the surface of a sphere
    sphere_points = generate_sphere(center, corona_radius, num_points).get_points()

    # add Gaussian noise to the points
    noise = np.random.normal(loc=0, scale=sigma, size=sphere_points.shape)
    corona_points = sphere_points + noise

    # return a PointCloud object
    return PointCloud.from_numpy(corona_points)
