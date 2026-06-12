## dependencies
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from typing import List, Union

## custom dependencies
from Min3D.core.containers.path_3d import Path3D
from Min3D.core.helpers.geo_shape_helper import GeoShapeHelper

__all__ = ["make_3d_path"]


# %% Visualization functions
def make_3d_path(
    path: Union[np.ndarray, List[np.ndarray]],
    color: Union[np.ndarray, List[np.ndarray]],
    cmap: str = "viridis",
    scale_factor: float = 1.0,
    magnitude: Union[str, float] = "auto",
) -> Path3D:
    ## generate 3D visualization with opne 3d linesets

    # convert path to numpy array if it's a list of points
    if isinstance(path, list):
        path = np.array(path)

    # check if color is a single color or an array of colors
    if isinstance(color, list):
        color = np.array(color)

    # select colormap
    if isinstance(cmap, str):
        cmap = plt.colormaps[cmap]

    # color the wireframe
    if not (color.ndim == 1 and len(color) == 3) and not (
        color.ndim == 1 and color.shape[0] == len(path)
    ):
        raise ValueError(
            "Color must be either a single RGB color or an array of colors with the same length as the path."
        )
    if color.ndim == 1 and color.shape[0] == len(path):
        # apply colormap to the color array
        norm = plt.Normalize(vmin=np.min(color), vmax=np.max(color))
        color_array = cmap(norm(color))[:, :3]

    if magnitude == "auto":
        magnitude = __estimate_magnitude_scaler__(path, scale_adjust=-2)
    scale = magnitude * scale_factor

    # create lineset for the path
    path_edges = [(i, i + 1) for i in range(len(path) - 1)]
    path_lineset = GeoShapeHelper.generate_lineset_from_edges(path, path_edges)

    # create spheres for the start and end nodes
    start_sphere = GeoShapeHelper.generate_icosahedron_on_point(
        path[0], radius=3 * scale, color=color_array[0], make_line=False
    )
    end_sphere = GeoShapeHelper.generate_tetrahedron_on_point(
        path[-1], radius=3 * scale, color=color_array[-1], make_line=False
    )

    # add smaller spheres for the intermediate nodes in the path
    intermediate_nodes = []
    for node in range(1, len(path) - 1):
        sphere = GeoShapeHelper.generate_sphere_on_point(
            path[node], radius=2 * scale, color=color_array[node], make_line=False
        )  # red for intermediate nodes
        intermediate_nodes.append(sphere)

    # color the path lineset
    path_lineset.colors = o3d.utility.Vector3dVector(
        [color_array[i] for i in range(len(path_lineset.lines))]
    )

    # return the 3D path container
    return Path3D(
        edges=path_lineset,
        start_point=start_sphere,
        end_point=end_sphere,
        intermediate_points=intermediate_nodes,
    )


# %% Helper functions
def __estimate_magnitude_scaler__(path: np.ndarray, scale_adjust: int = -2) -> float:
    return np.power(10, np.floor(np.log10(np.diff(path, axis=0).max())) + scale_adjust)
