## dependencies
import matplotlib
import numpy as np
import open3d as o3d
import matplotlib.colors

from typing import List, Union

## custom dependencies
from cellnav.core.containers.path_3d import Path3D
from cellnav.core.helpers.geo_shape_helper import GeoShapeHelper
from cellnav.core.helpers.estimate_magnitude_from_data import estimate_magnitude_from_data

__all__ = ["make_3d_path"]


# %% Visualization functions
def make_3d_path(
    path: Union[np.ndarray, List[np.ndarray]],
    color: Union[np.ndarray, List[np.ndarray]],
    cmap: Union[str, matplotlib.colors.Colormap] = "viridis",
    size: float = 1.0,
    magnitude: Union[str, float] = "auto",
    draw_faces: bool = False,
) -> Path3D:
    """
    Build a Path3D object for visualization based on the given 3D path and color information.
    The path is visualized as a lineset connecting (point)-meshes, with optional coloring and scaling.

    Args:
        path (Union[np.ndarray, List[np.ndarray]]): The 3D path to visualize, either as a numpy array of shape (N, 3) or a list of 3D points.
        color (Union[np.ndarray, List[np.ndarray]]): The colors for the path.
        cmap (Union[str, matplotlib.colors.Colormap], optional): The colormap for the path. Defaults to "viridis".
        size (float, optional): The scale factor for the path. Defaults to 1.0.
        magnitude (Union[str, float], optional): The magnitude for the path. Defaults to "auto".
        draw_faces (bool, optional): Whether to draw faces for the path. Defaults to False.

    Raises:
        ValueError: If the color array has an invalid shape.

    Returns:
        Path3D: The constructed 3D path object.
    """

    ## generate 3D visualization with open 3d linesets

    # convert path to numpy array if it's a list of points
    if isinstance(path, list):
        path = np.array(path)

    # check if color is a single color or an array of colors
    if isinstance(color, list):
        color = np.array(color)

    # select colormap
    if isinstance(cmap, str):
        cmap = matplotlib.colormaps[cmap]

    # color the wireframe
    if not (color.ndim == 1 and len(color) == 3) and not (
        color.ndim == 1 and color.shape[0] == len(path)
    ):
        raise ValueError(
            "Color must be either a single RGB color or an array of colors with the same length as the path."
        )
    if color.ndim == 1 and color.shape[0] == len(path):
        # apply colormap to the color array
        norm = matplotlib.colors.Normalize(vmin=np.min(color), vmax=np.max(color))
        rgb_array: np.ndarray = cmap(norm(color))[:, :3]

    if isinstance(magnitude, str) and magnitude == "auto":
        magnitude = estimate_magnitude_from_data(path) - 2 # heuristic to adjust the scale for visualization
    scale = float(np.power(10, magnitude)) * size

    # create lineset for the path
    path_edges = [(i, i + 1) for i in range(len(path) - 1)]
    path_lineset = GeoShapeHelper.generate_lineset_from_edges(path, path_edges)

    # create spheres for the start and end nodes
    start_sphere = GeoShapeHelper.generate_icosahedron_on_point(
        path[0],
        radius=3 * scale,
        color=rgb_array[0],  # type: ignore
        make_line=not draw_faces,
    )
    end_sphere = GeoShapeHelper.generate_tetrahedron_on_point(
        path[-1],
        radius=3 * scale,
        color=rgb_array[-1],  # type: ignore
        make_line=not draw_faces,
    )

    # add smaller spheres for the intermediate nodes in the path
    intermediate_nodes = []
    for node in range(1, len(path) - 1):
        sphere = GeoShapeHelper.generate_sphere_on_point(
            path[node],
            radius=2 * scale,
            color=rgb_array[node],  # type: ignore
            make_line=not draw_faces,
        )  # red for intermediate nodes
        intermediate_nodes.append(sphere)

    # color the path lineset
    path_lineset.colors = o3d.utility.Vector3dVector(
        [rgb_array[i] for i in range(len(path_lineset.lines))]  # type: ignore
    )

    # return the 3D path container
    return Path3D(
        edges=path_lineset,
        start_point=start_sphere,
        end_point=end_sphere,
        intermediate_points=intermediate_nodes,
    )