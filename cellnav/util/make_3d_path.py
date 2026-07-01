## dependencies
import matplotlib
import numpy as np
import open3d as o3d
import matplotlib.colors

from typing import List, Union

## custom dependencies
from cellnav.core.containers.path_3d_light import Path3dLight
from cellnav.core.containers.path_3d_rendered import Path3dRendered
from cellnav.core.helpers.geo_shape_helper import GeoShapeHelper
from cellnav.core.helpers.estimate_magnitude_from_data import (
    estimate_magnitude_from_data,
)

__all__ = ["make_3d_path"]


# %% Visualization functions
def make_3d_path(
    path: Union[np.ndarray, List[np.ndarray]],
    color: Union[np.ndarray, List[np.ndarray]],
    cmap: Union[str, matplotlib.colors.Colormap] = "viridis",
    size: float = 3.0,
    magnitude: Union[str, float] = "auto",
    draw_faces: bool = False,
    auto_scale_color: bool = True,
) -> Union[Path3dLight, Path3dRendered]:
    """
    Build a Path3D object for visualization based on the given 3D path and color information.
    The path is visualized as a lineset connecting (point)-meshes, with optional coloring and scaling.

    Args:
        path (Union[np.ndarray, List[np.ndarray]]): The 3D path to visualize, either as a numpy array of shape (N, 3) or a list of 3D points.
        color (Union[np.ndarray, List[np.ndarray]]): The colors for the path.
        cmap (Union[str, matplotlib.colors.Colormap], optional): The colormap for the path. Defaults to "viridis".
        size (float, optional): The scale factor for the path. Defaults to 3.0.
        magnitude (Union[str, float], optional): The magnitude for the path. Defaults to "auto".
        draw_faces (bool, optional): Whether to represent the vertices of the path as spheres or points. Defaults to False.
        auto_scale_color (bool, optional): Whether to automatically scale the color values. Defaults to True.

    Raises:
        ValueError: If the color array has an invalid shape.

    Returns:
        Union[Path3dLight, Path3dRendered]: The constructed 3D path object.
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
    elif color.ndim == 1 and color.shape[0] == len(path):
        # apply colormap to the color array
        if auto_scale_color:
            norm = matplotlib.colors.Normalize(vmin=np.min(color), vmax=np.max(color))
            rgb_array: np.ndarray = cmap(norm(color))[:, :3]
        else:
            rgb_array: np.ndarray = cmap(color)[:, :3]

    if isinstance(magnitude, str) and magnitude == "auto":
        magnitude = (
            estimate_magnitude_from_data(path) - 2
        )  # heuristic to adjust the scale for visualization
    scale = float(np.power(10, magnitude)) * size

    # create lineset for the path
    path_edges = [(i, i + 1) for i in range(len(path) - 1)]
    path_lineset = GeoShapeHelper.generate_lineset_from_edges(path, path_edges)

    # color the path lineset
    path_lineset.colors = o3d.utility.Vector3dVector(
        [rgb_array[i] for i in range(len(path_lineset.lines))]  # type: ignore
    )

    if draw_faces:
        # create spheres for the start and end nodes
        start = GeoShapeHelper.generate_icosahedron_on_point(
            path[0],
            radius=3 * scale,
            color=rgb_array[0],  # type: ignore
            make_line=False,  # represent as spheres
        )
        stop = GeoShapeHelper.generate_icosahedron_on_point(
            path[-1],
            radius=3 * scale,
            color=rgb_array[-1],  # type: ignore
            make_line=False,  # represent as spheres
        )

        # add smaller spheres for the intermediate nodes in the path
        intermediate = []
        for node in range(1, len(path) - 1):
            sphere = GeoShapeHelper.generate_sphere_on_point(
                path[node],
                radius=2 * scale,
                color=rgb_array[node],  # type: ignore
                make_line=False,  # represent as spheres
            )  # red for intermediate nodes
            intermediate.append(sphere)

        # return rendered Path3D object
        return Path3dRendered(
            edges=path_lineset,
            start=start,
            stop=stop,
            intermediary=intermediate,
        )

    else:
        # make start
        start = o3d.geometry.PointCloud()
        start.points = o3d.utility.Vector3dVector([path[0]])
        start.colors = o3d.utility.Vector3dVector([rgb_array[0]])  # type: ignore

        # make stop
        stop = o3d.geometry.PointCloud()
        stop.points = o3d.utility.Vector3dVector([path[-1]])
        stop.colors = o3d.utility.Vector3dVector([rgb_array[-1]])  # type: ignore

        # make intermediate nodes
        intermediate = o3d.geometry.PointCloud()
        intermediate.points = o3d.utility.Vector3dVector(path[1:-1])
        intermediate.colors = o3d.utility.Vector3dVector(rgb_array[1:-1])  # type: ignore

        # return lightweight Path3D object
        return Path3dLight(
            edges=path_lineset,
            start=start,
            stop=stop,
            intermediary=intermediate,
        )
