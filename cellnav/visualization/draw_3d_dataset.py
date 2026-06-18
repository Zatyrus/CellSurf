## dependencies
import numpy as np
import open3d as o3d

from typing import List, Union

## custom dependencies
from cellnav.util.make_3d_path import make_3d_path
from cellnav.core.helpers import draw_geometries

__all__ = ["draw_3d_dataset"]


# %% Visualization functions
def draw_3d_dataset(
    paths: Union[List[Union[np.ndarray, List[np.ndarray]]], np.ndarray],
    colors: Union[List[Union[np.ndarray, List[np.ndarray]]], np.ndarray],
    cmap: str = "viridis",
    scalebar: Union[float, int, bool] = False,
    size: float = 1.0,
    magnitude: Union[str, float] = "auto",
    draw_faces: bool = False,
) -> None:
    """Draw multiple 3D paths using open3d visualization. Each path is visualized as a lineset connecting (point)-meshes, with optional coloring and scaling.

    Args:
        paths (Union[List[Union[np.ndarray, List[np.ndarray]]], np.ndarray]): The 3D paths to visualize.
        colors (Union[List[Union[np.ndarray, List[np.ndarray]]], np.ndarray]): The colors for the paths.
        cmap (str, optional): The colormap for the paths. Defaults to "viridis".
        size (float, optional): The scale factor for the paths. Defaults to 1.0.
        magnitude (Union[str, float], optional): The magnitude for the paths. Defaults to "auto".
        draw_faces (bool, optional): Whether to draw faces for the paths. Defaults to False.

    Returns:
        None: This function does not return anything. It opens an interactive visualization window.
    """

    ## generate path3D objects from the paths and colors
    path_3ds = [
        make_3d_path(
            path=path,
            color=color,
            cmap=cmap,
            size=size,
            magnitude=magnitude,
            draw_faces=draw_faces,
        )
        for path, color in zip(paths, colors)
    ]

    # combine lists of geometries from all path_3d objects    geometries = []
    geometries = []
    for path_3d in path_3ds:
        geometries.extend(path_3d.to_list())

    # visualize the base wireframe and the path linesets together
    draw_geometries(
        geometries,
        scalebar=scalebar,
    )
