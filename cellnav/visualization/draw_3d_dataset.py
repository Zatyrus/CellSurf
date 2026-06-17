## dependencies
import numpy as np
import open3d as o3d

from typing import List, Union

## custom dependencies
from cellnav.core.util.make_3d_path import make_3d_path

__all__ = ["draw_3d_dataset"]


# %% Visualization functions
def draw_3d_dataset(
    paths: List[Union[np.ndarray, List[np.ndarray]]],
    colors: List[Union[np.ndarray, List[np.ndarray]]],
    cmap: str = "viridis",
    scale_factor: float = 1.0,
    magnitude: Union[str, float] = "auto",
) -> None:
    """Draw multiple 3D paths using open3d visualization. Each path is visualized as a lineset connecting (point)-meshes, with optional coloring and scaling.

    Args:
        paths (List[Union[np.ndarray, List[np.ndarray]]]): The 3D paths to visualize.
        colors (List[Union[np.ndarray, List[np.ndarray]]]): The colors for the paths.
        cmap (str, optional): The colormap for the paths. Defaults to "viridis".
        scale_factor (float, optional): The scale factor for the paths. Defaults to 1.0.
        magnitude (Union[str, float], optional): The magnitude for the paths. Defaults to "auto".

    Returns:
        None: This function does not return anything. It opens an interactive visualization window.
    """

    ## generate path3D objects from the paths and colors
    path_3ds = [
        make_3d_path(
            path=path,
            color=color,
            cmap=cmap,
            scale_factor=scale_factor,
            magnitude=magnitude,
        )
        for path, color in zip(paths, colors)
    ]

    # combine lists of geometries from all path_3d objects    geometries = []
    geometries = []
    for path_3d in path_3ds:
        geometries.extend(path_3d.to_list())

    # visualize the base wireframe and the path linesets together
    o3d.visualization.draw_geometries(  # type: ignore
        geometries,
        window_name="Dataset Visualization",
    )
