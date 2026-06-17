## dependencies
import numpy as np
import open3d as o3d

from typing import List, Union

## custom dependencies
from cellnav.core.util.make_3d_path import make_3d_path

__all__ = ["draw_3d_path"]


# %% Visualization functions
def draw_3d_path(
    path: Union[np.ndarray, List[np.ndarray]],
    color: Union[np.ndarray, List[np.ndarray]],
    cmap: str = "viridis",
    scale_factor: float = 1.0,
    magnitude: Union[str, float] = "auto",
) -> None:
    """Draw a 3D path using open3d visualization. The path is visualized as a lineset connecting (point)-meshes, with optional coloring and scaling.

    Args:
        path (Union[np.ndarray, List[np.ndarray]]): The 3D path to visualize.
        color (Union[np.ndarray, List[np.ndarray]]): The colors for the path.
        cmap (str, optional): The colormap for the path. Defaults to "viridis".
        scale_factor (float, optional): The scale factor for the path. Defaults to 1.0.
        magnitude (Union[str, float], optional): The magnitude for the path. Defaults to "auto".

    Returns:
        None: This function does not return anything. It opens an interactive visualization window.
    """

    ## generate path3D object from the path and color information
    path_3d = make_3d_path(
        path=path,
        color=color,
        cmap=cmap,
        scale_factor=scale_factor,
        magnitude=magnitude,
    )
    # visualize the base wireframe and the path lineset together
    o3d.visualization.draw_geometries(  # type: ignore
        path_3d.to_list(),
        window_name="Path Visualization",
    )
