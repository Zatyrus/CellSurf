## dependencies
import numpy as np

from typing import List, Union

## custom dependencies
from cellnav.util.make_3d_path import make_3d_path
from cellnav.core.helpers import draw_geometries

__all__ = ["draw_3d_path"]


# %% Visualization functions
def draw_3d_path(
    path: Union[np.ndarray, List[np.ndarray]],
    color: Union[np.ndarray, List[np.ndarray]],
    cmap: str = "viridis",
    scalebar: Union[float, int, bool] = False,
    size: float = 3.0,
    magnitude: Union[str, float] = "auto",
    draw_faces: bool = False,
    auto_scale_color: bool = True,
) -> None:
    """Draw a 3D path using open3d visualization. The path is visualized as a lineset connecting (point)-meshes, with optional coloring and scaling.

    Args:
        path (Union[np.ndarray, List[np.ndarray]]): The 3D path to visualize.
        color (Union[np.ndarray, List[np.ndarray]]): The colors for the path.
        cmap (str, optional): The colormap for the path. Defaults to "viridis".
        size (float, optional): The scale factor for the path. Defaults to 3.0.
        magnitude (Union[str, float], optional): The magnitude for the path. Defaults to "auto".
        scalebar (Union[float, int, bool], optional): The scale bar for the visualization. Defaults to False.
        draw_faces (bool, optional): Whether to draw faces for the path. Defaults to False.
        auto_scale_color (bool, optional): Whether to automatically scale the color values. Defaults to True.

    Returns:
        None: This function does not return anything. It opens an interactive visualization window.
    """

    ## generate path3D object from the path and color information
    path_3d = make_3d_path(
        path=path,
        color=color,
        cmap=cmap,
        size=size,
        magnitude=magnitude,
        draw_faces=draw_faces,
        auto_scale_color=auto_scale_color,
    )
    # visualize the base wireframe and the path lineset together
    draw_geometries(
        path_3d.to_list(),
        scalebar=scalebar,
    )
