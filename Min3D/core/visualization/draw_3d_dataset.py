## dependencies
import numpy as np
import open3d as o3d

from typing import List, NoReturn, Union

## custom dependencies
from Min3D.core.util.make_3d_path import make_3d_path

__all__ = ["draw_3d_dataset"]


# %% Visualization functions
def draw_3d_dataset(
    paths: List[Union[np.ndarray, List[np.ndarray]]],
    colors: List[Union[np.ndarray, List[np.ndarray]]],
    cmap: str = "viridis",
    scale_factor: float = 1.0,
    magnitude: Union[str, float] = "auto",
) -> NoReturn:
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
    o3d.visualization.draw_geometries(
        geometries,
        window_name="Dataset Visualization",
    )
