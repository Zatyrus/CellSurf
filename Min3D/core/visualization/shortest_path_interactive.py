## dependencies
import numpy as np
import open3d as o3d
from copy import copy

from typing import NoReturn

## custom dependencies
from Min3D.core.framework.surface_graph import SurfaceGraph
from Min3D.core.helpers.geo_shape_helper import GeoShapeHelper

__all__ = ["shortest_path_interactive"]


# %% Visualization functions
def shortest_path_interactive(
    graph: SurfaceGraph, start_node: int, end_node: int
) -> NoReturn:
    ## generate 3D visualization with opne 3d linesets
    # select base wireframe for visualization
    wireframe = copy(graph.edges)
    path = graph.get_shortest_path(start_node, end_node)
    approximate_scaler = __estimate_magnitude_scaler__(graph, scale_adjust=-2)

    # create lineset for the path
    path_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
    path_lineset = GeoShapeHelper.generate_lineset_from_edges(
        graph.vertices, path_edges
    )

    # create spheres for the start and end nodes
    start_sphere = GeoShapeHelper.generate_sphere_on_point(
        wireframe.points[start_node], radius=3 * approximate_scaler, color=[0, 1, 0]
    )  # green for start node
    end_sphere = GeoShapeHelper.generate_sphere_on_point(
        wireframe.points[end_node], radius=3 * approximate_scaler, color=[0, 0, 1]
    )  # blue for end node

    # add smaller spheres for the intermediate nodes in the path
    intermediate_nodes = []
    for node in path[1:-1]:
        sphere = GeoShapeHelper.generate_sphere_on_point(
            wireframe.points[node], radius=2 * approximate_scaler, color=[1, 0, 0]
        )  # red for intermediate nodes
        intermediate_nodes.append(sphere)

    # color both the base wireframe and the path lineset
    wireframe.colors = o3d.utility.Vector3dVector(
        [[0.8, 0.8, 0.8] for _ in range(len(wireframe.lines))]
    )
    path_lineset.colors = o3d.utility.Vector3dVector(
        [[1, 0, 0] for _ in range(len(path_lineset.lines))]
    )

    # visualize the base wireframe and the path lineset together
    o3d.visualization.draw_geometries(
        [path_lineset, wireframe.geometry, start_sphere, end_sphere]
        + intermediate_nodes,
        window_name=f"Shortest Path from Node {start_node} to Node {end_node}",
    )


# %% Helper functions
def __estimate_magnitude_scaler__(graph: SurfaceGraph, scale_adjust: int = -2) -> float:
    return np.power(
        10,
        np.floor(np.log10(np.diff(graph.vertices.get_points(), axis=0).max()))
        + scale_adjust,
    )
