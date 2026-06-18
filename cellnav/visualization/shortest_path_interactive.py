## dependencies
import numpy as np
import open3d as o3d
from copy import copy


## custom dependencies
from cellnav.core.framework.surface_graph import SurfaceGraph
from cellnav.core.helpers.geo_shape_helper import GeoShapeHelper
from cellnav.core.helpers.estimate_magnitude_from_data import estimate_magnitude_from_data

__all__ = ["shortest_path_interactive"]


# %% Visualization functions
def shortest_path_interactive(graph: SurfaceGraph, source: int, target: int) -> None:
    """Spin up an interactive visualization of the shortest path between source and target nodes in the given surface graph.

    Args:
        graph (SurfaceGraph): The surface graph containing the nodes and edges.
        source (int): The node from which to start the path.
        target (int): The node at which to end the path.

    Returns:
        None: This function does not return anything. It opens an interactive visualization window.
    """

    ## generate 3D visualization with open 3d linesets
    # select base wireframe for visualization
    wireframe = copy(graph.edges)
    path = graph.get_shortest_path(source, target)
    approximate_scaler = np.power(10, estimate_magnitude_from_data(graph.vertices.get_points()) - 2)

    # create lineset for the path
    path_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
    path_lineset = GeoShapeHelper.generate_lineset_from_edges(
        graph.vertices.points, path_edges
    )

    # create spheres for the start and end nodes
    start_sphere = GeoShapeHelper.generate_sphere_on_point(
        wireframe.points[source], radius=3 * approximate_scaler, color=[0, 1, 0]
    )  # green for start node
    end_sphere = GeoShapeHelper.generate_sphere_on_point(
        wireframe.points[target], radius=3 * approximate_scaler, color=[0, 0, 1]
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
    o3d.visualization.draw_geometries(  # type: ignore
        [path_lineset, wireframe.geometry, start_sphere, end_sphere]
        + intermediate_nodes,
        window_name=f"Shortest Path from Node {source} to Node {target}",
    )