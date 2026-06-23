## dependencies
import os
import sys
import numpy as np
import open3d as o3d
import rustworkx as rx
import matplotlib.figure
from rustworkx.visualization import mpl_draw, graphviz_draw

from typing import Callable, Dict, List, Tuple, Union, Optional, Any

if sys.platform.startswith("win"):
    import PyFileDialogue as pyfd
else:
    pyfd = None  # placeholder for non-Windows systems, as tkinter is not supported on Unix-based systems

# tqdm for progress bars - automatically selects the right version for notebooks vs. terminal
from IPython.core.getipython import get_ipython

try:
    ipy_str = str(type(get_ipython()))
    if "zmqshell" in ipy_str:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
except ImportError:
    from tqdm import tqdm

# custom imports
from cellnav.core.framework.point_cloud import PointCloud
from cellnav.core.framework.surface_wireframe import SurfaceWireframe
from cellnav.core.framework.unique_surface_wireframe import UniqueSurfaceWireframe
from cellnav.util.geometry_transformer import GeometryTransformer


__all__ = ["SurfaceGraph"]


# class implementation
class SurfaceGraph:
    # base data
    _vertices: PointCloud
    _edges: Union[SurfaceWireframe, UniqueSurfaceWireframe]

    # graph representation
    _graph: rx.PyGraph

    # lookup tables for fast access to edge lengths and distances between nodes
    _edge_length_LUT: Optional[Dict[Tuple[int, int], float]]

    # optional - can be computed on the fly if not provided, but can be useful to precompute for faster access
    _distance_LUT: Optional[
        Union[Dict[int, Dict[int, float]], rx.AllPairsPathLengthMapping]
    ]
    _path_LUT: Optional[Union[Dict[int, Dict[int, List[int]]], rx.AllPairsPathMapping]]
    _distance_matrix: Optional[np.ndarray]  # table of node distances for fast access

    def __init__(
        self,
        vertices: PointCloud,
        edges: Union[SurfaceWireframe, UniqueSurfaceWireframe],
        graph: rx.PyGraph,
        edge_length_LUT: Optional[Dict[Tuple[int, int], float]] = None,
        distance_LUT: Optional[
            Union[Dict[int, Dict[int, float]], rx.AllPairsPathLengthMapping]
        ] = None,
        path_LUT: Optional[
            Union[Dict[int, Dict[int, List[int]]], rx.AllPairsPathMapping]
        ] = None,
        distance_matrix: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        """
        Initialize a SurfaceGraph object, which represents a graph structure derived from a surface wireframe,
        where vertices correspond to points in the wireframe and edges correspond to connections between those points.

        Args:
            vertices (PointCloud): The point cloud representing the vertices of the surface graph.
            edges (Union[SurfaceWireframe, UniqueSurfaceWireframe]): The surface wireframe representing the edges of the surface graph.
            graph (rx.PyGraph): The rustworkx graph representing the surface graph.
            edge_length_LUT (Optional[Dict[Tuple[int, int], float]], optional): The lookup table for edge lengths. Defaults to None.
            distance_LUT (Optional[ Union[Dict[int, Dict[int, float]], rx.AllPairsPathLengthMapping] ], optional): The lookup table for distances between nodes. Defaults to None.
            path_LUT (Optional[ Union[Dict[int, Dict[int, List[int]]], rx.AllPairsPathMapping] ], optional): The lookup table for paths between nodes. Defaults to None.
            distance_matrix (Optional[np.ndarray], optional): The matrix of node distances for fast access. Defaults to None.
        """
        self._vertices = vertices
        self._edges = edges
        self._graph = graph

        # set optional LUTs and distance matrix
        self._edge_length_LUT = edge_length_LUT
        self._distance_LUT = distance_LUT
        self._path_LUT = path_LUT
        self._distance_matrix = distance_matrix

        # config options for decorators - currently a placeholder
        self._config = {"verbose": True, "DEBUG": False}

        # update from kwargs if provided
        for key in self._config.keys():
            if key in kwargs:
                self._config[key] = kwargs[key]

        # override if DEBUG
        if self._config["DEBUG"]:
            self._config["verbose"] = True

        # run post init
        self.__post_init__()

    def __post_init__(self) -> None:
        """Post-setup function to check consistency of the graph and its components, and to compute any necessary LUTs for graph algorithms.

        Raises:
            TypeError: If vertices are not of type PointCloud.
            TypeError: If edges are neither of type SurfaceWireframe or UniqueSurfaceWireframe.
            TypeError: If graph is not of type rustworkx PyGraph.
            ValueError: If edge length LUT is not consistent with graph edges.
        """
        # check that vertices and edges are of the correct types
        if not isinstance(self._vertices, PointCloud):
            raise TypeError(
                f"Vertices must be a PointCloud, got {type(self._vertices)}"
            )
        if not isinstance(self._edges, (SurfaceWireframe, UniqueSurfaceWireframe)):
            raise TypeError(
                f"Edges must be a SurfaceWireframe or UniqueSurfaceWireframe, got {type(self._edges)}"
            )
        if not isinstance(self._graph, rx.PyGraph):
            raise TypeError(
                f"Graph must be a rustworkx PyGraph, got {type(self._graph)}"
            )

        # convert SurfaceWireframe to UniqueSurfaceWireframe if necessary,
        # to ensure consistent edge representation and avoid issues with duplicate edges in graph construction
        if isinstance(self._edges, SurfaceWireframe):
            self._edges = UniqueSurfaceWireframe.from_wireframe(self._edges)

        # make edge_length_LUT and check consistency if not provided - this is necessary for graph construction and shortest path algorithms
        if self._edge_length_LUT is None:
            self._edge_length_LUT = GeometryTransformer.edge_length_LUT_from(
                self._edges
            )

        # check consistency of edge_length_LUT with graph edges to catch any issues with graph construction or edge representation early on
        assert self.check_consistency_of_edge_length_LUT()

    @classmethod
    def from_wireframe(
        cls, wireframe: Union[SurfaceWireframe, UniqueSurfaceWireframe], **kwargs
    ) -> "SurfaceGraph":
        """Create a SurfaceGraph from a given surface wireframe by constructing the corresponding graph structure and computing necessary LUTs for graph algorithms.

        Args:
            wireframe (Union[SurfaceWireframe, UniqueSurfaceWireframe]): A surface wireframe from which to create the surface graph.

        Returns:
            SurfaceGraph: The created surface graph.
        """
        # make vertices
        edges = UniqueSurfaceWireframe.from_wireframe(wireframe)
        vertices = PointCloud.from_numpy(edges.get_points())

        # make graph
        graph = rx.PyGraph()

        # populate graph with vertices
        graph.add_nodes_from(range(len(vertices.points)))

        # populate graph with edges
        edge_length_LUT = GeometryTransformer.edge_length_LUT_from(edges)
        graph.add_edges_from(
            [
                (line[0], line[1], edge_length_LUT[(line[0], line[1])])
                for line in edges.lines
            ]
        )

        return cls(
            vertices=vertices,
            edges=edges,
            graph=graph,
            edge_length_LUT=edge_length_LUT,
            **kwargs,
        )

    @classmethod
    def from_point_cloud(
        cls, point_cloud: PointCloud, k: int = 6, **kwargs
    ) -> "SurfaceGraph":
        """Create a SurfaceGraph from a given point cloud by constructing a k-nearest neighbor graph based on the spatial proximity of the points in the cloud.

        Args:
            point_cloud (PointCloud): The point cloud from which to create the surface graph.
            k (int, optional): The number of nearest neighbors to consider for each point. Defaults to 6.

        Returns:
            SurfaceGraph: The created surface graph.
        """
        # build kNN graph from point cloud
        edges = GeometryTransformer.kNN_wireframe_from(point_cloud, k=k)
        return cls.from_wireframe(edges, **kwargs)

    @classmethod
    def from_npz(cls, file_path: Optional[str], **kwargs) -> "SurfaceGraph":
        if not file_path:
            if pyfd is not None:
                file_path = pyfd.call_file(
                    title="Select Surface Mesh NPZ File",
                    filetypes=[("NPZ files", "*.npz")],
                )
            else:
                raise ValueError(
                    "Invalid file path. Please provide a valid file path ending with .npz or ensure that pyfiledialog is installed for file dialog support."
                )

        if not file_path.endswith(".npz"):
            raise ValueError("Invalid file type. Please select a .npz file.")

        # load npz
        loaded = np.load(file_path, allow_pickle=True)

        # unpack npz
        vertices = PointCloud.from_dict(loaded["vertices"].item())
        edges = UniqueSurfaceWireframe.from_dict(loaded["edges"].item())
        edge_length_LUT = loaded["edge_length_LUT"].item()
        path_LUT = loaded["path_LUT"].item()
        distance_LUT = loaded["distance_LUT"].item()
        distance_matrix = (
            loaded["distance_matrix"] if loaded["distance_matrix"].size > 1 else None
        )

        # make graph
        graph = rx.PyGraph()

        # populate graph with vertices
        graph.add_nodes_from(range(len(vertices.points)))

        # populate graph with edges
        if edge_length_LUT is None:
            edge_length_LUT = GeometryTransformer.edge_length_LUT_from(edges)
        graph.add_edges_from(
            [
                (line[0], line[1], edge_length_LUT[(line[0], line[1])])
                for line in edges.lines
            ]
        )

        return cls(
            vertices=vertices,
            edges=edges,
            graph=graph,
            edge_length_LUT=edge_length_LUT,
            path_LUT=path_LUT,
            distance_LUT=distance_LUT,
            distance_matrix=distance_matrix,
            **kwargs,
        )

    # %% Utility
    def get_vertices(self) -> np.ndarray:
        return self._vertices.get_points()

    def get_edges(self) -> np.ndarray:
        return self._edges.get_lines()

    def get_closest_node(
        self, point: Union[np.ndarray, List[float], Tuple[float, float, float]]
    ) -> int:
        """
        Find the node in the graph that is closest to the given point.

        Args:
            point (Union[np.ndarray, List[float], Tuple[float, float, float]]): The 3D coordinates of the point.

        Returns:
            int: The index of the closest node.
        """
        if len(point) != 3:
            raise ValueError(f"Point must be a 3D coordinate, got {point}")

        if isinstance(point, (list, tuple)):
            point = np.array(point)

        distances = np.linalg.norm(self._vertices.get_points() - point, axis=1)
        return np.argmin(distances, axis=0)

    # %% Path and distance queries
    def get_shortest_path(self, source: int, target: int, **kwargs) -> List[int]:
        """
        Main API function for retrieving the shortest path between two nodes in the surface graph,
        with optional parameters for customizing the pathfinding algorithm and its behavior.
        The method will first check if a precomputed path LUT is available for fast access to the shortest path.
        If not, it will compute the shortest path on the fly using Dijkstra's algorithm.

        Args:
            source (int): The index of the source node from which to start the pathfinding.
            target (int): The index of the target node to which to find the shortest path.

        Returns:
            List[int]: A list of node indices representing the shortest path from source to target. The list will include both the source and target nodes, as well as any intermediate nodes along the path. They are in order from source to target.
        """
        if self._path_LUT is None:
            return list(self.dijkstra_shortest_path(source, target, **kwargs)[target])
        return list(self._path_LUT[source][target])  # type: ignore

    def get_shortest_distance(self, source: int, target: int, **kwargs) -> float:
        """
        Main API function for retrieving the shortest distance between two nodes in the surface graph,
        with optional parameters for customizing the pathfinding algorithm and its behavior.
        The method will preferably check if a precomputed distance matrix is available for O(1) access to the shortest distance.
        If not, it will check if a distance LUT is available for O(1) access to the shortest distance.
        If neither are available, it will compute the shortest distance on the fly using Dijkstra's algorithm.

        Args:
            source (int): The index of the source node from which to start the pathfinding.
            target (int): The index of the target node to which to find the shortest path.

        Returns:
            float: The shortest distance between the source and target nodes.
        """
        if self._distance_matrix is not None:
            return self._distance_matrix[source, target]
        elif self._distance_LUT is not None:
            return self._distance_LUT[source][target]  # type: ignore
        else:
            return self.dijkstra_shortest_distance(source, target, **kwargs)[target]  # type: ignore

    # %% Graph parameters
    def transitivity(self) -> float:
        """
        Compute the transitivity of the graph, which is a measure of the likelihood that the adjacent vertices of a vertex are connected.
        It is defined as the ratio of the number of edges between the neighbors of a vertex to the number of possible edges between those neighbors.

        Returns:
            float: The transitivity of the graph.
        """
        return rx.transitivity(self._graph)

    def is_planar(self) -> bool:
        """Determine whether the graph is planar, meaning that it can be drawn on a plane without any edges crossing.

        Returns:
            bool: True if the graph is planar, False otherwise.
        """
        return rx.is_planar(self._graph)

    def is_connected(self) -> bool:
        """Determine whether the graph is connected, meaning that there is a path between any two vertices in the graph.

        Returns:
            bool: True if the graph is connected, False otherwise.
        """
        return rx.is_connected(self._graph)

    def number_of_connected_components(self) -> int:
        """
        Compute the number of connected components in the graph,
        which is the number of subgraphs in which any two vertices are connected to each other by paths,
        and which are connected to no additional vertices in the supergraph.

        Returns:
            int: The number of connected components in the graph.
        """
        return rx.number_connected_components(self._graph)

    def has_negative_edge_weights(self) -> bool:
        """
        Determine whether the graph contains any negative edge weights, w
        hich can affect the choice of shortest path algorithms and their correctness.

        Returns:
            bool: True if the graph contains negative edge weights, False otherwise.
        """
        return bool(np.any(np.array(self._graph.edges()) < 0))

    # %% Graph visualization
    def draw_graph_with_matplotlib(
        self, **kwargs
    ) -> Optional[matplotlib.figure.Figure]:
        """
        Draw the graph using Matplotlib, with optional parameters for customizing the visualization.
        Don not use this for large graphs, as it can be slow and cluttered. For larger graphs, consider using graphviz_draw instead.

        Raises:
            ImportError: If Matplotlib is not installed, an ImportError will be raised with a message indicating that Matplotlib is required for this function.

        Returns:
            Optional[matplotlib.figure.Figure]: The Matplotlib figure object containing the graph visualization, or None if the graph could not be drawn.
        """
        try:
            return mpl_draw(self._graph, **kwargs)
        except ImportError:
            raise ImportError(
                "Matplotlib is not installed. Please install matplotlib to use this function."
            )

    def draw_graph_with_graphviz(self, **kwargs) -> Optional[Any]:
        """
        Draw the graph using Graphviz, with optional parameters for customizing the visualization.
        Graphviz is generally more efficient and produces clearer visualizations for larger graphs
        compared to Matplotlib, especially when using layout algorithms like 'sfdp' or 'neato'.

        However, you will need to have Graphviz installed on your system to use this function, and it may require additional configuration to work properly.

        Raises:
            ImportError: If Graphviz is not installed, an ImportError will be raised with a message indicating that Graphviz is required for this function.

        Returns:
            Optional[Any]: The Graphviz figure object containing the graph visualization, or None if the graph could not be drawn.
        """
        try:
            return graphviz_draw(self._graph, **kwargs)
        except ImportError:
            raise ImportError(
                "Graphviz is not installed. Please install graphviz to use this function."
            )

    # %% Shortest path algorithm - source -> target
    def dijkstra_shortest_path(
        self,
        source: int,
        target: int,
        weight_fn: Optional[Callable] = float,
        default_weight: float = 1.0,
    ) -> rx.PathMapping:
        """
        Compute the shortest path between a source node and a target node in the graph using Dijkstra's algorithm,
        which is efficient for graphs with non-negative edge weights.

        Args:
            source (int): The source node.
            target (int): The target node.
            weight_fn (Optional[Callable], optional): A function to compute the weight of an edge. It will be given the edge-object as input and needs to return a float output by which to determine the edge weight. Defaults to float.
            default_weight (float, optional): If `weight_fn` is None, this can be optionally used to specify a default weight to use for all edges. Defaults to 1.0.

        Raises:
            ValueError: If the graph contains negative edge weights.

        Returns:
            rx.PathMapping: The shortest path between the source and target nodes. The mapping will be a dictionary where the keys are target node indices and the values are lists of node indices representing the path from the source to that target node.
        """
        if self.has_negative_edge_weights():
            raise ValueError(
                "Graph contains negative edge weights. Dijkstra's algorithm cannot be used.\nConsider using Bellman-Ford algorithm instead."
            )
        return rx.dijkstra_shortest_paths(
            self._graph,
            source=source,
            target=target,
            weight_fn=weight_fn,
            default_weight=default_weight,
        )

    def bellman_ford_shortest_path(
        self,
        source: int,
        target: int,
        weight_fn: Optional[Callable] = float,
        default_weight: float = 1.0,
    ) -> rx.PathMapping:
        """
        Compute the shortest path between a source node and a target node in the graph using Bellman-Ford algorithm,
        which is suitable for graphs that may contain negative edge weights, but does not allow for negative weight cycles.

        Args:
            source (int): The source node.
            target (int): The target node.
            weight_fn (Optional[Callable], optional): A function to compute the weight of an edge. It will be given the edge-object as input and needs to return a float output by which to determine the edge weight. Defaults to float.
            default_weight (float, optional): If `weight_fn` is None, this can be optionally used to specify a default weight to use for all edges. Defaults to 1.0.

        Returns:
            rx.PathMapping: The shortest path between the source and target nodes. The mapping will be a dictionary where the keys are target node indices and the values are lists of node indices representing the path from the source to that target node.
        """
        return rx.bellman_ford_shortest_paths(
            self._graph,
            source=source,
            target=target,
            weight_fn=weight_fn,
            default_weight=default_weight,
        )

    # %% Shortest distance algorithm - source -> target
    def dijkstra_shortest_distance(
        self, source: int, target: int, weight_fn: Optional[Callable] = float
    ) -> rx.PathLengthMapping:
        """
        Compute the shortest distance between a source node and a target node in the graph using Dijkstra's algorithm,
        which is efficient for graphs with non-negative edge weights.

        Args:
            source (int): The source node.
            target (int): The target node.
            weight_fn (Optional[Callable], optional): A function to compute the weight of an edge. It will be given the edge-object as input and needs to return a float output by which to determine the edge weight. Defaults to float.

        Raises:
            ValueError: If the graph contains negative edge weights.

        Returns:
            rx.PathLengthMapping: The shortest distance between the source and target nodes.
        """
        if self.has_negative_edge_weights():
            raise ValueError(
                "Graph contains negative edge weights. Dijkstra's algorithm cannot be used.\nConsider using Bellman-Ford algorithm instead."
            )
        return rx.dijkstra_shortest_path_lengths(
            self._graph, node=source, goal=target, edge_cost_fn=weight_fn
        )

    def bellman_ford_shortest_distance(
        self, source: int, target: int, weight_fn: Optional[Callable] = float
    ) -> rx.PathLengthMapping:
        """
        Compute the shortest distance between a source node and a target node in the graph using Bellman-Ford algorithm,
        which is suitable for graphs that may contain negative edge weights, but does not allow for negative weight cycles.

        Args:
            source (int): The source node.
            target (int): The target node.
            weight_fn (Optional[Callable], optional): A function to compute the weight of an edge. It will be given the edge-object as input and needs to return a float output by which to determine the edge weight. Defaults to float.

        Returns:
            rx.PathLengthMapping: The shortest distance between the source and target nodes.
        """
        return rx.bellman_ford_shortest_path_lengths(
            self._graph, node=source, goal=target, edge_cost_fn=weight_fn
        )

    # %% Shortest path algorithm - all pairs
    def dijkstra_shortest_path_global(
        self, weight_fn: Optional[Callable] = float
    ) -> rx.AllPairsPathMapping:
        """
        Compute the shortest paths between all pairs of nodes in the graph using Dijkstra's algorithm,
        which is efficient for graphs with non-negative edge weights.

        Args:
            weight_fn (Optional[Callable], optional): A function to compute the weight of an edge. It will be given the edge-object as input and needs to return a float output by which to determine the edge weight. Defaults to float.

        Raises:
            ValueError: If the graph contains negative edge weights.

        Returns:
            rx.AllPairsPathMapping: The shortest paths between all pairs of nodes. The mapping will be a dictionary where the keys are source node indices and the values are dictionaries mapping target node indices to lists of node indices representing the path from the source to that target node.
        """
        if self.has_negative_edge_weights():
            raise ValueError(
                "Graph contains negative edge weights. Dijkstra's algorithm cannot be used.\nConsider using Bellman-Ford algorithm instead."
            )
        self._path_LUT = rx.all_pairs_dijkstra_shortest_paths(
            self._graph, edge_cost_fn=weight_fn
        )
        return self._path_LUT

    def bellman_ford_shortest_path_global(
        self, weight_fn: Optional[Callable] = float
    ) -> rx.AllPairsPathMapping:
        """
        Compute the shortest paths between all pairs of nodes in the graph using Bellman-Ford algorithm,
        which is suitable for graphs that may contain negative edge weights, but does not allow for negative weight cycles.

        Args:
            weight_fn (Optional[Callable], optional): A function to compute the weight of an edge. It will be given the edge-object as input and needs to return a float output by which to determine the edge weight. Defaults to float.

        Returns:
            rx.AllPairsPathMapping: The shortest paths between all pairs of nodes.
        """
        self._path_LUT = rx.all_pairs_bellman_ford_shortest_paths(
            self._graph, edge_cost_fn=weight_fn
        )
        return self._path_LUT

    def floyd_warshall_shortest_paths_global(
        self,
        weight_fn: Optional[Callable] = float,
        default_weight: float = 1.0,
        parallel_threshold: int = 300,
    ) -> rx.AllPairsPathLengthMapping:
        """
        Compute the shortest paths between all pairs of nodes in the graph using Floyd-Warshall algorithm,
        which is a dynamic programming algorithm that can handle negative edge weights (but not negative weight cycles) and is efficient for dense graphs (O(V^3)).

        Args:
            weight_fn (Optional[Callable], optional): A function to compute the weight of an edge. It will be given the edge-object as input and needs to return a float output by which to determine the edge weight. Defaults to float.
            default_weight (float, optional): If `weight_fn` is None, this can be optionally used to specify a default weight to use for all edges. Defaults to 1.0.
            parallel_threshold (int, optional): The threshold for the number of nodes above which the algorithm will use parallel processing. Defaults to 300.

        Returns:
            rx.AllPairsPathLengthMapping: The shortest paths between all pairs of nodes. The mapping will be a dictionary where the keys are source node indices and the values are dictionaries mapping target node indices to the shortest path from the source to that target node.
        """
        self.path_length_LUT = rx.floyd_warshall(
            self._graph,
            weight_fn=weight_fn,
            default_weight=default_weight,
            parallel_threshold=parallel_threshold,
        )
        return self.path_length_LUT

    # %% Shortest distance algorithm - all pairs
    def dijkstra_shortest_distance_global(
        self, weight_fn: Optional[Callable] = float
    ) -> rx.AllPairsPathLengthMapping:
        """Compute the shortest distances between all pairs of nodes in the graph using Dijkstra's algorithm,
        which is efficient for graphs with non-negative edge weights.

        Args:
            weight_fn (Optional[Callable], optional): A function to compute the weight of an edge. It will be given the edge-object as input and needs to return a float output by which to determine the edge weight. Defaults to float.

        Raises:
            ValueError: If the graph contains negative edge weights.

        Returns:
            rx.AllPairsPathLengthMapping: The shortest distances between all pairs of nodes. The mapping will be a dictionary where the keys are source node indices and the values are dictionaries mapping target node indices to the shortest distance from the source to that target node.
        """
        if self.has_negative_edge_weights():
            raise ValueError(
                "Graph contains negative edge weights. Dijkstra's algorithm cannot be used.\nConsider using Bellman-Ford algorithm instead."
            )
        self._distance_LUT = rx.all_pairs_dijkstra_path_lengths(
            self._graph, edge_cost_fn=weight_fn
        )
        return self._distance_LUT

    def bellman_ford_shortest_distance_global(
        self, weight_fn: Optional[Callable] = float
    ) -> rx.AllPairsPathLengthMapping:
        """
        Compute the shortest distances between all pairs of nodes in the graph using Bellman-Ford algorithm,
        which is suitable for graphs that may contain negative edge weights, but does not allow for negative weight cycles.

        Args:
            weight_fn (Optional[Callable], optional): A function to compute the weight of an edge. It will be given the edge-object as input and needs to return a float output by which to determine the edge weight. Defaults to float.

        Returns:
            rx.AllPairsPathLengthMapping: The shortest distances between all pairs of nodes. The mapping will be a dictionary where the keys are source node indices and the values are dictionaries mapping target node indices to the shortest distance from the source to that target node.
        """
        self._distance_LUT = rx.all_pairs_bellman_ford_path_lengths(
            self._graph, edge_cost_fn=weight_fn
        )
        return self._distance_LUT

    # %% Helper functions
    def suggest_shortest_path_algorithm(self) -> None:
        """
        Helpful utility function to suggest the most efficient shortest path algorithm to use for this graph based on its properties,
        such as the number of vertices and edges, and whether it contains negative edge weights.

        The function will compute the time complexity of following algorithms based on the graph properties,
        and will suggest the most efficient algorithm based on these complexities:
        - Dijkstra's algorithm
        - Bellman-Ford algorithm
        - Floyd-Warshall algorithm
        """
        E: float = self._edges.get_lines().shape[0]
        V: float = self._vertices.get_points().shape[0]

        # Compute the time complexity of each implemented algorithm
        def dijkstra_time_complexity(E: float, V: float) -> float:
            """Dijkstra's algorithm has a time complexity of O(E + V log V) when implemented with a priority queue (e.g., using a binary heap)."""
            return E + V * np.log(V)

        def bellman_ford_time_complexity(E: float, V: float) -> float:
            """Bellman-Ford algorithm has a time complexity of O(VE)."""
            return V * E

        def floyd_warshall_time_complexity(E: float, V: float) -> float:
            """Floyd-Warshall algorithm has a time complexity of O(V^3)."""
            return V**3

        complexities = {
            "Dijkstra": dijkstra_time_complexity(E, V),
            "Bellman-Ford": bellman_ford_time_complexity(E, V),
            "Floyd-Warshall": floyd_warshall_time_complexity(E, V),
        }

        print(f"Dijkstra's time complexity: O({complexities['Dijkstra']:.2e})")
        print(f"Bellman-Ford's time complexity: O({complexities['Bellman-Ford']:.2e})")
        print(
            f"Floyd-Warshall's time complexity: O({complexities['Floyd-Warshall']:.2e})"
        )

        ## suggest the most efficient algorithm based on the time complexities
        most_efficient = min(complexities, key=complexities.get)  # type: ignore
        print(
            f"\nBased on the time complexities, the {most_efficient} algorithm is the most efficient for this graph."
        )

    def build_distance_matrix(
        self,
        distance_LUT: Optional[
            Union[Dict[int, Dict[int, float]], rx.AllPairsPathLengthMapping]
        ] = None,
        silent: bool = False,
    ) -> np.ndarray:
        """
        Build a distance matrix from a given distance LUT, which is a symmetric 2D array where the entry at row i and column j
        represents the shortest distance between node i and node j in the graph.

        Args:
            distance_LUT (Optional[ Union[Dict[int, Dict[int, float]], rx.AllPairsPathLengthMapping] ], optional): Distance lookup table. Defaults to None.
            silent (bool, optional): Whether to suppress progress bar. Defaults to False.

        Raises:
            ValueError: If distance LUT is not provided and not precomputed in the object.

        Returns:
            np.ndarray: The computed distance matrix.
        """
        if distance_LUT is None:
            if self._distance_LUT is None:
                raise ValueError(
                    "Distance LUT must be provided or precomputed before building distance matrix."
                )
            distance_LUT = self._distance_LUT

        # initialize distance matrix with zeros
        self._distance_matrix = np.zeros(
            (self._graph.num_nodes(), self._graph.num_nodes())
        )
        # compute the distance matrix from the shortest path LUT - with progress bar
        with tqdm(
            total=self._graph.num_nodes(),
            desc="Building distance matrix from LUT",
            disable=silent,
        ) as pbar:
            for i in range(self._graph.num_nodes()):
                # triangular indexing to avoid redundant computations, as the distance matrix is symmetric and the diagonal is zero
                self._distance_matrix[i, i + 1 :] = np.array(
                    list(distance_LUT[i].values())
                )[i:]  # we need to start +1 to skip the diagonal, which is zero.
                pbar.update(1)

        # make the distance matrix symmetric
        self._distance_matrix = self._distance_matrix + self._distance_matrix.T

        return self._distance_matrix

    def build_distance_LUT_from_matrix(
        self, distance_matrix: Optional[np.ndarray] = None, silent: bool = False
    ) -> Dict[int, Dict[int, float]]:
        """Build a distance LUT from a given distance matrix, which is a symmetric 2D array where the entry at row i and column j represents
        the shortest distance between node i and node j in the graph.

        The LUT is a nested dictionary where the outer keys are source node indices and the inner keys are target node indices,
        mapping to the shortest distance between those nodes.

        Args:
            distance_matrix (np.ndarray): The distance matrix. If None, the method will use the precomputed distance matrix in the object if available. Defaults to None.
            silent (bool, optional): Whether to suppress progress bar. Defaults to False.

        Returns:
            Dict[int, Dict[int, float]]: The computed distance lookup table.
        """

        if distance_matrix is None and self._distance_matrix is None:
            raise ValueError(
                "Distance matrix must be provided or precomputed before building distance LUT."
            )

        if (
            distance_matrix is not None
            and self._distance_matrix is not None
            and not np.array_equal(distance_matrix, self._distance_matrix)
        ):
            raise ValueError(
                "Provided distance matrix does not match the precomputed distance matrix in the object. Please provide a consistent distance matrix or set distance_matrix to None to use the precomputed one."
            )

        if distance_matrix is None and isinstance(self._distance_matrix, np.ndarray):
            distance_matrix = self._distance_matrix

        distance_LUT: Dict[int, Dict[int, float]] = {}
        with tqdm(
            total=distance_matrix.shape[0],  # type: ignore
            desc="Building distance LUT from matrix",
            disable=silent,
        ) as pbar:
            for i in range(distance_matrix.shape[0]):  # type: ignore
                distance_LUT[i] = {
                    j: distance_matrix[i, j]  # type: ignore
                    for j in range(distance_matrix.shape[1])  # type: ignore
                    if i != j
                }
                pbar.update(1)
        self._distance_LUT = distance_LUT
        return self._distance_LUT

    def path_to_distance(self, path: Union[List[int], Tuple[int, ...]]) -> float:
        """
        Convert a given path, which is a list or tuple of node indices representing a sequence of nodes in the graph,
        to its corresponding distance by summing the lengths of the edges along the path using the edge length LUT.

        Args:
            path (Union[List[int], Tuple[int, ...]]): The path to convert, represented as a list or tuple of node indices. The path should be in order from source to target, and should include all intermediate nodes along the path.

        Raises:
            ValueError: If no edlge length LUT is available for computing the distance.
            ValueError: If an edge in the path is not found in the edge length LUT.

        Returns:
            float: The total distance of the path.
        """
        # catch self._edge_length_LUT is None
        if self._edge_length_LUT is None:
            raise ValueError(
                "Edge length LUT must be computed before converting path to distance."
            )

        distance = 0.0
        for i in range(len(path) - 1):
            edge = (path[i], path[i + 1])
            if edge in self._edge_length_LUT.keys():
                distance += self._edge_length_LUT[edge]
            elif (
                edge[1],
                edge[0],
            ) in self._edge_length_LUT:  # check for undirected edge
                distance += self._edge_length_LUT.get((edge[1], edge[0]), 0.0)
            else:
                raise ValueError(f"Edge {edge} not found in edge length LUT.")
        return distance

    def path_LUT_to_distance_LUT(
        self,
    ) -> Union[Dict[int, Dict[int, float]], rx.AllPairsPathLengthMapping]:
        """
        Convert the path LUT, which is a nested dictionary mapping source node indices to target node indices to lists of node indices
        representing the shortest paths between those nodes, to a distance LUT, which is a nested dictionary mapping
        source node indices to target node indices to the shortest distance between those nodes.

        Raises:
            ValueError: If no path LUT is available for computing the distance LUT.

        Returns:
            Union[Dict[int, Dict[int, float]], rx.AllPairsPathLengthMapping]: The converted distance LUT.
        """

        # catch self._edge_length_LUT is None
        if self._path_LUT is None:
            raise ValueError(
                "Path LUT must be computed before converting to distance LUT."
            )

        distance_LUT: Dict[int, Dict[int, float]] = {}
        for source, target_paths in self._path_LUT.items():
            distance_LUT[source] = {}
            for target, path in target_paths.items():
                distance_LUT[source][target] = self.path_to_distance(path)  # type: ignore
        self._distance_LUT = distance_LUT
        return self._distance_LUT

    # %% I/O

    # %% Check-up functions
    def check_consistency_of_edge_length_LUT(self) -> bool:
        """
        Check that the edge length LUT is consistent with the edges in the graph,
        meaning that all edges in the graph are present in the LUT and vice versa.

        Raises:
            ValueError: If the edge length LUT is not yet computed.
            ValueError: If there are edges in the graph that are missing from the LUT, or if there are edges in the LUT that are missing from the graph.

        Returns:
            bool: True if the edge length LUT is consistent with the graph edges, False otherwise.
        """
        if self._edge_length_LUT is None:
            raise ValueError(
                "Edge length LUT must be computed before checking consistency."
            )

        # check that all edges in the graph are present in the edge_length_LUT and vice versa
        # create frozensets of the edge keys in the LUT and the graph edges for efficient comparison
        # frozensets are used to ensure that the order of edges does not affect the comparison
        frozen_edge_keys = frozenset(self._edge_length_LUT.keys())
        frozen_graph_edges = frozenset(self._graph.edge_list())

        # check for consistency
        if frozen_edge_keys != frozen_graph_edges:
            missing_in_LUT = frozen_graph_edges - frozen_edge_keys
            missing_in_graph = frozen_edge_keys - frozen_graph_edges
            raise ValueError(
                f"Edge length LUT is inconsistent with graph edges.\nMissing in LUT: {missing_in_LUT}\nMissing in graph: {missing_in_graph}"
            )

        return True

    # %% I/O
    def save(self, file_path: Optional[str] = None, compress: bool = True) -> None:
        if file_path is None or not os.path.isdir(
            os.path.dirname(os.path.abspath(file_path))
        ):
            if pyfd is not None:
                file_path = pyfd.call_save_as_file(
                    defaultextension=".npz",
                    initialfile="*.npz",
                    title="Select Surface Mesh NPZ File",
                    filetypes=[("NPZ files", "*.npz")],
                )
            else:
                raise ValueError(
                    "Invalid file path. Please provide a valid file path ending with .npz or ensure that pyfiledialog is installed for file dialog support."
                )

        if file_path and not file_path.endswith(".npz"):
            file_path += ".npz"

        # pack npz
        to_npz: Dict[str, Any] = {
            "vertices": self._vertices.to_dict(),
            "edges": self._edges.to_dict(),
            "edge_length_LUT": self._edge_length_LUT,
            "path_LUT": self._path_LUT,
            "distance_LUT": self._distance_LUT,
            "distance_matrix": self._distance_matrix,
        }

        # save npz
        if compress:
            np.savez_compressed(os.path.abspath(file_path), **to_npz)
        else:
            np.savez(os.path.abspath(file_path), **to_npz)

    def load(self, file_path: Optional[str] = None) -> None:
        if file_path is None or not os.path.isfile(os.path.abspath(file_path)):
            if pyfd is not None:
                file_path = pyfd.call_file(
                    title="Select Surface Mesh NPZ File",
                    filetypes=[("NPZ files", "*.npz")],
                )
            else:
                raise ValueError(
                    "Invalid file path. Please provide a valid file path ending with .npz or ensure that pyfiledialog is installed for file dialog support."
                )

        if file_path and not file_path.endswith(".npz"):
            raise ValueError("Invalid file type. Please select a .npz file.")

        # load npz
        loaded = np.load(os.path.abspath(file_path), allow_pickle=True)

        # unpack npz
        self._vertices = PointCloud.from_dict(loaded["vertices"].item())
        self._edges = UniqueSurfaceWireframe.from_dict(loaded["edges"].item())
        self._edge_length_LUT = loaded["edge_length_LUT"].item()
        self._path_LUT = loaded["path_LUT"].item()
        self._distance_LUT = loaded["distance_LUT"].item()
        self._distance_matrix = (
            loaded["distance_matrix"] if loaded["distance_matrix"].size > 1 else None
        )

        # make graph
        graph = rx.PyGraph()

        # populate graph with vertices
        graph.add_nodes_from(range(len(self._vertices.points)))

        # populate graph with edges
        if self._edge_length_LUT is None:
            self._edge_length_LUT = GeometryTransformer.edge_length_LUT_from(
                self._edges
            )
        graph.add_edges_from(
            [
                (line[0], line[1], self._edge_length_LUT[(line[0], line[1])])
                for line in self._edges.lines
            ]
        )

    # %% Dunder methods
    def __repr__(self) -> str:
        return f"SurfaceGraph with {len(self._vertices.points)} vertices and {len(self._edges.lines)} edges.\nThe graph is {'connected' if self.is_connected() else 'disconnected'} with {self.number_of_connected_components()} connected components. The graph is {'planar' if self.is_planar() else 'non-planar'} with a transitivity of {self.transitivity():.4f}.\nThe graph has {'negative edge weights' if self.has_negative_edge_weights() else 'no negative edge weights'}."

    def __str__(self) -> str:
        return self.__repr__()

    # %% Properties
    @property
    def vertices(self) -> PointCloud:
        return self._vertices

    @vertices.setter
    def vertices(self, value: PointCloud) -> None:
        raise ValueError(
            "Vertices is a read-only property and cannot be set directly. Please set the edges to update the vertices accordingly, as the vertices are derived from the edges in the current implementation of the surface graph."
        )

    @property
    def edges(self) -> Union[SurfaceWireframe, UniqueSurfaceWireframe]:
        return self._edges

    @edges.setter
    def edges(
        self,
        value: Union[SurfaceWireframe, UniqueSurfaceWireframe, o3d.geometry.LineSet],
    ) -> None:
        if isinstance(value, SurfaceWireframe):
            value = UniqueSurfaceWireframe.from_wireframe(value)
        elif isinstance(value, o3d.geometry.LineSet):
            value = UniqueSurfaceWireframe.from_wireframe(
                SurfaceWireframe.from_o3d(value)
            )
        elif not isinstance(value, UniqueSurfaceWireframe):
            raise ValueError(
                "Edges must be a SurfaceWireframe, UniqueSurfaceWireframe, or o3d.geometry.LineSet."
            )

        print("Setting edges and rebuilding graph...")

        # set edges
        self._edges = value

        # re-build the graph with the new edges
        self._vertices = PointCloud.from_numpy(self._edges.get_points())

        # make graph
        self._graph = rx.PyGraph()

        # populate graph with vertices
        self._graph.add_nodes_from(range(len(self._vertices.points)))

        # populate graph with edges
        self._edge_length_LUT = GeometryTransformer.edge_length_LUT_from(self._edges)
        self._graph.add_edges_from(
            [
                (line[0], line[1], self._edge_length_LUT[(line[0], line[1])])
                for line in self._edges.lines
            ]
        )

        # reset path and distance LUTs and matrix, as they are no longer valid with the new graph structure
        print("Resetting path and distance LUTs and matrix...")
        self._path_LUT = None
        self._distance_LUT = None
        self._distance_matrix = None

    @property
    def graph(self) -> rx.PyGraph:
        return self._graph

    @graph.setter
    def graph(self, value: rx.PyGraph) -> None:
        raise ValueError(
            "Graph is a read-only property and cannot be set directly. Please set the edges to update the graph."
        )

    @property
    def edge_length_LUT(self) -> Optional[Dict[Tuple[int, int], float]]:
        return self._edge_length_LUT

    @edge_length_LUT.setter
    def edge_length_LUT(self, value: Optional[Dict[Tuple[int, int], float]]) -> None:
        raise ValueError(
            "Edge length LUT is a read-only property and cannot be set directly. It is automatically computed from the edges and vertices."
        )

    @property
    def path_LUT(
        self,
    ) -> Optional[Union[Dict[int, Dict[int, List[int]]], rx.AllPairsPathMapping]]:
        return self._path_LUT

    @path_LUT.setter
    def path_LUT(
        self,
        value: Optional[Union[Dict[int, Dict[int, List[int]]], rx.AllPairsPathMapping]],
    ) -> None:
        raise ValueError(
            "Path LUT is a read-only property and cannot be set directly. It is automatically computed from the graph when computing shortest paths."
        )

    @property
    def distance_LUT(
        self,
    ) -> Optional[Union[Dict[int, Dict[int, float]], rx.AllPairsPathLengthMapping]]:
        return self._distance_LUT

    @distance_LUT.setter
    def distance_LUT(
        self,
        value: Optional[
            Union[Dict[int, Dict[int, float]], rx.AllPairsPathLengthMapping]
        ],
    ) -> None:
        raise ValueError(
            "Distance LUT is a read-only property and cannot be set directly. It is automatically computed from the graph when computing shortest path lengths."
        )

    @property
    def distance_matrix(self) -> Optional[np.ndarray]:
        return self._distance_matrix

    @distance_matrix.setter
    def distance_matrix(self, value: Optional[np.ndarray]) -> None:
        raise ValueError(
            "Distance matrix is a read-only property and cannot be set directly. It is automatically computed from the distance LUT when building the distance matrix."
        )
