## dependencies
import numpy as np
import rustworkx as rx
from rustworkx.visualization import mpl_draw, graphviz_draw

from typing import Callable, Dict, List, Tuple, Union

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
from Min3D.core.framework.point_cloud import PointCloud
from Min3D.core.framework.surface_wireframe import SurfaceWireframe
from Min3D.core.framework.unique_surface_wireframe import UniqueSurfaceWireframe
from Min3D.core.util.geometry_transformation_tool import GeometryTransformationTool


# class implementation
class SurfaceGraph:
    # base data
    vertices: PointCloud
    edges: Union[SurfaceWireframe, UniqueSurfaceWireframe]

    # graph representation
    graph: rx.PyGraph

    # lookup tables for fast access to edge lengths and distances between nodes
    edge_length_LUT: Union[Dict[Tuple[int, int], float], None]

    # optional - can be computed on the fly if not provided, but can be useful to precompute for faster access
    distance_LUT: Union[Dict[int, Dict[int, float]], rx.AllPairsPathLengthMapping, None]
    path_LUT: Union[Dict[int, Dict[int, List[int]]], rx.AllPairsPathMapping, None]
    distance_matrix: Union[np.ndarray, None]  # table of node distances for fast access

    def __init__(
        self,
        vertices: PointCloud,
        edges: Union[SurfaceWireframe, UniqueSurfaceWireframe],
        graph: rx.PyGraph,
        edge_length_LUT: Union[Dict[Tuple[int, int], float], None] = None,
        distance_LUT: Union[
            Dict[int, Dict[int, float]], rx.AllPairsPathLengthMapping, None
        ] = None,
        path_LUT: Union[
            Dict[int, Dict[int, List[int]]], rx.AllPairsPathMapping, None
        ] = None,
        distance_matrix: Union[np.ndarray, None] = None,
        **kwargs,
    ) -> None:
        self.vertices = vertices
        self.edges = edges
        self.graph = graph

        # set optional LUTs and distance matrix
        self.edge_length_LUT = edge_length_LUT
        self.distance_LUT = distance_LUT
        self.path_LUT = path_LUT
        self.distance_matrix = distance_matrix

        # config options for decorators - currently a placeholder
        self.config = {"verbose": True, "DEBUG": False}

        # update from kwargs if provided
        for key in self.config.keys():
            if key in kwargs:
                self.config[key] = kwargs[key]

        # override if DEBUG
        if self.config["DEBUG"]:
            self.config["verbose"] = True

        # run post init
        self.__post_init__()

    def __post_init__(self) -> None:
        # check that vertices and edges are of the correct types
        if not isinstance(self.vertices, PointCloud):
            raise TypeError(f"Vertices must be a PointCloud, got {type(self.vertices)}")
        if not isinstance(self.edges, (SurfaceWireframe, UniqueSurfaceWireframe)):
            raise TypeError(
                f"Edges must be a SurfaceWireframe or UniqueSurfaceWireframe, got {type(self.edges)}"
            )
        if not isinstance(self.graph, rx.PyGraph):
            raise TypeError(
                f"Graph must be a rustworkx PyGraph, got {type(self.graph)}"
            )

        # convert SurfaceWireframe to UniqueSurfaceWireframe if necessary,
        # to ensure consistent edge representation and avoid issues with duplicate edges in graph construction
        if isinstance(self.edges, SurfaceWireframe):
            self.edges = UniqueSurfaceWireframe.from_wireframe(self.edges)

        # make edge_length_LUT and check consistency if not provided - this is necessary for graph construction and shortest path algorithms
        if self.edge_length_LUT is None:
            self.edge_length_LUT = GeometryTransformationTool.edge_length_LUT_from(
                self.edges
            )
        self.check_consistency_of_edge_length_LUT()

    @classmethod
    def from_wireframe(
        cls, wireframe: Union[SurfaceWireframe, UniqueSurfaceWireframe], **kwargs
    ) -> "SurfaceGraph":
        # make vertices
        edges = UniqueSurfaceWireframe.from_wireframe(wireframe)
        vertices = PointCloud.from_numpy(edges.get_points())

        # make graph
        graph = rx.PyGraph()

        # populate graph with vertices
        graph.add_nodes_from(range(len(vertices.points)))

        # populate graph with edges
        edge_length_LUT = GeometryTransformationTool.edge_length_LUT_from(edges)
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
        # build kNN graph from point cloud
        edges = GeometryTransformationTool.kNN_wireframe_from(point_cloud, k=k)
        return cls.from_wireframe(edges, **kwargs)

    # %% Utility
    def get_vertices(self) -> PointCloud:
        return self.vertices

    def get_edges(self) -> Union[SurfaceWireframe, UniqueSurfaceWireframe]:
        return self.edges

    def get_graph(self) -> Union[rx.PyGraph, None]:
        return self.graph

    def get_edge_length_LUT(self) -> Union[Dict[Tuple[int, int], float], None]:
        return self.edge_length_LUT

    def get_distance_LUT(
        self,
    ) -> Union[Dict[int, Dict[int, float]], rx.AllPairsPathLengthMapping, None]:
        return self.distance_LUT

    def get_path_LUT(
        self,
    ) -> Union[Dict[int, Dict[int, List[int]]], rx.AllPairsPathMapping, None]:
        return self.path_LUT

    # %% Path and distance queries
    def get_shortest_path(self, source: int, target: int, **kwargs) -> List[int]:
        if self.path_LUT is None:
            return list(self.dijkstra_shortest_path(source, target, **kwargs)[target])
        return list(self.path_LUT[source][target])  # type: ignore

    def get_shortest_distance(self, source: int, target: int, **kwargs) -> float:
        if self.distance_matrix is not None:
            return self.distance_matrix[source, target]
        elif self.distance_LUT is not None:
            return self.distance_LUT[source][target]  # type: ignore
        else:
            return self.dijkstra_shortest_distance(source, target, **kwargs)[target]  # type: ignore

    # %% Graph parameters
    def transitivity(self) -> float:
        return rx.transitivity(self.graph)

    def is_planar(self) -> bool:
        return rx.is_planar(self.graph)

    def is_connected(self) -> bool:
        return rx.is_connected(self.graph)

    def number_of_connected_components(self) -> int:
        return rx.number_connected_components(self.graph)

    def has_negative_edge_weights(self) -> bool:
        return bool(np.any(np.array(self.graph.edges()) < 0))

    # %% Graph visualization
    def draw_graph_with_matplotlib(self, **kwargs) -> None:
        try:
            mpl_draw(self.graph, **kwargs)
        except ImportError:
            raise ImportError(
                "Matplotlib is not installed. Please install matplotlib to use this function."
            )

    def draw_graph_with_graphviz(self, **kwargs) -> None:
        try:
            graphviz_draw(self.graph, **kwargs)
        except ImportError:
            raise ImportError(
                "Graphviz is not installed. Please install graphviz to use this function."
            )

    # %% Shortest path algorithm - source -> target
    def dijkstra_shortest_path(
        self,
        source: int,
        target: int,
        weight_fn: Callable = float,
        default_weight: float = 1.0,
    ) -> rx.PathMapping:
        if self.has_negative_edge_weights():
            raise ValueError(
                "Graph contains negative edge weights. Dijkstra's algorithm cannot be used.\nConsider using Bellman-Ford algorithm instead."
            )
        return rx.dijkstra_shortest_paths(
            self.graph,
            source=source,
            target=target,
            weight_fn=weight_fn,
            default_weight=default_weight,
        )

    def bellman_ford_shortest_path(
        self,
        source: int,
        target: int,
        weight_fn: Callable = float,
        default_weight: float = 1.0,
    ) -> rx.PathMapping:
        return rx.bellman_ford_shortest_paths(
            self.graph,
            source=source,
            target=target,
            weight_fn=weight_fn,
            default_weight=default_weight,
        )

    # %% Shortest distance algorithm - source -> target
    def dijkstra_shortest_distance(
        self, source: int, target: int, weight_fn: Callable = float
    ) -> rx.PathLengthMapping:
        if self.has_negative_edge_weights():
            raise ValueError(
                "Graph contains negative edge weights. Dijkstra's algorithm cannot be used.\nConsider using Bellman-Ford algorithm instead."
            )
        return rx.dijkstra_shortest_path_lengths(
            self.graph, node=source, goal=target, edge_cost_fn=weight_fn
        )

    def bellman_ford_shortest_distance(
        self, source: int, target: int, weight_fn: Callable = float
    ) -> rx.PathLengthMapping:
        return rx.bellman_ford_shortest_path_lengths(
            self.graph, node=source, goal=target, edge_cost_fn=weight_fn
        )

    # %% Shortest path algorithm - all pairs
    def dijkstra_shortest_path_global(
        self, weight_fn: Callable = float
    ) -> rx.AllPairsPathMapping:
        if self.has_negative_edge_weights():
            raise ValueError(
                "Graph contains negative edge weights. Dijkstra's algorithm cannot be used.\nConsider using Bellman-Ford algorithm instead."
            )
        self.path_LUT = rx.all_pairs_dijkstra_shortest_paths(
            self.graph, edge_cost_fn=weight_fn
        )
        return self.path_LUT

    def bellman_ford_shortest_path_global(
        self, weight_fn: Callable = float
    ) -> rx.AllPairsPathMapping:
        self.path_LUT = rx.all_pairs_bellman_ford_shortest_paths(
            self.graph, edge_cost_fn=weight_fn
        )
        return self.path_LUT

    # %% Shortest distance algorithm - all pairs
    def dijkstra_shortest_distance_global(
        self, weight_fn: Callable = float
    ) -> rx.AllPairsPathLengthMapping:
        if self.has_negative_edge_weights():
            raise ValueError(
                "Graph contains negative edge weights. Dijkstra's algorithm cannot be used.\nConsider using Bellman-Ford algorithm instead."
            )
        self.distance_LUT = rx.all_pairs_dijkstra_path_lengths(
            self.graph, edge_cost_fn=weight_fn
        )
        return self.distance_LUT

    def bellman_ford_shortest_distance_global(
        self, weight_fn: Callable = float
    ) -> rx.AllPairsPathLengthMapping:
        self.distance_LUT = rx.all_pairs_bellman_ford_path_lengths(
            self.graph, edge_cost_fn=weight_fn
        )
        return self.distance_LUT

    def floyd_warshall_shortest_paths_global(
        self,
        weight_fn: Callable = float,
        default_weight: float = 1.0,
        parallel_threshold: int = 300,
    ) -> rx.AllPairsPathLengthMapping:
        self.path_length_LUT = rx.floyd_warshall(
            self.graph,
            weight_fn=weight_fn,
            default_weight=default_weight,
            parallel_threshold=parallel_threshold,
        )
        return self.path_length_LUT

    # %% Helper functions
    def suggest_shortest_path_algorithm(self) -> None:
        E: float = self.edges.get_lines().shape[0]
        V: float = self.vertices.get_points().shape[0]

        # Compute the time complexity of each implemented algorithm
        def dijkstra_time_complexity(E: float, V: float) -> float:
            return E + V * np.log(V)

        def bellman_ford_time_complexity(E: float, V: float) -> float:
            return V * E

        def floyd_warshall_time_complexity(E: float, V: float) -> float:
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
        distance_LUT: Union[
            Dict[int, Dict[int, float]], rx.AllPairsPathLengthMapping, None
        ] = None,
        silent: bool = False,
    ) -> np.ndarray:
        if distance_LUT is None:
            if self.distance_LUT is None:
                raise ValueError(
                    "Distance LUT must be provided or precomputed before building distance matrix."
                )
            distance_LUT = self.distance_LUT

        # initialize distance matrix with zeros
        self.distance_matrix = np.zeros(
            (self.graph.num_nodes(), self.graph.num_nodes())
        )
        # compute the distance matrix from the shortest path LUT - with progress bar
        with tqdm(
            total=self.graph.num_nodes(),
            desc="Building distance matrix from LUT",
            disable=silent,
        ) as pbar:
            for i in range(self.graph.num_nodes()):
                # triangular indexing to avoid redundant computations, as the distance matrix is symmetric and the diagonal is zero
                self.distance_matrix[i, i + 1 :] = np.array(
                    list(distance_LUT[i].values())
                )[i:]  # we need to start +1 to skip the diagonal, which is zero.
                pbar.update(1)

        # make the distance matrix symmetric
        self.distance_matrix = self.distance_matrix + self.distance_matrix.T

        return self.distance_matrix

    def build_distance_LUT_from_matrix(
        self, distance_matrix: np.ndarray, silent: bool = False
    ) -> Dict[int, Dict[int, float]]:
        distance_LUT: Dict[int, Dict[int, float]] = {}
        with tqdm(
            total=distance_matrix.shape[0],
            desc="Building distance LUT from matrix",
            disable=silent,
        ) as pbar:
            for i in range(distance_matrix.shape[0]):
                distance_LUT[i] = {
                    j: distance_matrix[i, j]
                    for j in range(distance_matrix.shape[1])
                    if i != j
                }
                pbar.update(1)
        self.distance_LUT = distance_LUT
        return self.distance_LUT

    def path_to_distance(self, path: list) -> float:
        # catch self.edge_length_LUT is None
        if self.edge_length_LUT is None:
            raise ValueError(
                "Edge length LUT must be computed before converting path to distance."
            )

        distance = 0.0
        for i in range(len(path) - 1):
            edge = (path[i], path[i + 1])
            if edge in self.edge_length_LUT.keys():
                distance += self.edge_length_LUT[edge]
            elif (
                edge[1],
                edge[0],
            ) in self.edge_length_LUT:  # check for undirected edge
                distance += self.edge_length_LUT.get((edge[1], edge[0]), 0.0)
            else:
                raise ValueError(f"Edge {edge} not found in edge length LUT.")
        return distance

    def path_LUT_to_distance_LUT(self) -> Dict[int, Dict[int, float]]:
        # catch self.edge_length_LUT is None
        if self.path_LUT is None:
            raise ValueError(
                "Path LUT must be computed before converting to distance LUT."
            )

        distance_LUT: Dict[int, Dict[int, float]] = {}
        for source, target_paths in self.path_LUT.items():
            distance_LUT[source] = {}
            for target, path in target_paths.items():
                distance_LUT[source][target] = self.path_to_distance(path)  # type: ignore
        self.distance_LUT = distance_LUT
        return self.distance_LUT

    # %% Check-up functions
    def check_consistency_of_edge_length_LUT(self) -> bool:
        if self.edge_length_LUT is None:
            raise ValueError(
                "Edge length LUT must be computed before checking consistency."
            )

        # check that all edges in the graph are present in the edge_length_LUT and vice versa
        # create frozensets of the edge keys in the LUT and the graph edges for efficient comparison
        # frozensets are used to ensure that the order of edges does not affect the comparison
        frozen_edge_keys = frozenset(self.edge_length_LUT.keys())
        frozen_graph_edges = frozenset(self.graph.edge_list())

        # check for consistency
        if frozen_edge_keys != frozen_graph_edges:
            missing_in_LUT = frozen_graph_edges - frozen_edge_keys
            missing_in_graph = frozen_edge_keys - frozen_graph_edges
            raise ValueError(
                f"Edge length LUT is inconsistent with graph edges.\nMissing in LUT: {missing_in_LUT}\nMissing in graph: {missing_in_graph}"
            )

        return True

    # %% Dunder methods
    def __repr__(self) -> str:
        return f"SurfaceGraph with {len(self.vertices.points)} vertices and {len(self.edges.lines)} edges.\nThe graph is {'connected' if self.is_connected() else 'disconnected'} with {self.number_of_connected_components()} connected components. The graph is {'planar' if self.is_planar() else 'non-planar'} with a transitivity of {self.transitivity():.4f}.\nThe graph has {'negative edge weights' if self.has_negative_edge_weights() else 'no negative edge weights'}."

    def __str__(self) -> str:
        return self.__repr__()
