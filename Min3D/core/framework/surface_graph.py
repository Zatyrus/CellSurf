## dependencies
import numpy as np
import rustworkx as rx
from dataclasses import dataclass
from rustworkx.visualization import mpl_draw, graphviz_draw

from typing import Callable, Dict, List, NoReturn, Tuple, Union

# tqdm for progress bars - automatically selects the right version for notebooks vs. terminal
from IPython import get_ipython

try:
    ipy_str = str(type(get_ipython()))
    if "zmqshell" in ipy_str:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
except ImportError:
    from tqdm import tqdm

# custom imports
from Min3D.core.framework.surface_wireframe import SurfaceWireframe
from Min3D.core.framework.point_cloud import PointCloud
from Min3D.core.framework.unique_surface_wireframe import UniqueSurfaceWireframe


# class implementation
@dataclass
class SurfaceGraph:
    # base data
    vertices: PointCloud
    edges: Union[SurfaceWireframe, UniqueSurfaceWireframe]

    # graph representation
    graph: rx.PyGraph = None

    # lookup tables for fast access to edge lengths and distances between nodes
    edge_length_LUT: Dict[Tuple[int, int], float] = None
    # optional - can be computed on the fly if not provided, but can be useful to precompute for faster access
    distance_LUT: Dict[int, Dict[int, float]] = None
    path_LUT: Dict[int, Dict[int, list]] = None

    # optional
    distance_matrix: np.ndarray = None  # table of node distances for fast access

    def __post_init__(self) -> NoReturn:
        # check that vertices and edges are of the correct types
        if not isinstance(self.vertices, PointCloud):
            raise TypeError(f"Vertices must be a PointCloud, got {type(self.vertices)}")
        if not isinstance(self.edges, (SurfaceWireframe, UniqueSurfaceWireframe)):
            raise TypeError(
                f"Edges must be a SurfaceWireframe or UniqueSurfaceWireframe, got {type(self.edges)}"
            )

        # convert SurfaceWireframe to UniqueSurfaceWireframe if necessary, to ensure consistent edge representation and avoid issues with duplicate edges in graph construction
        if isinstance(self.edges, SurfaceWireframe):
            self.edges = UniqueSurfaceWireframe.from_wireframe_container(self.edges)

        # make edge_length_LUT
        self.edge_length_LUT = self.edge_length_LUT_from(self.edges)

        # make graph
        self.graph = rx.PyGraph()

        # populate graph with vertices
        self.graph.add_nodes_from(range(len(self.vertices)))

        # populate graph with edges
        self.graph.add_edges_from(
            [
                (line[0], line[1], self.edge_length_LUT[(line[0], line[1])])
                for line in self.edges.lines
            ]
        )

    # %% Utility
    def get_vertices(self) -> PointCloud:
        return self.vertices

    def get_edges(self) -> UniqueSurfaceWireframe:
        return self.edges

    def get_graph(self) -> rx.PyGraph:
        return self.graph

    def get_edge_length_LUT(self) -> Dict[Tuple[int, int], float]:
        return self.edge_length_LUT

    def get_distance_LUT(self) -> Dict[int, Dict[int, float]]:
        return self.distance_LUT

    def get_path_LUT(self) -> Dict[int, Dict[int, List[int]]]:
        return self.path_LUT

    # %% Path and distance queries
    def get_shortest_path(self, source: int, target: int, **kwargs) -> List[int]:
        if self.path_LUT is None:
            return list(self.dijkstra_shortest_path(source, target, **kwargs)[target])
        return list(self.path_LUT[source][target])

    def get_shortest_distance(self, source: int, target: int, **kwargs) -> float:
        if self.distance_matrix is not None:
            return self.distance_matrix[source, target]
        elif self.distance_LUT is not None:
            return self.distance_LUT[source][target]
        else:
            return self.dijkstra_shortest_distance(source, target, **kwargs)[target]

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
        return np.any(np.array(self.graph.edges()) < 0)

    # %% Graph visualization
    def draw_graph_with_matplotlib(self, **kwargs) -> NoReturn:
        mpl_draw(self.graph, **kwargs)

    def draw_graph_with_graphviz(self, **kwargs) -> NoReturn:
        graphviz_draw(self.graph, **kwargs)

    # %% Shortest path algorithm - source -> target
    def dijkstra_shortest_path(
        self,
        source: int,
        target: int,
        weight_fn: Callable = float,
        default_weight: float = 1.0,
    ) -> Dict[int, Dict[int, float]]:
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
    ) -> Dict[int, Dict[int, float]]:
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
    ) -> Dict[int, Dict[int, float]]:
        if self.has_negative_edge_weights():
            raise ValueError(
                "Graph contains negative edge weights. Dijkstra's algorithm cannot be used.\nConsider using Bellman-Ford algorithm instead."
            )
        return rx.dijkstra_shortest_path_lengths(
            self.graph, node=source, goal=target, edge_cost_fn=weight_fn
        )

    def bellman_ford_shortest_distance(
        self, source: int, target: int, weight_fn: Callable = float
    ) -> Dict[int, Dict[int, float]]:
        return rx.bellman_ford_shortest_path_lengths(
            self.graph, node=source, goal=target, edge_cost_fn=weight_fn
        )

    # %% Shortest path algorithm - all pairs
    def dijkstra_shortest_path_global(
        self, weight_fn: Callable = float
    ) -> Dict[int, Dict[int, float]]:
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
    ) -> Dict[int, Dict[int, float]]:
        self.path_LUT = rx.all_pairs_bellman_ford_shortest_paths(
            self.graph, edge_cost_fn=weight_fn
        )
        return self.path_LUT

    def floyd_warshall_shortest_paths_global(
        self,
        weight_fn: Callable = float,
        default_weight: float = 1.0,
        parallel_threshold: int = 300,
    ) -> Dict[int, Dict[int, float]]:
        self.path_LUT = rx.floyd_warshall(
            self.graph,
            weight_fn=weight_fn,
            default_weight=default_weight,
            parallel_threshold=parallel_threshold,
        )
        return self.path_LUT

    # %% Shortest distance algorithm - all pairs
    def dijkstra_shortest_distance_global(
        self, weight_fn: Callable = float
    ) -> Dict[int, Dict[int, float]]:
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
    ) -> Dict[int, Dict[int, float]]:
        self.distance_LUT = rx.all_pairs_bellman_ford_path_lengths(
            self.graph, edge_cost_fn=weight_fn
        )
        return self.distance_LUT

    # %% Helper functions
    def edge_length_LUT_from(
        self, wireframe: UniqueSurfaceWireframe
    ) -> Dict[Tuple[int, int], float]:
        edge_length_LUT: Dict[Tuple[int, int], float] = {}
        for edge in wireframe.lines:
            point1 = wireframe.points[edge[0]]
            point2 = wireframe.points[edge[1]]
            edge_length_LUT[(edge[0], edge[1])] = np.linalg.norm(point1 - point2)

        return edge_length_LUT

    def suggest_shortest_path_algorithm(self) -> NoReturn:
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
        most_efficient = min(complexities, key=complexities.get)
        print(
            f"\nBased on the time complexities, the {most_efficient} algorithm is the most efficient for this graph."
        )

    def build_distance_matrix(
        self,
        distance_LUT: Union[Dict[int, Dict[int, float]], None] = None,
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
        distance = 0.0
        for i in range(len(path) - 1):
            edge = (path[i], path[i + 1])
            if edge in self.edge_length_LUT:
                distance += self.edge_length_LUT[edge]
            elif (
                edge[1],
                edge[0],
            ) in self.edge_length_LUT:  # check for undirected edge
                distance += self.edge_length_LUT[(edge[1], edge[0])]
            else:
                raise ValueError(f"Edge {edge} not found in edge length LUT.")
        return distance

    def path_LUT_to_distance_LUT(self) -> Dict[int, Dict[int, float]]:
        distance_LUT: Dict[int, Dict[int, float]] = {}
        for source, target_paths in self.path_LUT.items():
            distance_LUT[source] = {}
            for target, path in target_paths.items():
                distance_LUT[source][target] = self.path_to_distance(path)
        self.distance_LUT = distance_LUT
        return self.distance_LUT

    # %% Dunder methods
    def __repr__(self) -> str:
        return f"SurfaceGraph with {len(self.vertices.points)} vertices and {len(self.edges.lines)} edges.\nThe graph is {'connected' if self.is_connected() else 'disconnected'} with {self.number_of_connected_components()} connected components. The graph is {'planar' if self.is_planar() else 'non-planar'} with a transitivity of {self.transitivity():.4f}.\nThe graph has {'negative edge weights' if self.has_negative_edge_weights() else 'no negative edge weights'}."

    def __str__(self) -> str:
        return self.__repr__()
