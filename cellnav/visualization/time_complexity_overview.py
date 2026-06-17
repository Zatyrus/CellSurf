# dependencies
import numpy as np
from typing import Union
import matplotlib.figure
from PltStyler import PltStyler
import matplotlib.pyplot as plt

__all__ = ["time_complexity_overview"]


# %% Time complexity functions for shortest path algorithms
def time_complexity_overview(
    max_V: int = 5000, max_E: int = 5000, return_fig: bool = False
) -> Union[matplotlib.figure.Figure, None]:
    """
    A utility function to visualize the time complexity of different shortest path algorithms
    (Dijkstra, Bellman-Ford, Floyd-Warshall, A*) as a function of the number of vertices (V) and edges (E) in a graph.

    Args:
        max_V (int, optional): The maximum number of vertices in the graph. Defaults to 5000.
        max_E (int, optional): The maximum number of edges in the graph. Defaults to 5000.
        return_fig (bool, optional): Whether to return the matplotlib figure. Defaults to False.

    Returns:
        Union[matplotlib.figure.Figure, None]: The matplotlib figure object if return_fig is True, otherwise None.
    """

    # stylesheet for plotting
    PltStyler().set_stylesheet("bright").set_font(size=11).apply()

    fig, axs = plt.subplots(2, 1, figsize=(8, 5), dpi=100)
    fig.subplots_adjust(hspace=0.5)
    fig.suptitle("Time Complexity of Shortest Path Algorithms")

    E = np.full(100, max_E // 4)  # constant number of edges
    V = np.linspace(1000, max_V, 100)

    time_compelxities = {
        "Dijkstra": dijkstra_time_complexity(E, V),
        "Bellman-Ford": bellman_ford_time_complexity(E, V),
        "Floyd-Warshall": floyd_warshall_time_complexity(E, V),
        "A*": astar_time_complexity(E, V),
    }

    for algorithm, time_complexity in time_compelxities.items():
        axs[0].semilogy(
            V,
            time_complexity,
            label=algorithm,
            color=["r", "g", "b", "orange"][
                list(time_compelxities.keys()).index(algorithm)
            ],
        )

    axs[0].set_xlabel("Number of Vertices (V)")
    axs[0].set_ylabel("Time Complexity")
    axs[0].set_title(f"Constant Number of Edges (E={max_E // 4})")
    axs[0].legend(loc="upper right")
    axs[0].set_ylim(1e3, 1e15)

    E = np.linspace(1000, max_E, 100)
    V = np.full(100, max_V // 4)  # constant number of vertices

    time_compelxities = {
        "Dijkstra": dijkstra_time_complexity(E, V),
        "Bellman-Ford": bellman_ford_time_complexity(E, V),
        "Floyd-Warshall": floyd_warshall_time_complexity(E, V),
        "A*": astar_time_complexity(E, V),
    }

    for algorithm, time_complexity in time_compelxities.items():
        axs[1].semilogy(
            E,
            time_complexity,
            label=algorithm,
            color=["r", "g", "b", "orange"][
                list(time_compelxities.keys()).index(algorithm)
            ],
        )

    axs[1].set_xlabel("Number of Edges (E)")
    axs[1].set_ylabel("Time Complexity")
    axs[1].set_title(f"Constant Number of Vertices (V={max_V // 4})")
    axs[1].legend(loc="upper right")
    axs[1].set_ylim(1e3, 1e15)

    if return_fig:
        return fig
    else:
        plt.show()


def dijkstra_time_complexity(E: float, V: float) -> float:
    """Calculate the time complexity (big-O) of Dijkstra's algorithm given the number of edges (E) and vertices (V) in the graph.

    Args:
        E (float): The number of edges in the graph.
        V (float): The number of vertices in the graph.

    Returns:
        float: The time complexity of Dijkstra's algorithm.
    """
    return E + V * np.log(V)


def bellman_ford_time_complexity(E: float, V: float) -> float:
    """Calculate the time complexity (big-O) of Bellman-Ford algorithm given the number of edges (E) and vertices (V) in the graph.

    Args:
        E (float): The number of edges in the graph.
        V (float): The number of vertices in the graph.

    Returns:
        float: The time complexity of Bellman-Ford algorithm.
    """
    return V * E


def floyd_warshall_time_complexity(E: float, V: float) -> float:
    """Calculate the time complexity (big-O) of Floyd-Warshall algorithm given the number of edges (E) and vertices (V) in the graph.

    Args:
        E (float): The number of edges in the graph.
        V (float): The number of vertices in the graph.

    Returns:
        float: The time complexity of Floyd-Warshall algorithm.
    """
    return V**3


def astar_time_complexity(E: float, V: float) -> float:
    """Calculate the time complexity (big-O) of A* algorithm given the number of edges (E) and vertices (V) in the graph.

    Args:
        E (float): The number of edges in the graph.
        V (float): The number of vertices in the graph.

    Returns:
        float: The time complexity of A* algorithm.
    """
    return E * np.log(V)
