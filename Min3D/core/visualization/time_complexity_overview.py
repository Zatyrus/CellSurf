# dependencies
import numpy as np
from PltStyler import PltStyler
import matplotlib.pyplot as plt
from typing import NoReturn, Union

__all__ = ["time_complexity_overview"]


# %% Time complexity functions for shortest path algorithms
def time_complexity_overview(
    max_V: int = 20000, max_E: int = 20000, return_fig: bool = False
) -> Union[plt.Figure, NoReturn]:
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
    return E + V * np.log(V)


def bellman_ford_time_complexity(E: float, V: float) -> float:
    return V * E


def floyd_warshall_time_complexity(E: float, V: float) -> float:
    return V**3


def astar_time_complexity(E: float, V: float) -> float:
    return E * np.log(V)
