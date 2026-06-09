## dependencies
import open3d as o3d
import rustworkx as rx
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple

# class implementation
@dataclass
class GraphContainer:
    # base data
    vertices: o3d.utility.PointCloud
    edges: o3d.utility.LineSet

    # graph representation
    graph: rx.PyGraph
    
    # lookup tables
    edge_length_LUT: Dict[Tuple[int, int], float]
    distance_LUT: Dict[int, Dict[int, float]]
    
    # optional table of node distances for fast access
    distance_matrix: np.ndarray