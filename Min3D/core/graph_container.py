## dependencies
import numpy as np
import rustworkx as rx
from dataclasses import dataclass
from typing import Dict, Tuple

# custom imports
from Min3D.core.point_cloud_container import PointCloudContainer
from Min3D.core.wireframe_container import WireframeContainer

# class implementation
@dataclass
class GraphContainer:
    # base data
    vertices: PointCloudContainer
    edges: WireframeContainer

    # graph representation
    graph: rx.PyGraph
    
    # lookup tables
    edge_length_LUT: Dict[Tuple[int, int], float]
    distance_LUT: Dict[int, Dict[int, float]]
    
    # optional table of node distances for fast access
    distance_matrix: np.ndarray
    
    def __repr__(self) -> str:
        return f"GraphContainer with {len(self.vertices.points)} vertices and {len(self.edges.lines)} edges."
    
    def __str__(self) -> str:
        return self.__repr__()