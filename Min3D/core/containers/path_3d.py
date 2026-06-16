import open3d as o3d
from typing import List
from dataclasses import dataclass

__all__ = ["Path3D"]


@dataclass
class Path3D:
    # base geometry attributes
    edges: o3d.geometry.LineSet

    # additional attributes for visualization
    start_point: o3d.geometry.TriangleMesh
    end_point: o3d.geometry.TriangleMesh
    intermediate_points: List[o3d.geometry.TriangleMesh]

    def to_list(self) -> List[o3d.geometry.Geometry]:
        return [self.edges, self.start_point, self.end_point] + self.intermediate_points
