import open3d as o3d
from overrides import override
from typing import List

# custom imports
from cellnav.core.containers.path_3d_interface import Path3dInterface

__all__ = ["Path3dRendered"]


class Path3dRendered(Path3dInterface):
    """
    A class to represent a 3D path in the Min3D framework.
    This class contains the geometry of the path as well as additional attributes for visualization purposes.

    A path is defined as a sequence of 3D points that represent a trajectory or curve in 3D space.
    In our case, we represent the path as a LineSet geometry in Open3D, which consists of a set of points and a set of edges that connect those points.
    Vertices on the path are visualized as 3D meshes of different geometry, with the start and end points highlighted in different colors for better visualization.
    """

    # base geometry attributes
    edges: o3d.geometry.LineSet

    # additional attributes for visualization
    start: o3d.geometry.TriangleMesh
    stop: o3d.geometry.TriangleMesh
    intermediary: List[o3d.geometry.TriangleMesh]

    @override
    def to_list(self) -> List[o3d.geometry.Geometry]:
        """Forward the geometry attributes of the Path3D object as a list of Open3D geometries for visualization purposes.

        Returns:
            List[o3d.geometry.Geometry]: A list of Open3D geometries that represent the path, including the edges and the start, end, and intermediate points.
        """
        return [self.edges, self.start, self.stop] + self.intermediary
