import open3d as o3d
from overrides import override
from typing import List

# custom imports
from cellnav.core.containers.path_3d_interface import Path3dInterface

__all__ = ["Path3dLight"]


class Path3dLight(Path3dInterface):
    """
    A class to represent a 3D path in the Min3D framework.
    This class contains the geometry of the path as well as additional attributes for visualization purposes.

    A path is defined as a sequence of 3D points that represent a trajectory or curve in 3D space.
    In our case, we represent the path as a LineSet geometry in Open3D, which consists of a set of points and a set of edges that connect those points.

    In this lightweight version of the Path3D class, we represent the start, end, and intermediate points as PointCloud geometries, reducing the computational cost.
    """

    # base geometry attributes
    edges: o3d.geometry.LineSet

    # additional attributes for visualization
    start: o3d.geometry.PointCloud
    stop: o3d.geometry.PointCloud
    intermediary: o3d.geometry.PointCloud

    @override
    def to_list(self) -> List[o3d.geometry.Geometry]:
        """Forward the geometry attributes of the Path3D object as a list of Open3D geometries for visualization purposes.

        Returns:
            List[o3d.geometry.Geometry]: A list of Open3D geometries that represent the path, including the edges and the start, end, and intermediate points.
        """
        combined_points = self.start + self.intermediary + self.stop + self.intermediary
        return [combined_points] + [self.edges]
