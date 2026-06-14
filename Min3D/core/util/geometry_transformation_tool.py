## dependencies
import numpy as np
import open3d as o3d

from typing import Dict, List, Tuple, Union

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

## custom dependencies
from Min3D.core.framework.point_cloud import PointCloud
from Min3D.core.framework.surface_mesh import SurfaceMesh
from Min3D.core.helpers.alpha_shape_helper import AlphaShapeHelper
from Min3D.core.framework.surface_wireframe import SurfaceWireframe
from Min3D.core.framework.unique_surface_wireframe import UniqueSurfaceWireframe


## main class implementation - Cell membrane extraction tool
class GeometryTransformationTool:
    def __init__(self) -> None:
        # run post init
        self.__post_init__()

    def __post_init__(self) -> None:
        pass

    # %% Surface reconstruction - Point Cloud to Mesh
    @staticmethod
    def convex_hull_from(point_cloud: PointCloud) -> SurfaceMesh:
        hull, _ = point_cloud.geometry.compute_convex_hull()
        return SurfaceMesh.from_o3d(hull)

    @staticmethod
    def concave_hull_from(point_cloud: PointCloud, alpha: float) -> SurfaceMesh:
        concave_hull = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            point_cloud.geometry, alpha
        )
        return SurfaceMesh.from_o3d(concave_hull)

    @staticmethod
    def watertight_concave_hull_from(
        point_cloud: PointCloud,
        alpha: float,
        cluster_by: str = "area",
        max_iter: int = 10,
        alpha_increase_percentage: float = 0.2,
    ) -> SurfaceMesh:
        watertight_mesh = AlphaShapeHelper.iterate_until_watertight(
            point_cloud.geometry, alpha, cluster_by, max_iter, alpha_increase_percentage
        )
        return SurfaceMesh.from_o3d(watertight_mesh)

    @staticmethod
    def ball_pivoting_mesh_from(
        point_cloud: PointCloud, radii: List[float]
    ) -> SurfaceMesh:
        return SurfaceMesh.from_o3d(
            o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                point_cloud.geometry, o3d.utility.DoubleVector(radii)
            )
        )

    # %% Surface reconstruction - Point Cloud to Wireframe
    @staticmethod
    def kNN_wireframe_from(point_cloud: PointCloud, k: int) -> SurfaceWireframe:
        # get kNN indices
        ind, _ = point_cloud.kNN_search(k)

        # assemble lines from the knn indices
        lines_out = []
        with tqdm(total=len(ind), desc="Creating lines from knn indices") as pbar:
            for lines in ind:
                for j in range(1, len(lines)):
                    lines_out.append([lines[0], lines[j]])
                lines_out.append([lines[-1], lines[0]])
                pbar.update(1)

        # build line set
        kNN_line_set = o3d.geometry.LineSet()
        kNN_line_set.points = point_cloud.points
        kNN_line_set.lines = o3d.utility.Vector2iVector(np.array(lines_out))

        return SurfaceWireframe.from_o3d(kNN_line_set)

    @staticmethod
    def radiusNN_wireframe_from(
        point_cloud: PointCloud, radius: float
    ) -> SurfaceWireframe:
        # get radiusNN indices
        ind, _, split = point_cloud.radiusNN_search(radius)
        ind = [ind[start_i:end_i] for start_i, end_i in zip(split, split[1:])]

        # assemble lines from the knn indices
        lines_out = []
        with tqdm(total=len(ind), desc="Creating lines from knn indices") as pbar:
            for lines in ind:
                for j in range(1, len(lines)):
                    lines_out.append([lines[0], lines[j]])
                lines_out.append([lines[-1], lines[0]])
                pbar.update(1)

        # build line set
        kNN_line_set = o3d.geometry.LineSet()
        kNN_line_set.points = point_cloud.points
        kNN_line_set.lines = o3d.utility.Vector2iVector(np.array(lines_out))

        return SurfaceWireframe.from_o3d(kNN_line_set)

    # %% Mesh Transformations
    @staticmethod
    def mesh_to_wireframe(mesh: SurfaceMesh) -> SurfaceWireframe:
        line_set = o3d.geometry.LineSet.create_from_triangle_mesh(mesh.geometry)
        return SurfaceWireframe.from_o3d(line_set)

    @staticmethod
    def mesh_to_point_cloud(mesh: SurfaceMesh) -> PointCloud:
        pcd = o3d.geometry.PointCloud()
        pcd.points = mesh.vertices
        return PointCloud.from_o3d(pcd)

    @staticmethod
    def sample_uniformly_from(mesh: SurfaceMesh, number_of_points: int) -> PointCloud:
        pcd = mesh.geometry.sample_points_uniformly(number_of_points=number_of_points)
        return PointCloud.from_o3d(pcd)

    @staticmethod
    def sample_poisson_disk_from(
        mesh: SurfaceMesh, number_of_points: int
    ) -> PointCloud:
        pcd = mesh.geometry.sample_points_poisson_disk(
            number_of_points=number_of_points
        )
        return PointCloud.from_o3d(pcd)

    # %% Wireframe Transformations
    @staticmethod
    def wireframe_to_point_cloud(wireframe: SurfaceWireframe) -> PointCloud:
        pcd = o3d.geometry.PointCloud()
        pcd.points = wireframe.points
        return PointCloud.from_o3d(pcd)

    @staticmethod
    def unique_wireframe_from(wireframe: SurfaceWireframe) -> SurfaceWireframe:
        return UniqueSurfaceWireframe.from_wireframe(wireframe)

    @staticmethod
    def unpack_to_vertices_and_edges(
        wireframe: SurfaceWireframe,
    ) -> Tuple[PointCloud, Union[SurfaceWireframe, UniqueSurfaceWireframe]]:
        return PointCloud.from_o3d(wireframe.geometry.points), wireframe

    @staticmethod
    def edge_length_LUT_from(
        wireframe: UniqueSurfaceWireframe,
    ) -> Dict[Tuple[int, int], float]:
        edge_length_LUT: Dict[Tuple[int, int], float] = {}
        for edge in wireframe.lines:
            point1 = wireframe.points[edge[0]]
            point2 = wireframe.points[edge[1]]
            edge_length_LUT[(edge[0], edge[1])] = float(np.linalg.norm(point1 - point2))

        return edge_length_LUT
