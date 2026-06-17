## dependencies
import numpy as np
import open3d as o3d

from typing import Dict, List, Tuple

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
from cellnav.core.framework.point_cloud import PointCloud
from cellnav.core.framework.surface_mesh import SurfaceMesh
from cellnav.core.helpers.alpha_shape_helper import AlphaShapeHelper
from cellnav.core.framework.surface_wireframe import SurfaceWireframe
from cellnav.core.framework.unique_surface_wireframe import UniqueSurfaceWireframe


__all__= ["GeometryTransformer"]


## main class implementation - Cell membrane extraction tool
class GeometryTransformer:
    """A utility class for transforming between different geometry representations (point clouds, meshes, wireframes) and performing various geometric transformations."""

    def __init__(self) -> None:
        pass

    # %% Surface reconstruction - Point Cloud to Mesh
    @staticmethod
    def convex_hull_from(point_cloud: PointCloud) -> SurfaceMesh:
        """Construct a convex hull mesh from a given point cloud. Colors are preserved from the original point cloud.

        Args:
            point_cloud (PointCloud): The point cloud from which to construct the convex hull.

        Returns:
            SurfaceMesh: The constructed convex hull mesh.
        """
        hull, ind = point_cloud.geometry.compute_convex_hull()
        mesh = SurfaceMesh.from_o3d(hull)
        mesh.colors = o3d.utility.Vector3dVector(np.asarray(point_cloud.colors)[ind])
        return mesh

    @staticmethod
    def concave_hull_from(point_cloud: PointCloud, alpha: float) -> SurfaceMesh:
        """Construct a concave hull mesh from a given point cloud using alpha shapes. Colors are preserved from the original point cloud.

        Args:
            point_cloud (PointCloud): The point cloud from which to construct the concave hull.
            alpha (float): The alpha value for the alpha shape creation.

        Returns:
            SurfaceMesh: The constructed concave hull mesh.
        """
        # colors are automatically preserved in the alpha shape creation process, so we can just create the mesh and return it
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
        """
        Construct a watertight concave hull mesh from a given point cloud using iterative alpha shape creation.
        The function will keep increasing the alpha value until a watertight mesh is obtained or the maximum number of iterations is reached.

        Args:
            point_cloud (PointCloud): The point cloud from which to construct the watertight concave hull.
            alpha (float): The initial alpha value for the alpha shape creation.
            cluster_by (str, optional): The criterion for clustering points. Defaults to "area".
            max_iter (int, optional): The maximum number of iterations. Defaults to 10.
            alpha_increase_percentage (float, optional): The percentage by which to increase the alpha value in each iteration. Defaults to 0.2.

        Returns:
            SurfaceMesh: The constructed watertight concave hull mesh.
        """
        watertight_mesh = AlphaShapeHelper.iterate_until_watertight(
            point_cloud.geometry, alpha, cluster_by, max_iter, alpha_increase_percentage
        )
        return SurfaceMesh.from_o3d(watertight_mesh)

    @staticmethod
    def ball_pivoting_mesh_from(
        point_cloud: PointCloud, radii: List[float]
    ) -> SurfaceMesh:
        """
        Construct a mesh from a given point cloud using the Ball Pivoting algorithm.
        Not recommended for large point clouds due to computational intensity.

        Args:
            point_cloud (PointCloud): The point cloud from which to construct the mesh.
            radii (List[float]): The radii for the ball pivoting algorithm.

        Returns:
            SurfaceMesh: The constructed mesh.
        """
        return SurfaceMesh.from_o3d(
            o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                point_cloud.geometry, o3d.utility.DoubleVector(radii)
            )
        )

    # %% Surface reconstruction - Point Cloud to Wireframe
    @staticmethod
    def kNN_wireframe_from(point_cloud: PointCloud, k: int) -> SurfaceWireframe:
        """Build a wireframe from a point cloud using k-nearest neighbors (kNN) to determine connectivity. Each point is connected to its k nearest neighbors.

        Args:
            point_cloud (PointCloud): The point cloud from which to construct the wireframe.
            k (int): The number of nearest neighbors to consider.

        Returns:
            SurfaceWireframe: The constructed wireframe.
        """

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
        """Build a wireframe from a point cloud using radius-based nearest neighbors (radiusNN) to determine connectivity. Each point is connected to all points within the specified radius.

        Args:
            point_cloud (PointCloud): The point cloud from which to construct the wireframe.
            radius (float): The radius within which to consider points as neighbors.

        Returns:
            SurfaceWireframe: The constructed wireframe.
        """
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
        """Create a wireframe from a given mesh by extracting the edges of the mesh. Note that this will create duplicate edges for shared edges between triangles.

        Args:
            mesh (SurfaceMesh): The mesh from which to create the wireframe.

        Returns:
            SurfaceWireframe: The created wireframe.
        """
        line_set = o3d.geometry.LineSet.create_from_triangle_mesh(mesh.geometry)
        return SurfaceWireframe.from_o3d(line_set)

    @staticmethod
    def mesh_to_point_cloud(mesh: SurfaceMesh) -> PointCloud:
        """Create a point cloud from a given mesh by extracting the vertices of the mesh.

        Args:
            mesh (SurfaceMesh): The mesh from which to create the point cloud.

        Returns:
            PointCloud: The created point cloud.
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = mesh.vertices
        return PointCloud.from_o3d(pcd)

    @staticmethod
    def sample_uniformly_from(mesh: SurfaceMesh, num_points: int) -> PointCloud:
        """Create a point cloud by uniformly sampling points from the surface of a given mesh.

        Args:
            mesh (SurfaceMesh): The mesh from which to sample points.
            num_points (int): The number of points to sample.

        Returns:
            PointCloud: The created point cloud.
        """
        pcd = mesh.geometry.sample_points_uniformly(number_of_points=num_points)
        return PointCloud.from_o3d(pcd)

    @staticmethod
    def sample_poisson_disk_from(
        mesh: SurfaceMesh, num_points: int
    ) -> PointCloud:
        """
        Create a point cloud by sampling points from the surface of a given mesh using Poisson disk sampling.
        This ensures that points are more evenly distributed across the surface.

        Args:
            mesh (SurfaceMesh): The mesh from which to sample points.
            num_points (int): The number of points to sample.

        Returns:
            PointCloud: The created point cloud.
        """

        pcd = mesh.geometry.sample_points_poisson_disk(
            number_of_points=num_points
        )
        return PointCloud.from_o3d(pcd)

    # %% Wireframe Transformations
    @staticmethod
    def wireframe_to_point_cloud(wireframe: SurfaceWireframe) -> PointCloud:
        """Create a point cloud from a given wireframe by extracting the points (vertices) of the wireframe.

        Args:
            wireframe (SurfaceWireframe): The wireframe from which to create the point cloud.

        Returns:
            PointCloud: The created point cloud.
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = wireframe.points
        return PointCloud.from_o3d(pcd)

    @staticmethod
    def unique_wireframe_from(wireframe: SurfaceWireframe) -> SurfaceWireframe:
        """
        Create a unique wireframe from a given wireframe by removing duplicate edges.
        This is useful for visualizing the underlying structure of a mesh without redundant edges or
        when building a SurfaceGraph where duplicate edges would cause issues (computationally and conceptually).

        Args:
            wireframe (SurfaceWireframe): The wireframe from which to create the unique wireframe.

        Returns:
            SurfaceWireframe: The created unique wireframe.
        """
        return UniqueSurfaceWireframe.from_wireframe(wireframe)

    @staticmethod
    def edge_length_LUT_from(
        wireframe: UniqueSurfaceWireframe,
    ) -> Dict[Tuple[int, int], float]:
        """
        Create a lookup table (LUT) for edge lengths from a given unique wireframe.
        The LUT is a dictionary where the keys are tuples of vertex indices representing an edge,
        and the values are the corresponding edge lengths.

        This can be useful for algorithms that require quick access to edge lengths,
        such as pathfinding algorithms on surface graphs.

        Args:
            wireframe (UniqueSurfaceWireframe): The unique wireframe from which to create the edge length LUT.

        Returns:
            Dict[Tuple[int, int], float]: The created edge length lookup table.
        """
        edge_length_LUT: Dict[Tuple[int, int], float] = {}
        for edge in wireframe.lines:
            point1 = wireframe.points[edge[0]]
            point2 = wireframe.points[edge[1]]
            edge_length_LUT[(edge[0], edge[1])] = float(np.linalg.norm(point1 - point2))

        return edge_length_LUT
