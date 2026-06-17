## dependencies

import numpy as np
import open3d as o3d
import pymeshfix as pmf

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


class AlphaShapeHelper:
    """
    Helper class for working with alpha shapes, including functions for checking watertightness, repairing meshes, and iterapy_mesh_fixg until a watertight mesh is found.
    Methods this class implements are meant for internal use onyl, and are not intended to be called directly by users.
    They are used as part of the process of generapy_mesh_fixg a watertight mesh from a point cloud using alpha shapes.

    Please refer to the SurfaceMesh class for generapy_mesh_fixg a watertight mesh from a point cloud, which internally uses the methods of this helper class.

    Raises:
        NotImplementedError: When an unsupported criterion for clustering is provided.
        ValueError: When the mesh cannot be made watertight.
        ValueError: When the point cloud is empty.
    """

    @staticmethod
    def check_watertight(mesh: o3d.geometry.TriangleMesh) -> bool:
        """Check if a given mesh is watertight, meaning it has no holes and is a closed surface.

        Args:
            mesh (o3d.geometry.TriangleMesh): The mesh to check for watertightness.

        Returns:
            bool: True if the mesh is watertight, False otherwise.
        """
        return mesh.is_watertight()

    @staticmethod
    def cluster_mesh(
        mesh: o3d.geometry.TriangleMesh, cluster_by: str = "area"
    ) -> o3d.geometry.TriangleMesh:
        """
        Cluster the triangles of a mesh based on a specified criterion.
        The function clusters the triangles of the mesh and retains only the largest cluster based on the specified criterion, which can be either "area" or "vertices".

        Args:
            mesh (o3d.geometry.TriangleMesh): The mesh to cluster.
            cluster_by (str, optional): The criterion to use for clustering. Choices are "area" or "vertices". Defaults to "area".

        Raises:
            NotImplementedError: When an unsupported criterion for clustering is provided.

        Returns:
            o3d.geometry.TriangleMesh: The clustered mesh.
        """
        triangle_clusters, cluster_n_triangles, cluster_tri_areas = (
            mesh.cluster_connected_triangles()
        )
        if cluster_by == "area":
            criterion = cluster_tri_areas
        elif cluster_by == "vertices":
            criterion = cluster_n_triangles
        else:
            raise NotImplementedError(
                "The provided criterion for clustering is not implemented."
            )

        triangles_to_remove = triangle_clusters != np.asarray(criterion).argmax()
        mesh.remove_triangles_by_mask(triangles_to_remove)

        return mesh

    @staticmethod
    def clean_mesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
        """Clean the mesh by removing degenerate triangles, duplicated triangles, duplicated vertices, and non-manifold edges.

        Args:
            mesh (o3d.geometry.TriangleMesh): The mesh to clean.

        Returns:
            o3d.geometry.TriangleMesh: The cleaned mesh.
        """
        mesh = mesh.remove_degenerate_triangles()
        mesh = mesh.remove_duplicated_triangles()
        mesh = mesh.remove_duplicated_vertices()
        mesh = mesh.remove_non_manifold_edges()
        return mesh

    @staticmethod
    def repair_mesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
        """Repair the mesh using the PyMeshFix library, which attempts to fill holes and remove small components to create a more watertight mesh.

        Args:
            mesh (o3d.geometry.TriangleMesh): The mesh to repair.

        Returns:
            o3d.geometry.TriangleMesh: The repaired mesh.
        """

        # setup the PyMeshFix object and load the mesh vertices and triangles into it
        py_mesh_fix = pmf.PyTMesh()
        py_mesh_fix.load_array(np.asarray(mesh.vertices), np.asarray(mesh.triangles))

        py_mesh_fix.join_closest_components()  # first round of repairing
        py_mesh_fix.fill_small_boundaries()
        py_mesh_fix.remove_smallest_components()  # abrasive repairing

        py_mesh_fix.join_closest_components()  # second round of repairing
        py_mesh_fix.fill_small_boundaries()

        vclean, fclean = py_mesh_fix.return_arrays()

        return o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(vclean), o3d.utility.Vector3iVector(fclean)
        )

    @staticmethod
    def make_watertight(
        mesh: o3d.geometry.TriangleMesh, cluster_by: str = "area"
    ) -> o3d.geometry.TriangleMesh:
        """Attempt to make a mesh watertight by first checking if it is already watertight,
        and if not, cleaning it, repairing it, and clustering it to retain only the largest cluster.

        Args:
            mesh (o3d.geometry.TriangleMesh): The mesh to make watertight.
            cluster_by (str, optional): The criterion to use for clustering. Defaults to "area".

        Raises:
            ValueError: If the mesh cannot be made watertight.

        Returns:
            o3d.geometry.TriangleMesh: The watertight mesh.
            If the original mesh is already watertight, it is returned as is.
            Otherwise, the cleaned, repaired, and clustered mesh is returned if it is watertight.
            If not, a ValueError is raised.
        """
        if AlphaShapeHelper.check_watertight(mesh):
            return mesh
        else:
            mesh = AlphaShapeHelper.clean_mesh(mesh)
            repaired_mesh = AlphaShapeHelper.repair_mesh(mesh)
            cleaned_mesh = AlphaShapeHelper.clean_mesh(repaired_mesh)
            clustered_mesh = AlphaShapeHelper.cluster_mesh(
                cleaned_mesh, cluster_by=cluster_by
            )
            if AlphaShapeHelper.check_watertight(clustered_mesh):
                return clustered_mesh
            else:
                raise ValueError("Failed to make the mesh watertight.")

    @staticmethod
    def iterate_until_watertight(
        pointCloud: o3d.geometry.PointCloud,
        alpha: float,
        cluster_by: str = "area",
        max_iter: int = 10,
        alpha_increase_percentage: float = 0.2,
    ) -> o3d.geometry.TriangleMesh:
        """An iterative process that attempts to generate a watertight mesh from a point cloud using alpha shapes.
        The process starts with a given alpha value and generates an alpha shape mesh from the point cloud.
        If the generated mesh is not watertight, it attempts to repair it using the repair_mesh function,
        then cleans it and clusters it to retain only the largest cluster, and checks if the resulting mesh is watertight.

        If a watertight mesh is found at any point in the process, it is returned immediately.

        Args:
            pointCloud (o3d.geometry.PointCloud): The input point cloud from which to generate the alpha shape mesh.
            alpha (float): The initial alpha value for the alpha shape generation.
            cluster_by (str, optional): The criterion to use for clustering. Defaults to "area".
            max_iter (int, optional): The maximum number of iterations to perform. Defaults to 10.
            alpha_increase_percentage (float, optional): The percentage by which to increase the alpha value in each iteration. Defaults to 0.2.

        Raises:
            ValueError: If a watertight mesh cannot be found after the maximum number of iterations.

        Returns:
            o3d.geometry.TriangleMesh: The watertight mesh.
        """
        with tqdm(total=max_iter, desc="Mending alpha shape") as pbar:
            for i in range(max_iter):
                # generate new alpha shape mesh and check if it is watertight
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                    pointCloud, alpha=alpha
                )
                if AlphaShapeHelper.check_watertight(mesh):
                    print("New mesh is watertight.")
                    print("Returning new mesh with alpha = {:.2e}.".format(alpha))

                    pbar.colour = "green"
                    pbar.update(max_iter - i)

                    return mesh

                # attempt to repair the mesh. then clean it and select the largest one. then check if it is watertight
                mesh = AlphaShapeHelper.repair_mesh(
                    AlphaShapeHelper.cluster_mesh(
                        AlphaShapeHelper.clean_mesh(mesh), cluster_by=cluster_by
                    )
                )
                if AlphaShapeHelper.check_watertight(mesh):
                    print("Repaired mesh is watertight.")
                    print("Returning repaired mesh with alpha = {:.2e}.".format(alpha))

                    pbar.colour = "green"
                    pbar.update(max_iter - i)

                    return mesh

                else:
                    alpha *= 1 + alpha_increase_percentage
                    pbar.update(1)

        raise ValueError(
            "Failed to find a watertight mesh after {} iterations.".format(max_iter)
        )

    @staticmethod
    def clean_and_cluster(
        mesh: o3d.geometry.TriangleMesh, cluster_by: str = "area"
    ) -> o3d.geometry.TriangleMesh:
        """
        Post-process the mesh by cleaning it and clustering it to retain only the largest cluster based on the specified criterion.
        This function is intended to be used after the iterative process of making a mesh watertight, to further clean and refine the resulting mesh.

        It can also be used independently to clean and cluster any mesh, regardless of whether it was generated from an alpha shape or not.

        Args:
            mesh (o3d.geometry.TriangleMesh): The mesh to post-process.
            cluster_by (str, optional): The criterion to use for clustering. Defaults to "area".

        Returns:
            o3d.geometry.TriangleMesh: The post-processed mesh.
        """
        return AlphaShapeHelper.cluster_mesh(
            AlphaShapeHelper.clean_mesh(mesh), cluster_by=cluster_by
        )
