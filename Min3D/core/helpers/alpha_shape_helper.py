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
    @staticmethod
    def check_watertight(mesh: o3d.geometry.TriangleMesh) -> bool:
        return mesh.is_watertight()

    @staticmethod
    def cluster_mesh(mesh, cluster_by: str = "area") -> o3d.geometry.TriangleMesh:
        triangle_clusters, cluster_n_triangles, cluster_tri_areas = (
            mesh.cluster_connected_triangles()
        )
        if cluster_by == "area":
            criterion = cluster_tri_areas
        elif cluster_by == "number":
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
        mesh = mesh.remove_degenerate_triangles()
        mesh = mesh.remove_duplicated_triangles()
        mesh = mesh.remove_duplicated_vertices()
        mesh = mesh.remove_non_manifold_edges()
        return mesh

    @staticmethod
    def repair_mesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
        tin = pmf.PyTMesh()  # setup the tin
        tin.load_array(np.asarray(mesh.vertices), np.asarray(mesh.triangles))

        tin.join_closest_components()  # first round of repairing
        tin.fill_small_boundaries()
        tin.remove_smallest_components()  # abrasive repairing

        tin.join_closest_components()  # second round of repairing
        tin.fill_small_boundaries()

        vclean, fclean = tin.return_arrays()

        return o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(vclean), o3d.utility.Vector3iVector(fclean)
        )

    @staticmethod
    def make_watertight(
        mesh: o3d.geometry.TriangleMesh, cluster_by: str = "area"
    ) -> o3d.geometry.TriangleMesh:
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
        # kick off an iterative process that will go over a range of alpha values and check if they are watertight.
        # if not, they will be repaired and checked again.
        # the process will stop as soon as a watertight mesh is found, or the maximum number of iterations is reached.
        # the alpha value will be increased by a percentage after each iteration.

        # iterate until a watertight mesh is found or the maximum number of iterations is reached
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
    def post_process_mesh(
        mesh: o3d.geometry.TriangleMesh, cluster_by: str = "area"
    ) -> o3d.geometry.TriangleMesh:
        return AlphaShapeHelper.cluster_mesh(
            AlphaShapeHelper.clean_mesh(mesh), cluster_by=cluster_by
        )
