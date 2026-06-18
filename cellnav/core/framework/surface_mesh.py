## dependencies
import os
import sys
import numpy as np
import open3d as o3d
from typing import Union, Optional
from overrides import overrides

if sys.platform.startswith("win"):
    import PyFileDialogue as pyfd
else:
    pyfd = None  # placeholder for non-Windows systems, as tkinter is not supported on Unix-based systems

## custom dependencies
from cellnav.core.containers.geometry_base import GeometryBase
from cellnav.core.helpers.alpha_shape_helper import AlphaShapeHelper


__all__ = ["SurfaceMesh"]


## main class implementation - Cell membrane extraction tool
class SurfaceMesh(GeometryBase):
    def __init__(self, geometry: o3d.geometry.TriangleMesh, **kwargs) -> None:
        """
        Geometry class for representing surface meshes, built on top of open3d's TriangleMesh class.
        Provides additional functionality for geometric analysis, data manipulation, and visualization.

        Args:
            geometry (o3d.geometry.TriangleMesh): Main geometry object representing the surface mesh. Must be an instance of open3d.geometry.TriangleMesh.
            **kwargs: Additional keyword arguments for configuration (e.g., verbose mode). Currently a placeholder.
        """
        super().__init__(geometry=geometry, **kwargs)

    # %% Classmethods
    @classmethod
    @overrides
    def from_ply(cls, file_path: Optional[str] = None, **kwargs) -> "SurfaceMesh":
        if file_path is None or not os.path.isfile(file_path):
            if pyfd is None:
                raise RuntimeError(
                    "File dialog is only supported on Windows. Please provide a file path."
                )
            file_path = pyfd.call_file(
                title="Select Surface Mesh PLY File",
                filetypes=[("PLY files", "*.ply")],
            )
            if file_path is None:
                raise ValueError("No file selected. Please provide a valid file path.")

        geometry = o3d.io.read_triangle_mesh(file_path)
        return cls(geometry=geometry, **kwargs)

    @classmethod
    @overrides
    def from_o3d(cls, geometry: o3d.geometry.TriangleMesh, **kwargs) -> "SurfaceMesh":
        return cls(geometry=geometry, **kwargs)

    # %% Utility functions
    def get_vertex(self, ind: int) -> np.ndarray:
        """Return the vertex at the specified index.

        Args:
            ind (int): The index of the vertex to retrieve.

        Raises:
            IndexError: If the provided index is out of range.

        Returns:
            np.ndarray: The vertex coordinates as a numpy array.
        """
        if ind < 0 or ind >= len(self._geometry.vertices):
            raise IndexError(
                f"Index {ind} is out of bounds for vertices array of length {len(self._geometry.vertices)}."
            )
        return np.asarray(self._geometry.vertices)[ind]

    def get_vertices(self) -> np.ndarray:
        """Return all vertices of the mesh as a numpy array.

        Returns:
            np.ndarray: An array of shape (N, 3) containing the coordinates of all vertices.
        """
        return np.asarray(self._geometry.vertices)

    def get_triangle(self, ind: int) -> np.ndarray:
        """Return the triangle at the specified index.

        Args:
            ind (int): The index of the triangle to retrieve.

        Raises:
            IndexError: If the provided index is out of range.

        Returns:
            np.ndarray: The triangle vertices as a numpy array.
        """
        if ind < 0 or ind >= len(self._geometry.triangles):
            raise IndexError(
                f"Index {ind} is out of bounds for triangles array of length {len(self._geometry.triangles)}."
            )
        triangle_indices = np.asarray(self._geometry.triangles)[ind]
        triangle_vertices = self.get_vertices()[triangle_indices]
        return triangle_vertices

    def get_triangles(self) -> np.ndarray:
        """Return all triangles of the mesh as a numpy array.

        Returns:
            np.ndarray: An array of shape (M, 3) containing the vertices of all triangles.
        """
        triangles = []
        for i in range(len(self._geometry.triangles)):
            triangles.append(self.get_triangle(i))
        return np.array(triangles)

    def get_surface_area(self) -> float:
        """Compute and return the surface area of the mesh.

        Returns:
            float: The surface area of the mesh.
        """
        return self._geometry.get_surface_area()

    def get_volume(self) -> float:
        """
        Compute and return the volume enclosed by the mesh.
        Note that the mesh must be watertight for this computation to be valid.

        Raises:
            ValueError: If the mesh is not watertight.

        Returns:
            float: The volume enclosed by the mesh.
        """
        if not AlphaShapeHelper.check_watertight(self._geometry):
            raise ValueError("Mesh must be watertight to compute volume.")
        return self._geometry.get_volume()

    # %% Transformations
    def decimate_mesh(
        self, target_number_of_triangles: int, inplace: bool = True
    ) -> "SurfaceMesh":
        """Decimate (simplify) the mesh to reduce the number of triangles while preserving the overall shape.

        Args:
            target_number_of_triangles (int): The target number of triangles for the decimated mesh.
            inplace (bool, optional): Whether to modify the mesh in place. Defaults to True.

        Returns:
            SurfaceMesh: Either the modified mesh (if inplace=True) or a new SurfaceMesh instance containing the decimated mesh (if inplace=False).
        """
        decimated_mesh = self._geometry.simplify_quadric_decimation(
            target_number_of_triangles=target_number_of_triangles
        )

        if self.config["verbose"]:
            print(
                f"Original mesh had {len(self._geometry.vertices)} vertices and {len(self._geometry.triangles)} triangles."
            )
            print(
                f"Decimated (Simplified) mesh has {len(decimated_mesh.vertices)} vertices and {len(decimated_mesh.triangles)} triangles."
            )

        if inplace:
            self._geometry = decimated_mesh
            return self
        else:
            return SurfaceMesh.from_o3d(decimated_mesh)

    def subdivide_mesh(
        self, number_of_iterations: int, inplace: bool = True
    ) -> "SurfaceMesh":
        """Subdivide the mesh using midpoint subdivision to increase the number of triangles and create a smoother surface.

        Args:
            number_of_iterations (int): The number of subdivision iterations to perform.
            inplace (bool, optional): Whether to modify the mesh in place. Defaults to True.

        Returns:
            SurfaceMesh: Either the modified mesh (if inplace=True) or a new SurfaceMesh instance containing the subdivided mesh (if inplace=False).
        """

        subdivided_mesh = self._geometry.subdivide_midpoint(
            number_of_iterations=number_of_iterations
        )

        if self.config["verbose"]:
            print(
                f"Original mesh had {len(self._geometry.vertices)} vertices and {len(self._geometry.triangles)} triangles."
            )
            print(
                f"Subdivided mesh has {len(subdivided_mesh.vertices)} vertices and {len(subdivided_mesh.triangles)} triangles."
            )

        if inplace:
            self._geometry = subdivided_mesh
            return self
        else:
            return SurfaceMesh.from_o3d(subdivided_mesh)

    # %% Alpha shapes
    def check_watertight(self) -> bool:
        """Check if the mesh is watertight (i.e., has no holes and forms a closed surface).

        Returns:
            bool: True if the mesh is watertight, False otherwise.
        """
        return AlphaShapeHelper.check_watertight(self._geometry)

    def clean_mesh(self, inplace: bool = True) -> "SurfaceMesh":
        """Clean the mesh by removing duplicate vertices, degenerate triangles, and non-manifold edges.

        Args:
            inplace (bool, optional): Whether to modify the mesh in place. Defaults to True.

        Returns:
            SurfaceMesh: Either the modified mesh (if inplace=True) or a new SurfaceMesh instance containing the cleaned mesh (if inplace=False).
        """
        cleaned_mesh = AlphaShapeHelper.clean_mesh(self._geometry)
        if inplace:
            self._geometry = cleaned_mesh
            return self
        else:
            return SurfaceMesh.from_o3d(cleaned_mesh)

    def cluster_mesh(
        self, cluster_by: str = "area", inplace: bool = True
    ) -> "SurfaceMesh":
        """Cluster the mesh by merging adjacent triangles based on a specified criterion (e.g., area, number of vertices) to create a coarser representation of the surface.

        Args:
            cluster_by (str, optional): The criterion to use for clustering. Choices are "area" or "vertices". Defaults to "area".
            inplace (bool, optional): Whether to modify the mesh in place. Defaults to True.

        Returns:
            SurfaceMesh: Either the modified mesh (if inplace=True) or a new SurfaceMesh instance containing the clustered mesh (if inplace=False).
        """
        clustered_mesh = AlphaShapeHelper.cluster_mesh(
            self._geometry, cluster_by=cluster_by
        )
        if inplace:
            self._geometry = clustered_mesh
            return self
        else:
            return SurfaceMesh.from_o3d(clustered_mesh)

    def repair_mesh(self, inplace: bool = True) -> "SurfaceMesh":
        """Attempts to repair the mesh by filling holes, removing non-manifold edges, and fixing other common issues that can arise in surface meshes.

        Args:
            inplace (bool, optional): Whether to modify the mesh in place. Defaults to True.

        Returns:
            SurfaceMesh: Either the modified mesh (if inplace=True) or a new SurfaceMesh instance containing the repaired mesh (if inplace=False).
        """
        repaired_mesh = AlphaShapeHelper.repair_mesh(self._geometry)
        if inplace:
            self._geometry = repaired_mesh
            return self
        else:
            return SurfaceMesh.from_o3d(repaired_mesh)

    def make_watertight(
        self, cluster_by: str = "area", inplace: bool = True
    ) -> "SurfaceMesh":
        """
        Attempts to make the mesh watertight by filling holes and ensuring that the surface forms a closed manifold.
        This is important for accurate volume calculations and other analyses that require a closed surface.

        Args:
            cluster_by (str, optional): The criterion to use for clustering. Choices are "area" or "vertices". Defaults to "area".
            inplace (bool, optional): Whether to modify the mesh in place. Defaults to True.

        Returns:
            SurfaceMesh: Either the modified mesh (if inplace=True) or a new SurfaceMesh instance containing the watertight mesh (if inplace=False).
        """
        watertight_mesh = AlphaShapeHelper.make_watertight(
            self._geometry, cluster_by=cluster_by
        )
        if inplace:
            self._geometry = watertight_mesh
            return self
        else:
            return SurfaceMesh.from_o3d(watertight_mesh)

    def clean_and_cluster(
        self, cluster_by: str = "area", inplace: bool = True
    ) -> "SurfaceMesh":
        """Convenience function that combines mesh cleaning and clustering into a single step.
        This can be useful for quickly preparing a mesh for analysis or visualization by first removing any issues and then simplifying the surface based on the specified clustering criterion.

        Args:
            cluster_by (str, optional): The criterion to use for clustering. Choices are "area" or "vertices". Defaults to "area".
            inplace (bool, optional): Whether to modify the mesh in place. Defaults to True.

        Returns:
            SurfaceMesh: Either the modified mesh (if inplace=True) or a new SurfaceMesh instance containing the cleaned and clustered mesh (if inplace=False).
        """
        cleaned_and_clustered_mesh = self.clean_mesh(inplace=False).cluster_mesh(
            cluster_by=cluster_by, inplace=False
        )
        if inplace:
            self._geometry = cleaned_and_clustered_mesh.geometry
            return self
        else:
            return cleaned_and_clustered_mesh

    # %% IO
    @overrides
    def save(self, file_path: Optional[str] = None) -> None:
        if file_path is None or not os.path.isdir(os.path.dirname(file_path)):
            if pyfd is None:
                raise RuntimeError(
                    "File dialog is only supported on Windows. Please provide a file path."
                )
            file_path = pyfd.call_save_as_file(
                defaultextension=".ply",
                initialfile="*.ply",
                title="Select Surface Mesh PLY File",
                filetypes=[("PLY files", "*.ply")],
            )
            if file_path is None:
                raise ValueError("No file selected. Please provide a valid file path.")

        o3d.io.write_triangle_mesh(file_path, self._geometry)

    @overrides
    def load(self, file_path: Optional[str] = None) -> None:
        if file_path is None or not os.path.isfile(file_path):
            if pyfd is None:
                raise RuntimeError(
                    "File dialog is only supported on Windows. Please provide a file path."
                )
            file_path = pyfd.call_file(
                title="Select Surface Mesh PLY File",
                filetypes=[("PLY files", "*.ply")],
            )
            if file_path is None:
                raise ValueError("No file selected. Please provide a valid file path.")

        self._geometry = o3d.io.read_triangle_mesh(file_path)

    # %% Dunder methods
    @overrides
    def __repr__(self) -> str:
        return f"SurfaceMesh with {len(self._geometry.vertices)} vertices and {len(self._geometry.triangles)} triangles.\nThe mesh is {'watertight' if self.check_watertight() else 'not watertight'}."

    @overrides
    def __len__(self) -> int:
        return len(self._geometry.vertices)
    
    @overrides
    def __add__(self, other: "SurfaceMesh") -> "SurfaceMesh":
        if not isinstance(other, SurfaceMesh):
            raise TypeError(f"Unsupported operand type(s) for +: 'SurfaceMesh' and '{type(other).__name__}'")
        
        combined_geometry = self._geometry + other.geometry
        return SurfaceMesh.from_o3d(combined_geometry)
    
    @overrides
    def __sub__(self, other: "SurfaceMesh") -> "SurfaceMesh":
        if not isinstance(other, SurfaceMesh):
            raise TypeError(f"Unsupported operand type(s) for -: 'SurfaceMesh' and '{type(other).__name__}'")
        raise NotImplementedError("Subtraction of SurfaceMesh instances is not currently implemented.")
    
    @overrides
    def __iadd__(self, other: "SurfaceMesh") -> "SurfaceMesh":
        if not isinstance(other, SurfaceMesh):
            raise TypeError(f"Unsupported operand type(s) for +=: 'SurfaceMesh' and '{type(other).__name__}'")
        self._geometry += other.geometry
        return self
    
    @overrides
    def __isub__(self, other: "SurfaceMesh") -> "SurfaceMesh":
        if not isinstance(other, SurfaceMesh):
            raise TypeError(f"Unsupported operand type(s) for -=: 'SurfaceMesh' and '{type(other).__name__}'")
        raise NotImplementedError("In-place subtraction of SurfaceMesh instances is not currently implemented.")

    # %% Properties
    @property
    def vertices(self) -> o3d.utility.Vector3dVector:
        """Return the vertices of the mesh as an open3d Vector3dVector.

        Returns:
            o3d.utility.Vector3dVector: The vertices of the mesh.
        """
        return self._geometry.vertices

    @vertices.setter
    def vertices(self, vertex_array: Union[np.ndarray, o3d.utility.Vector3dVector]):
        """Set the vertices of the mesh using either a numpy array or an open3d Vector3dVector.

        Args:
            vertex_array (Union[np.ndarray, o3d.utility.Vector3dVector]): The array of vertices to set.

        """
        if isinstance(vertex_array, np.ndarray):
            vertex_array = o3d.utility.Vector3dVector(vertex_array)
        self._geometry.vertices = vertex_array

    @property
    def triangles(self) -> o3d.utility.Vector3iVector:
        """Return the triangles of the mesh as an open3d Vector3iVector, where each triangle is represented by the indices of its three vertices.

        Returns:
            o3d.utility.Vector3iVector: The triangles of the mesh.
        """
        return self._geometry.triangles

    @triangles.setter
    def triangles(self, triangle_array: Union[np.ndarray, o3d.utility.Vector3iVector]):
        """Set the triangles of the mesh using either a numpy array or an open3d Vector3iVector, where each triangle is represented by the indices of its three vertices.

        Args:
            triangle_array (Union[np.ndarray, o3d.utility.Vector3iVector]): The array of triangles to set.
        """
        if isinstance(triangle_array, np.ndarray):
            triangle_array = o3d.utility.Vector3iVector(triangle_array)
        self._geometry.triangles = triangle_array

    @property
    @overrides
    def colors(self) -> Union[np.ndarray, o3d.utility.Vector3dVector]:
        return self._geometry.vertex_colors

    @colors.setter
    @overrides
    def colors(
        self, color_array: Union[np.ndarray, o3d.utility.Vector3dVector]
    ) -> None:
        if isinstance(color_array, np.ndarray):
            color_array = o3d.utility.Vector3dVector(color_array)
        self._geometry.vertex_colors = color_array
