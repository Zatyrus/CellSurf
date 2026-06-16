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
from Min3D.core.containers.geometry_base import GeometryBase
from Min3D.core.helpers.alpha_shape_helper import AlphaShapeHelper


__all__ = ["SurfaceMesh"]


## main class implementation - Cell membrane extraction tool
class SurfaceMesh(GeometryBase):
    def __init__(self, geometry: o3d.geometry.TriangleMesh, **kwargs) -> None:
        super().__init__(geometry=geometry, **kwargs)

    def __post_init__(self) -> None:
        pass

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
        return np.asarray(self.geometry.vertices)[ind]

    def get_vertices(self) -> np.ndarray:
        return np.asarray(self.geometry.vertices)

    def get_triangle(self, ind: int) -> np.ndarray:
        return np.asarray(self.geometry.triangles)[ind]

    def get_triangles(self) -> np.ndarray:
        return np.asarray(self.geometry.triangles)

    def get_surface_area(self) -> float:
        return self.geometry.get_surface_area()

    def get_volume(self) -> float:
        if not AlphaShapeHelper.check_watertight(self.geometry):
            raise ValueError("Mesh must be watertight to compute volume.")
        return self.geometry.get_volume()

    # %% Transformations
    def decimate_mesh(
        self, target_number_of_triangles: int, inplace: bool = True
    ) -> Optional["SurfaceMesh"]:
        decimated_mesh = self.geometry.simplify_quadric_decimation(
            target_number_of_triangles=target_number_of_triangles
        )

        if self.config["verbose"]:
            print(
                f"Original mesh had {len(self.geometry.vertices)} vertices and {len(self.geometry.triangles)} triangles."
            )
            print(
                f"Decimated (Simplified) mesh has {len(decimated_mesh.vertices)} vertices and {len(decimated_mesh.triangles)} triangles."
            )

        if inplace:
            self.geometry = decimated_mesh
            return self
        else:
            return SurfaceMesh.from_o3d(decimated_mesh)

    def subdivide_mesh(
        self, number_of_iterations: int, inplace: bool = True
    ) -> Optional["SurfaceMesh"]:
        subdivided_mesh = self.geometry.subdivide_midpoint(
            number_of_iterations=number_of_iterations
        )

        if self.config["verbose"]:
            print(
                f"Original mesh had {len(self.geometry.vertices)} vertices and {len(self.geometry.triangles)} triangles."
            )
            print(
                f"Subdivided mesh has {len(subdivided_mesh.vertices)} vertices and {len(subdivided_mesh.triangles)} triangles."
            )

        if inplace:
            self.geometry = subdivided_mesh
            return self
        else:
            return SurfaceMesh.from_o3d(subdivided_mesh)

    # %% Alpha shapes
    def check_watertight(self) -> bool:
        return AlphaShapeHelper.check_watertight(self.geometry)

    def clean_mesh(self, inplace: bool = True) -> "SurfaceMesh":
        cleaned_mesh = AlphaShapeHelper.clean_mesh(self.geometry)
        if inplace:
            self.geometry = cleaned_mesh
            return self
        else:
            return SurfaceMesh.from_o3d(cleaned_mesh)

    def cluster_mesh(
        self, cluster_by: str = "area", inplace: bool = True
    ) -> "SurfaceMesh":
        clustered_mesh = AlphaShapeHelper.cluster_mesh(
            self.geometry, cluster_by=cluster_by
        )
        if inplace:
            self.geometry = clustered_mesh
            return self
        else:
            return SurfaceMesh.from_o3d(clustered_mesh)

    def repair_mesh(self, inplace: bool = True) -> "SurfaceMesh":
        repaired_mesh = AlphaShapeHelper.repair_mesh(self.geometry)
        if inplace:
            self.geometry = repaired_mesh
            return self
        else:
            return SurfaceMesh.from_o3d(repaired_mesh)

    def make_watertight(
        self, cluster_by: str = "area", inplace: bool = True
    ) -> "SurfaceMesh":
        watertight_mesh = AlphaShapeHelper.make_watertight(
            self.geometry, cluster_by=cluster_by
        )
        if inplace:
            self.geometry = watertight_mesh
            return self
        else:
            return SurfaceMesh.from_o3d(watertight_mesh)

    def clean_and_cluster(
        self, cluster_by: str = "area", inplace: bool = True
    ) -> "SurfaceMesh":
        cleaned_and_clustered_mesh = self.clean_mesh(inplace=False).cluster_mesh(
            cluster_by=cluster_by, inplace=False
        )
        if inplace:
            self.geometry = cleaned_and_clustered_mesh.geometry
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

        o3d.io.write_triangle_mesh(file_path, self.geometry)

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

        self.geometry = o3d.io.read_triangle_mesh(file_path)

    # %% Dunder methods
    @overrides
    def __repr__(self) -> str:
        return f"SurfaceMesh with {len(self.geometry.vertices)} vertices and {len(self.geometry.triangles)} triangles.\nThe mesh is {'watertight' if self.check_watertight() else 'not watertight'}."

    @overrides
    def __len__(self) -> int:
        return len(self.geometry.vertices)

    # %% Properties
    @property
    def vertices(self) -> np.ndarray:
        return self.geometry.vertices

    @property
    def triangles(self) -> np.ndarray:
        return self.geometry.triangles

    @property
    @overrides
    def colors(self) -> Union[np.ndarray, o3d.utility.Vector3dVector]:
        return self.geometry.vertex_colors

    @colors.setter
    @overrides
    def colors(
        self, color_array: Union[np.ndarray, o3d.utility.Vector3dVector]
    ) -> None:
        if isinstance(color_array, np.ndarray):
            color_array = o3d.utility.Vector3dVector(color_array)
        self.geometry.vertex_colors = color_array
