## dependencies
import os
import sys
import numpy as np
import open3d as o3d
from typing import Optional
from overrides import overrides

if sys.platform.startswith("win"):
    import PyFileDialogue as pyfd
else:
    pyfd = None  # placeholder for non-Windows systems, as tkinter is not supported on Unix-based systems

## custom dependency imports
from Min3D.core.framework.surface_wireframe import SurfaceWireframe


__all__ = ["UniqueSurfaceWireframe"]


# implementation - this is a direct copy of the wireframe, but with unique edges only (i.e. no duplicate edges)
class UniqueSurfaceWireframe(SurfaceWireframe):
    def __init__(self, geometry: o3d.geometry.LineSet, **kwargs) -> None:
        """
        Geometry class for representing a unique surface wireframe,
        which is a collection of vertices and edges that define the structure of a surface without filling in the faces.

        Importantly, this class ensures that all edges in the wireframe are unique, meaning that duplicate edges (edges that connect the same pair of vertices) are removed.
        This is particularly useful for applications where the presence of duplicate edges can lead to inaccuracies or inefficiencies in processing, such as in certain types of mesh analysis or visualization.

        Args:
            geometry (o3d.geometry.LineSet): The line set representing the unique surface wireframe.
        """
        super().__init__(geometry=geometry, **kwargs)

    @classmethod
    @overrides
    def from_ply(
        cls, file_path: Optional[str] = None, **kwargs
    ) -> "UniqueSurfaceWireframe":
        if file_path is None or not os.path.isfile(file_path):
            if pyfd is None:
                raise RuntimeError(
                    "File dialog is only supported on Windows. Please provide a file path."
                )
            file_path = pyfd.call_file(
                title="Select Surface Wireframe PLY File",
                filetypes=[("PLY files", "*.ply")],
            )
            if file_path is None:
                raise ValueError("No file selected. Please provide a valid file path.")

        geometry = o3d.io.read_line_set(file_path)
        return cls(geometry=geometry, **kwargs)

    @classmethod
    @overrides
    def from_o3d(
        cls, geometry: o3d.geometry.LineSet, **kwargs
    ) -> "UniqueSurfaceWireframe":
        return cls(geometry=geometry, **kwargs)

    @classmethod
    def from_wireframe(
        cls, wireframe: SurfaceWireframe, **kwargs
    ) -> "UniqueSurfaceWireframe":
        """Create a UniqueSurfaceWireframe from a given SurfaceWireframe by removing duplicate edges.

        Args:
            wireframe (SurfaceWireframe): The surface wireframe from which to create a unique wireframe.

        Returns:
            UniqueSurfaceWireframe: The unique surface wireframe.
        """
        unique_edges = set()
        for edge in wireframe.get_lines():
            sorted_edge = tuple(sorted(edge))
            unique_edges.add(sorted_edge)

        unique_edges_list = list(unique_edges)
        unique_line_set = o3d.geometry.LineSet()
        unique_line_set.points = wireframe.points
        unique_line_set.lines = o3d.utility.Vector2iVector(np.array(unique_edges_list))

        return cls.from_o3d(unique_line_set, **kwargs)
