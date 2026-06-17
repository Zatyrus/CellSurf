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


__all__ = ["SurfaceWireframe"]


## main class implementation - Cell membrane extraction tool
class SurfaceWireframe(GeometryBase):
    def __init__(self, geometry: o3d.geometry.LineSet, **kwargs) -> None:
        """
        Geometry class for representing a surface wireframe,
        which is a collection of vertices and edges that define the structure of a surface without filling in the faces.
        This class provides methods for loading, saving, and manipulating surface wireframe data.

        Args:
            geometry (o3d.geometry.LineSet): The line set representing the surface wireframe.
        """
        super().__init__(geometry=geometry, **kwargs)

    # %% Classmethods
    @classmethod
    @overrides
    def from_ply(cls, file_path: Optional[str] = None, **kwargs) -> "SurfaceWireframe":
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
    def from_o3d(cls, geometry: o3d.geometry.LineSet, **kwargs) -> "SurfaceWireframe":
        return cls(geometry=geometry, **kwargs)

    # %% Utility functions
    def get_point(self, ind: int) -> np.ndarray:
        """Return a point in the surface wireframe given its index.

        Args:
            ind (int): The index of the point to retrieve.

        Raises:
            IndexError: If the index is out of bounds for the points array.

        Returns:
            np.ndarray: The point at the specified index.
        """
        if ind < 0 or ind >= len(self.geometry.points):
            raise IndexError(
                f"Index {ind} is out of bounds for points array of length {len(self.geometry.points)}."
            )
        return np.asarray(self.geometry.points)[ind]

    def get_points(self) -> np.ndarray:
        """Return all points in the surface wireframe as a numpy array.

        Returns:
            np.ndarray: An array containing all points in the surface wireframe.
        """
        return np.asarray(self.geometry.points)

    def get_line(self, ind: int) -> np.ndarray:
        """Return a line in the surface wireframe given its index.

        Args:
            ind (int): The index of the line to retrieve.

        Raises:
            IndexError: If the index is out of bounds for the lines array.

        Returns:
            np.ndarray: The line at the specified index.
        """
        if ind < 0 or ind >= len(self.geometry.lines):
            raise IndexError(
                f"Index {ind} is out of bounds for lines array of length {len(self.geometry.lines)}."
            )
        return np.asarray(self.geometry.lines)[ind]

    def get_lines(self) -> np.ndarray:
        """Return all lines in the surface wireframe as a numpy array.

        Returns:
            np.ndarray: An array containing all lines in the surface wireframe.
        """
        return np.asarray(self.geometry.lines)

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
                title="Select Surface Wireframe PLY File",
                filetypes=[("PLY files", "*.ply")],
            )
            if file_path is None:
                raise ValueError("No file selected. Please provide a valid file path.")

        o3d.io.write_line_set(file_path, self.geometry)

    @overrides
    def load(self, file_path: Optional[str] = None) -> None:
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

        self.geometry = o3d.io.read_line_set(file_path)

    # %% Dunder methods
    @overrides
    def __repr__(self) -> str:
        return f"SurfaceWireframe with {len(self.geometry.points)} vertices and {len(self.geometry.lines)} edges."

    @overrides
    def __len__(self) -> int:
        return len(self.geometry.points)

    # %% Properties
    @property
    def points(self) -> o3d.utility.Vector3dVector:
        """Return the points of the surface wireframe as an open3d Vector3dVector.

        Returns:
            o3d.utility.Vector3dVector: The points of the surface wireframe.
        """
        return self.geometry.points

    @points.setter
    def points(self, point_array: Union[np.ndarray, o3d.utility.Vector3dVector]):
        """Set the points of the surface wireframe using either a numpy array or an open3d Vector3dVector.

        Args:
            point_array (Union[np.ndarray, o3d.utility.Vector3dVector]): The array of points to set.
        """
        if isinstance(point_array, np.ndarray):
            point_array = o3d.utility.Vector3dVector(point_array)
        self.geometry.points = point_array

    @property
    def lines(self) -> o3d.utility.Vector2iVector:
        """Return the lines of the surface wireframe as a numpy array, where each line is represented by the indices of its two endpoints.

        Returns:
            np.ndarray: An array containing the indices of the endpoints for each line in the surface wireframe.
        """
        return self.geometry.lines

    @lines.setter
    def lines(
        self,
        line_array: Union[
            np.ndarray, o3d.utility.Vector2iVector, o3d.utility.Vector3iVector
        ],
    ):
        """Set the lines of the surface wireframe using either a numpy array or an open3d Vector2iVector or Vector3iVector, where each line is represented by the indices of its two endpoints.

        Args:
            line_array (Union[np.ndarray, o3d.utility.Vector2iVector, o3d.utility.Vector3iVector]): The array of lines to set.
        """
        if isinstance(line_array, np.ndarray):
            line_array = o3d.utility.Vector2iVector(line_array)
        self.geometry.lines = line_array

    @property
    @overrides
    def colors(self) -> Union[np.ndarray, o3d.utility.Vector3dVector]:
        return self.geometry.colors

    @colors.setter
    @overrides
    def colors(
        self, color_array: Union[np.ndarray, o3d.utility.Vector3dVector]
    ) -> None:
        if isinstance(color_array, np.ndarray):
            color_array = o3d.utility.Vector3dVector(color_array)
        self.geometry.colors = color_array
