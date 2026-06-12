## dependencies
import numpy as np
import open3d as o3d
from overrides import overrides
from typing import NoReturn

## custom dependencies
from Min3D.core.containers.geometry_base import GeometryBase


## main class implementation - Cell membrane extraction tool
class SurfaceWireframe(GeometryBase):
    def __init__(self, geometry: o3d.geometry.LineSet, **kwargs) -> NoReturn:
        super().__init__(geometry=geometry, **kwargs)

    # %% Classmethods
    @classmethod
    @overrides
    def from_ply(cls, file_path: str, **kwargs) -> "SurfaceWireframe":
        geometry = o3d.io.read_line_set(file_path)
        return cls(geometry=geometry, **kwargs)

    @classmethod
    @overrides
    def from_o3d(cls, geometry: o3d.geometry.LineSet, **kwargs) -> "SurfaceWireframe":
        return cls(geometry=geometry, **kwargs)

    # %% Utility functions
    def get_point(self, ind: int) -> np.ndarray:
        return np.asarray(self.geometry.points)[ind]

    def get_points(self) -> np.ndarray:
        return np.asarray(self.geometry.points)

    def get_line(self, ind: int) -> np.ndarray:
        return np.asarray(self.geometry.lines)[ind]

    def get_lines(self) -> np.ndarray:
        return np.asarray(self.geometry.lines)

    # %% IO
    @overrides
    def save(self, file_path: str) -> NoReturn:
        o3d.io.write_line_set(file_path, self.geometry)

    @overrides
    def load(self, file_path: str) -> NoReturn:
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
    def points(self) -> np.ndarray:
        return self.geometry.points

    @property
    def lines(self) -> np.ndarray:
        return self.geometry.lines
