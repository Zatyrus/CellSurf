## dependencies
import numpy as np
import open3d as o3d
from overrides import overrides
from typing import NoReturn


## custom dependencies
from Min3D.core.geometry_container import GeometryContainer

## main class implementation - Cell membrane extraction tool
class WireframeContainer(GeometryContainer):
    def __init__(self, geometry: o3d.geometry.LineSet, **kwargs) -> NoReturn:
        super().__init__(geometry=geometry, **kwargs)

    # %% Classmethods
    @classmethod
    @overrides
    def from_ply(cls, file_path: str, **kwargs) -> "WireframeContainer":
        geometry = o3d.io.read_line_set(file_path)
        return cls(geometry=geometry, **kwargs)

    @classmethod
    @overrides
    def from_o3d(cls, geometry: o3d.geometry.LineSet, **kwargs) -> "WireframeContainer":
        return cls(geometry=geometry, **kwargs)

    
    # %% Utility functions
    def get_vertex(self, ind: int) -> np.ndarray:
        return np.asarray(self.geometry.points)[ind]
    
    def get_vertices(self) -> np.ndarray:
        return np.asarray(self.geometry.points)
 
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
        return f"WireframeContainer with {len(self.geometry.points)} vertices and {len(self.geometry.lines)} edges."