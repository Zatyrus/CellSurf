## dependencies
import numpy as np
import open3d as o3d
import matplotlib.colors
from copy import deepcopy
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Union


## main class implementation - Cell membrane extraction tool
class GeometryBase(ABC):
    geometry: Union[
        o3d.geometry.PointCloud, o3d.geometry.TriangleMesh, o3d.geometry.LineSet
    ]
    config: Dict[str, bool]

    def __init__(
        self,
        geometry: Union[
            o3d.geometry.PointCloud, o3d.geometry.TriangleMesh, o3d.geometry.LineSet
        ],
        **kwargs,
    ) -> None:
        self.geometry = geometry

        # config options for decorators - currently a placeholder
        self.config = {"verbose": True, "DEBUG": False}

        # update from kwargs if provided
        for key in self.config.keys():
            if key in kwargs:
                self.config[key] = kwargs[key]

        # override if DEBUG
        if self.config["DEBUG"]:
            self.config["verbose"] = True

        # run post init
        self.__post_init__()

    def __post_init__(self) -> None:
        pass

    # %% Classmethods
    @classmethod
    @abstractmethod
    def from_ply(cls, file_path: Union[str, None] = None, **kwargs) -> "GeometryBase":
        pass

    @classmethod
    @abstractmethod
    def from_o3d(
        cls,
        geometry: Union[
            o3d.geometry.PointCloud, o3d.geometry.TriangleMesh, o3d.geometry.LineSet
        ],
        **kwargs,
    ) -> "GeometryBase":
        pass

    # %% Utility functions
    def get_center_of_mass(self) -> np.ndarray:
        return self.geometry.get_center()

    def get_max_bound(self) -> np.ndarray:
        return self.geometry.get_max_bound()

    def get_min_bound(self) -> np.ndarray:
        return self.geometry.get_min_bound()

    def get_axis_aligned_bounding_box(self) -> o3d.geometry.AxisAlignedBoundingBox:
        return self.geometry.get_axis_aligned_bounding_box()

    def get_oriented_bounding_box(self) -> o3d.geometry.OrientedBoundingBox:
        return self.geometry.get_oriented_bounding_box()

    def get_center_of_bounding_box(self) -> np.ndarray:
        return self.geometry.get_axis_aligned_bounding_box().get_center()

    # %% Data Manipulation
    def translate(
        self, translation_vector: np.ndarray, inplace: bool = True
    ) -> Union["GeometryBase", None]:
        if inplace:
            self.geometry.translate(translation_vector)
        else:
            new_geometry = deepcopy(self.geometry).translate(translation_vector)
            return GeometryBase.from_o3d(new_geometry)

    def scale(
        self,
        scale_factor: float,
        center: Union[np.ndarray, bool] = True,
        inplace: bool = True,
    ) -> Union["GeometryBase", None]:
        if center is None:
            center = self.get_center_of_mass()
        if inplace:
            self.geometry.scale(scale_factor, center)
        else:
            new_geometry = deepcopy(self.geometry).scale(scale_factor, center)
            return GeometryBase.from_o3d(new_geometry)

    def center_on_origin(self, inplace: bool = True) -> Union["GeometryBase", None]:
        center_of_mass = self.get_center_of_mass()
        translation_vector = -center_of_mass

        if inplace:
            self.geometry.translate(translation_vector)
        else:
            new_geometry = deepcopy(self.geometry).translate(translation_vector)
            return GeometryBase.from_o3d(new_geometry)

    # %% 3D Visualization functions
    def visualize(self) -> None:
        o3d.visualization.draw_geometries([self.geometry])  # type: ignore

    def visualize_only(self, *args) -> None:
        o3d.visualization.draw_geometries([*args])  # type: ignore

    def visualize_with(self, *args) -> None:
        args = [arg.geometry if isinstance(arg, GeometryBase) else arg for arg in args]
        o3d.visualization.draw_geometries([self.geometry, *args])  # type: ignore

    def visual_editor(
        self, inplace: bool = True, full_screen: bool = False
    ) -> Union["GeometryBase", None]:
        # build the visualizer object and add the point cloud to it
        vis = o3d.visualization.VisualizerWithEditing()  # type: ignore
        # run the visualizer
        vis.create_window()
        # set full screen if requested
        # we put this behind an if statement because else it will raise a warning
        if full_screen:
            vis.set_full_screen(full_screen)
        vis.add_geometry(self.geometry)
        vis.run()
        # close the visualizer
        vis.destroy_window()

        # chosen points are stored in vis.get_cropped_geometry()
        edited_geometry = vis.get_cropped_geometry()

        if inplace:
            self.geometry = edited_geometry
            return self
        else:
            return GeometryBase.from_o3d(edited_geometry)

    # %% Visualization helper functions
    def paint_uniform_color(self, color: Tuple[float, float, float]) -> None:
        self.geometry.paint_uniform_color(color)

    def paint(
        self,
        color_array: np.ndarray,
        cmap: Union[str, matplotlib.colors.Colormap] = "viridis",
    ) -> None:
        # select colormap
        if isinstance(cmap, str):
            cmap = matplotlib.colormaps[cmap]

        # apply colormap to the color array
        norm = matplotlib.colors.Normalize(
            vmin=np.min(color_array), vmax=np.max(color_array)
        )
        color_array = cmap(norm(color_array))[
            :, :3
        ]  # apply colormap and take only RGB values

        # paint the point cloud with the new colors
        self.geometry.colors = o3d.utility.Vector3dVector(color_array)

    def print_visualizer_key_bindings(self) -> None:
        print(
            """
            -- Mouse view control --
                Left button + drag         : Rotate.
                Ctrl + left button + drag  : Translate.
                Wheel button + drag        : Translate.
                Shift + left button + drag : Roll.
                Wheel                      : Zoom in/out.

            -- Keyboard view control --
                [/]          : Increase/decrease field of view.
                R            : Reset view point.
                Ctrl/Cmd + C : Copy current view status into the clipboard.
                Ctrl/Cmd + V : Paste view status from clipboard.

            -- General control --
                Q, Esc       : Exit window.
                H            : Print help message.
                P, PrtScn    : Take a screen capture.
                D            : Take a depth capture.
                O            : Take a capture of current rendering settings.
                Alt + Enter  : Toggle between full screen and windowed mode.

            -- Render mode control --
                L            : Turn on/off lighting.
                +/-          : Increase/decrease point size.
                Ctrl + +/-   : Increase/decrease width of geometry::LineSet.
                N            : Turn on/off point cloud normal rendering.
                S            : Toggle between mesh flat shading and smooth shading.
                W            : Turn on/off mesh wireframe.
                B            : Turn on/off back face rendering.
                I            : Turn on/off image zoom in interpolation.
                T            : Toggle among image render:
                            no stretch / keep ratio / freely stretch.

            -- Color control --
                0..4,9       : Set point cloud color option.
                            0 - Default behavior, render point color.
                            1 - Render point color.
                            2 - x coordinate as color.
                            3 - y coordinate as color.
                            4 - z coordinate as color.
                            9 - normal as color.
                Ctrl + 0..4,9: Set mesh color option.
                            0 - Default behavior, render uniform gray color.
                            1 - Render point color.
                            2 - x coordinate as color.
                            3 - y coordinate as color.
                            4 - z coordinate as color.
                            9 - normal as color.
                Shift + 0..4 : Color map options.
                            0 - Gray scale color.
                            1 - JET color map.
                            2 - SUMMER color map.
                            3 - WINTER color map.
                            4 - HOT color map.
                            
            -- Editing control --
                Y            : Lock Y-Axis when picking points.
                X            : Lock X-Axis when picking points.
                Z            : Lock Z-Axis when picking points.
                F            : Go back to Free-View mode when picking points.
                K            : Lock view and start picking points. Drag to select points, then release to finish picking.
                C            : confirm selection of points when picking points. Press K again to start a new selection.
                S            : Save selected points to a .ply file when picking points.
            """
        )

    # %% IO
    @abstractmethod
    def save(self, file_path: Union[str, None] = None) -> None:
        pass

    @abstractmethod
    def load(self, file_path: Union[str, None] = None) -> None:
        pass

    # %% Dunder methods
    @abstractmethod
    def __repr__(self) -> str:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    # %% Properties
    @property
    def colors(self) -> Union[np.ndarray, o3d.utility.Vector3dVector]:
        return self.geometry.colors

    @colors.setter
    def colors(
        self, color_array: Union[np.ndarray, o3d.utility.Vector3dVector]
    ) -> None:
        if isinstance(color_array, np.ndarray):
            color_array = o3d.utility.Vector3dVector(color_array)
        self.geometry.colors = color_array
