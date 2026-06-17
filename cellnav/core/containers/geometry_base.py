## dependencies
import numpy as np
import open3d as o3d
import matplotlib.colors
from copy import deepcopy
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union, Optional

__all__ = ["GeometryBase"]


## main class implementation - Cell membrane extraction tool
class GeometryBase(ABC):
    _geometry: Union[
        o3d.geometry.PointCloud, o3d.geometry.TriangleMesh, o3d.geometry.LineSet
    ]
    _config: Dict[str, bool]

    def __init__(
        self,
        geometry: Union[
            o3d.geometry.PointCloud, o3d.geometry.TriangleMesh, o3d.geometry.LineSet
        ],
        **kwargs,
    ) -> None:
        """
        Main base class for all geometrical objects in the Min3D framework.
        This class is not meant to be instantiated directly,
        but rather to be subclassed by specific geometry types (e.g. PointCloud, SurfaceMesh, SurfaceWireframe).
        The implementation of this class provides common functionality for all geometrical objects,
        such as basic transformations (translate, scale, center), visualization functions,
        and utility functions to get properties of the geometry (e.g. center of mass, bounding box).

        Args:
            geometry (Union[ o3d.geometry.PointCloud, o3d.geometry.TriangleMesh, o3d.geometry.LineSet ]): The underlying Open3D geometry object that this class will wrap around. This can be a PointCloud, TriangleMesh, or LineSet depending on the specific subclass.
            **kwargs: Additional keyword arguments that can be used to configure the behavior of the class. Currently, this is a placeholder for future configuration options (e.g. verbose, debug mode) that can be set when instantiating the class.
        """
        self._geometry = geometry

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
        """
        Post initialization function that can be used to perform any additional setup after the object has been initialized.
        This is a placeholder for now, but can be used in the future to set up additional properties or perform checks on the geometry.
        """
        # default gray color for all geometries
        self.paint_uniform_color((0.5, 0.5, 0.5))  

    # %% Classmethods
    @classmethod
    @abstractmethod
    def from_ply(cls, file_path: Optional[str] = None, **kwargs) -> "GeometryBase":
        """
        Load geometry from a PLY file.
        If no file path is provided and on a Windows machine, a file dialog will be opened to select a PLY file.

        Args:
            file_path (Optional[str], optional): The path to the PLY file. Defaults to None.

        Returns:
            GeometryBase: The loaded geometry object.
        """
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
        """
        Create a GeometryBase object from an existing Open3D geometry object.
        This is a common classmethod that can be used by all subclasses to create an instance from an Open3D geometry.

        Args:
            geometry (Union[ o3d.geometry.PointCloud, o3d.geometry.TriangleMesh, o3d.geometry.LineSet ]): The existing Open3D geometry object.

        Returns:
            GeometryBase: The created geometry object.
        """
        pass

    # %% Utility functions
    def get_center_of_mass(self) -> np.ndarray:
        """Calculate the center of mass of the geometry.

        Returns:
            np.ndarray: The center of mass. 3D vector.
        """
        return self._geometry.get_center()

    def get_max_bound(self) -> np.ndarray:
        """Get the maximum bounding box coordinates.

        Returns:
            np.ndarray: The maximum bounding box coordinates. 3D vector.
        """
        return self._geometry.get_max_bound()

    def get_min_bound(self) -> np.ndarray:
        """Get the minimum bounding box coordinates.

        Returns:
            np.ndarray: The minimum bounding box coordinates. 3D vector.
        """
        return self._geometry.get_min_bound()

    def get_axis_aligned_bounding_box(self) -> o3d.geometry.AxisAlignedBoundingBox:
        """Get the axis-aligned bounding box of the geometry.

        Returns:
            o3d.geometry.AxisAlignedBoundingBox: The axis-aligned bounding box.
        """
        return self._geometry.get_axis_aligned_bounding_box()

    def get_oriented_bounding_box(self) -> o3d.geometry.OrientedBoundingBox:
        """Get the oriented bounding box of the geometry.

        Returns:
            o3d.geometry.OrientedBoundingBox: The oriented bounding box.
        """
        return self._geometry.get_oriented_bounding_box()

    def get_center_of_bounding_box(self) -> np.ndarray:
        """Get the center of the axis-aligned bounding box.

        Returns:
            np.ndarray: The center of the axis-aligned bounding box. 3D vector.
        """
        return self._geometry.get_axis_aligned_bounding_box().get_center()

    # %% Data Manipulation
    def translate(
        self, translation_vector: np.ndarray, inplace: bool = True
    ) -> "GeometryBase":
        """Translate (Move) the geometry by a given translation vector.

        Args:
            translation_vector (np.ndarray): The translation vector.
            inplace (bool, optional): Whether to modify the geometry in place. Defaults to True.

        Returns:
            GeometryBase: Either the original geometry (if inplace=True) or a new translated geometry.
        """
        if inplace:
            self._geometry.translate(translation_vector)
            return self
        else:
            new_geometry = deepcopy(self._geometry).translate(translation_vector)
            return GeometryBase.from_o3d(new_geometry)

    def scale(
        self,
        scale_factor: float,
        center: Union[np.ndarray, bool] = True,
        inplace: bool = True,
    ) -> "GeometryBase":
        """Scale the geometry by a given scale factor, optionally around a specified center point.

        Args:
            scale_factor (float): The scale factor.
            center (Union[np.ndarray, bool], optional): The center point around which to scale. If True, the center of mass is used. Defaults to True.
            inplace (bool, optional): Whether to modify the geometry in place. Defaults to True.

        Returns:
            GeometryBase: Either the original geometry (if inplace=True) or a new scaled geometry.
        """
        if center is None:
            center = self.get_center_of_mass()
        if inplace:
            self._geometry.scale(scale_factor, center)
            return self
        else:
            new_geometry = deepcopy(self._geometry).scale(scale_factor, center)
            return GeometryBase.from_o3d(new_geometry)

    def center_on_origin(self, inplace: bool = True) -> "GeometryBase":
        """Center the geometry on the origin by translating it such that its center of mass is at the origin.

        Args:
            inplace (bool, optional): Whether to modify the geometry in place. Defaults to True.

        Returns:
            GeometryBase: Either the original geometry (if inplace=True) or a new centered geometry.
        """
        center_of_mass = self.get_center_of_mass()
        translation_vector = -center_of_mass

        if inplace:
            self._geometry.translate(translation_vector)
            return self
        else:
            new_geometry = deepcopy(self._geometry).translate(translation_vector)
            return GeometryBase.from_o3d(new_geometry)

    # %% 3D Visualization functions
    def visualize(self) -> None:
        """
        Visualize the geometry object saved to self._geometry using Open3D's visualization tools.
        This will open a window where the geometry can be viewed.
        No interactions are possible or planned in this basic visualization mode.
        """
        o3d.visualization.draw_geometries([self._geometry])  # type: ignore

    def visualize_only(self, *args) -> None:
        """
        Visualizes only the provided geometries, without including the geometry of the current object.
        This can be useful for visualizing related geometries (e.g. a path on top of a mesh) without showing the base geometry.
        """
        o3d.visualization.draw_geometries([*args])  # type: ignore

    def visualize_with(self, *args) -> None:
        """
        Visualizes the current geometry together with additional geometries provided as arguments.
        This can be useful for visualizing the base geometry together with related geometries (e.g. a path on top of a mesh).
        """
        args = [arg.geometry if isinstance(arg, GeometryBase) else arg for arg in args]
        o3d.visualization.draw_geometries([self._geometry, *args])  # type: ignore

    def visual_editor(
        self, inplace: bool = True, full_screen: bool = False
    ) -> "GeometryBase":
        """
        Spin up the Open3D visualizer in editing mode, which allows the user to interactively select points on the geometry.
        After the user finishes editing and closes the visualizer, the edited geometry can be returned as either a new object or by modifying the original object in place.

        Args:
            inplace (bool, optional): Whether to modify the original geometry in place or return a new geometry object. Defaults to True.
            full_screen (bool, optional): Whether to open the visualizer in full screen mode. Defaults to False.

        Returns:
            "GeometryBase": The edited geometry object, either the original or a new instance.
        """

        # the visualizer blocks the execution until the user finishes editing and closes the visualizer,
        # so we can simply return the edited geometry after vis.run() is done
        vis = o3d.visualization.VisualizerWithEditing()  # type: ignore
        vis.create_window()

        # set full screen if requested
        # we put this behind an if statement because else it will raise a warning
        if full_screen:
            vis.set_full_screen(full_screen)

        # add the geometry to the visualizer and run it
        vis.add_geometry(self._geometry)
        vis.run()

        # close the visualizer after editing is done
        vis.destroy_window()

        # chosen points are stored in vis.get_cropped_geometry()
        edited_geometry = vis.get_cropped_geometry()

        if inplace:
            self._geometry = edited_geometry
            return self
        else:
            return GeometryBase.from_o3d(edited_geometry)

    # %% Visualization helper functions
    def paint_uniform_color(
        self, color: Union[Tuple[float, float, float], List[float], np.ndarray]
    ) -> None:
        """Paint the geometry with a uniform color.

        Args:
            color (Union[Tuple[float, float, float], List[float], np.ndarray]): The RGB color to paint the geometry with, where each component is in the range [0, 1].

        Returns:
            None: This function modifies the geometry in place and does not return anything.
        """
        self._geometry.paint_uniform_color(color)

    def paint(
        self,
        color_array: np.ndarray,
        cmap: Union[str, matplotlib.colors.Colormap] = "viridis",
    ) -> None:
        """Paint the geometry with a color map.

        Args:
            color_array (np.ndarray): An array of scalar values to be mapped to colors.
            cmap (Union[str, matplotlib.colors.Colormap], optional): The colormap to use. Defaults to "viridis".

        Returns:
            None: This function modifies the geometry in place and does not return anything.
        """

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
        self._geometry.colors = o3d.utility.Vector3dVector(color_array)

    def print_visualizer_key_bindings(self) -> None:
        """
        Print the key bindings for the Open3D visualizer.
        This can be useful for users who are not familiar with the Open3D visualizer and want to know how to interact with it.
        """

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
    def save(self, file_path: Optional[str] = None) -> None:
        """Save the object stored in self._geometry to a PLY file.
        If no file path is provided and on a Windows machine, a file dialog will be opened to select a save location.

        Args:
            file_path (Optional[str], optional): The path to the file where the object will be saved. Defaults to None.
        """
        pass

    @abstractmethod
    def load(self, file_path: Optional[str] = None) -> None:
        """Load a geometry object from a PLY file and place it in self._geometry.
        If no file path is provided and on a Windows machine, a file dialog will be opened to select a PLY file.

        Args:
            file_path (Optional[str], optional): The path to the file from which the object will be loaded. Defaults to None.
        """
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
    @abstractmethod
    def colors(self) -> Union[np.ndarray, o3d.utility.Vector3dVector]:
        """
        Color property of the geometry.
        This can be either a numpy array or an Open3D Vector3dVector,
        depending on the specific geometry type and how colors are stored in that geometry.
        The shape of the color array will depend on the geometry type (e.g. for a point cloud, it will be (N, 3) where N is the number of points, and for a mesh it will be (M, 3) where M is the number of vertices).

        Returns:
            Union[np.ndarray, o3d.utility.Vector3dVector]: The color array of the geometry.
        """
        pass

    @colors.setter
    @abstractmethod
    def colors(
        self, color_array: Union[np.ndarray, o3d.utility.Vector3dVector]
    ) -> None:
        """
        Set the color property of the geometry.

        Args:
            color_array (Union[np.ndarray, o3d.utility.Vector3dVector]): The color array to set.
        """
        pass

    @property
    def config(self) -> Dict[str, bool]:
        """
        Configuration dictionary for the geometry object.
        This can be used to store various configuration options that affect the behavior of the object, such as verbosity, debug mode, or other flags that can be checked in different methods to modify their behavior.

        Returns:
            Dict[str, bool]: The configuration dictionary.
        """
        return self._config

    @config.setter
    def config(self, config_dict: Dict[str, bool]) -> None:
        """
        Set the configuration dictionary for the geometry object.

        Args:
            config_dict (Dict[str, bool]): The configuration dictionary to set.
        """
        if not isinstance(config_dict, dict):
            raise ValueError("Config must be a dictionary.")

        self._config = config_dict

    @property
    def geometry(
        self,
    ) -> Union[
        o3d.geometry.PointCloud, o3d.geometry.TriangleMesh, o3d.geometry.LineSet
    ]:
        """
        The underlying Open3D geometry object that this class wraps around.
        This can be a PointCloud, TriangleMesh, or LineSet depending on the specific subclass.
        This property allows access to the raw Open3D geometry for cases where direct manipulation or access to Open3D functions is needed.

        Returns:
            Union[o3d.geometry.PointCloud, o3d.geometry.TriangleMesh, o3d.geometry.LineSet]: The underlying Open3D geometry object.
        """
        return self._geometry

    @geometry.setter
    def geometry(
        self,
        geometry: Union[
            o3d.geometry.PointCloud, o3d.geometry.TriangleMesh, o3d.geometry.LineSet
        ],
    ) -> None:
        """
        Set the underlying Open3D geometry object.

        Args:
            geometry (Union[o3d.geometry.PointCloud, o3d.geometry.TriangleMesh, o3d.geometry.LineSet]): The Open3D geometry object to set.
        """
        if not isinstance(
            geometry,
            (o3d.geometry.PointCloud, o3d.geometry.TriangleMesh, o3d.geometry.LineSet),
        ):
            raise ValueError(
                "Geometry must be an instance of open3d.geometry.PointCloud, open3d.geometry.TriangleMesh, or open3d.geometry.LineSet."
            )

        self._geometry = geometry
