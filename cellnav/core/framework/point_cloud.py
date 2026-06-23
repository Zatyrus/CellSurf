## dependencies
import os
import sys
import numpy as np
import open3d as o3d
import matplotlib.axes
import matplotlib.figure

from overrides import overrides
import matplotlib.pyplot as plt
from PltStyler import PltStyler

if sys.platform.startswith("win"):
    import PyFileDialogue as pyfd
else:
    pyfd = None  # placeholder for non-Windows systems, as tkinter is not supported on Unix-based systems

from typing import Any, Dict, List, Tuple, Union, Optional

## custom dependencies
from cellnav.core.containers.geometry_base import GeometryBase


__all__ = ["PointCloud"]


## main class implementation - Cell membrane extraction tool
class PointCloud(GeometryBase):
    def __init__(self, geometry: o3d.geometry.PointCloud, **kwargs) -> None:
        """
        Geometry class for point clouds, built on top of open3d's PointCloud class.
        Provides additional functionality for geometric analysis, data manipulation, and visualization.

        Args:
            geometry (o3d.geometry.PointCloud): Main geometry object representing the point cloud. Must be an instance of open3d.geometry.PointCloud.
            **kwargs: Additional keyword arguments for configuration and customization. Currently a placeholder.
        """

        super().__init__(geometry=geometry, **kwargs)

    # %% Classmethods
    @classmethod
    @overrides
    def from_ply(cls, file_path: Optional[str] = None, **kwargs) -> "PointCloud":
        if file_path is None or not os.path.isfile(os.path.abspath(file_path)):
            if pyfd is None:
                raise RuntimeError(
                    "File dialog is only supported on Windows. Please provide a file path."
                )
            file_path = pyfd.call_file(
                title="Select Point Cloud PLY File",
                filetypes=[("PLY files", "*.ply")],
            )
            if file_path is None:
                raise ValueError("No file selected. Please provide a valid file path.")

        point_cloud = o3d.io.read_point_cloud(os.path.abspath(file_path))
        return cls(geometry=point_cloud, **kwargs)

    @classmethod
    @overrides
    def from_o3d(cls, geometry: o3d.geometry.PointCloud, **kwargs) -> "PointCloud":
        return cls(geometry=geometry, **kwargs)

    @classmethod
    @overrides
    def from_dict(
        cls, geometry_dict: Dict[str, Optional[Any]], **kwargs
    ) -> "PointCloud":
        if "points" not in geometry_dict:
            raise ValueError(
                "Input dictionary must contain a 'points' key with the point cloud data."
            )

        geometry = o3d.geometry.PointCloud()
        geometry.points = o3d.utility.Vector3dVector(np.array(geometry_dict["points"]))

        if "colors" in geometry_dict and geometry_dict["colors"] is not None:
            colors = np.array(geometry_dict["colors"])
            geometry.colors = o3d.utility.Vector3dVector(colors)

        return cls(geometry=geometry, **kwargs)

    @classmethod
    def from_numpy(
        cls, table: Union[np.ndarray, List[List[float]]], **kwargs
    ) -> "PointCloud":
        """Build a PointCloud object from a numpy array of shape (N, 3) representing N points in 3D space.

        Args:
            table (np.ndarray): Numpy array of shape (N, 3) containing the XYZ coordinates of the points.

        Returns:
            PointCloud: PointCloud object initialized with the provided points.
        """
        # catch shape mismatch
        if isinstance(table, np.ndarray):
            if table.size == 0:
                raise ValueError(
                    "Input table is empty. Please provide a non-empty array of shape (N, 3)."
                )
            if table.ndim != 2 or table.shape[1] != 3:
                raise ValueError("Input table must be a numpy array of shape (N, 3).")
        elif isinstance(table, list):
            if not all(len(row) == 3 for row in table):
                raise ValueError(
                    "Input table must be a list of lists with 3 elements each."
                )
            table = np.array(table)  # convert to numpy array for easier processing
        else:
            raise TypeError("Input table must be a numpy array or a list of lists.")

        # open3d's PointCloud class expects an open3d.utility.Vector3dVector for the points,
        # so we need to convert the numpy array to that format
        geometry = o3d.geometry.PointCloud()
        try:
            geometry.points = o3d.utility.Vector3dVector(table)
        except RuntimeError as _:
            print(
                "Warning: Failed to convert input table to open3d PointCloud. Falling back to an empty point cloud."
            )
            geometry.points = o3d.utility.Vector3dVector(
                np.empty((0, 3))
            )  # fallback to empty point cloud if input is invalid
        return cls(geometry=geometry, **kwargs)

    # %% Utility functions
    def get_point(self, ind: int) -> np.ndarray:
        """Return the coordinates of the point at the specified index.

        Args:
            ind (int): Index of the point to retrieve.

        Raises:
            IndexError: If the provided index is out of bounds.

        Returns:
            np.ndarray: Array containing the XYZ coordinates of the specified point.
        """
        if ind < 0 or ind >= len(self._geometry.points):
            raise IndexError(
                f"Index {ind} is out of bounds for points array of length {len(self._geometry.points)}."
            )
        return np.asarray(self._geometry.points)[ind]

    def get_points(self) -> np.ndarray:
        """Return the coordinates of all points in the point cloud as a numpy array.

        Returns:
            np.ndarray: Array containing the XYZ coordinates of all points.
        """
        return np.asarray(self._geometry.points)

    def get_center_of_convex_hull(self) -> np.ndarray:
        """Compute the convex hull of the point cloud and return its center.

        Returns:
            np.ndarray: 3D array containing the XYZ coordinates of the center of the convex hull.
        """
        hull, _ = self._geometry.compute_convex_hull()
        return hull.get_center()

    # %% Geometric properties
    def get_farthest_point(self, point: np.ndarray) -> np.ndarray:
        """Return the coordinates of the point in the point cloud that is farthest from the specified point.

        Args:
            point (np.ndarray): 3D array containing the XYZ coordinates of the reference point.

        Returns:
            np.ndarray: 3D array containing the XYZ coordinates of the farthest point.
        """
        points = np.asarray(self._geometry.points)
        distances = np.linalg.norm(points - point, axis=1)
        farthest_point = points[np.argmax(distances)]
        return farthest_point

    def get_farthest_point_from_center(self) -> np.ndarray:
        """Return the coordinates of the point in the point cloud that is farthest from the center of mass.

        Returns:
            np.ndarray: 3D array containing the XYZ coordinates of the farthest point from the center of mass.
        """
        center_of_mass = self.get_center_of_mass()
        return self.get_farthest_point(center_of_mass)

    def get_average_distance_to_center(self) -> float:
        """Compute the average distance of all points in the point cloud to the center of mass.

        Returns:
            float: Average distance of all points to the center of mass.
        """
        center_of_mass = self.get_center_of_mass()
        points = np.asarray(self._geometry.points)
        distances = np.linalg.norm(points - center_of_mass, axis=1)
        return np.mean(distances)

    def get_average_distance_to_point(self, point: np.ndarray) -> float:
        """Compute the average distance of all points in the point cloud to a specified point.

        Args:
            point (np.ndarray): 3D array containing the XYZ coordinates of the reference point.

        Returns:
            float: Average distance of all points to the specified point.
        """
        points = np.asarray(self._geometry.points)
        distances = np.linalg.norm(points - point, axis=1)
        return np.mean(distances)

    def get_convex_volume(self) -> float:
        """
        Compute the convex hull of the point cloud and return its volume.
        Note that the convex hull must be watertight to compute the volume.

        Raises:
            ValueError: If the convex hull is not watertight.

        Returns:
            float: The volume of the convex hull.
        """
        hull, _ = self._geometry.compute_convex_hull()
        if not hull.is_watertight():
            raise ValueError("Convex hull mesh must be watertight to compute volume.")
        return hull.get_volume()

    def get_convex_surface_area(self) -> float:
        """Compute the convex hull of the point cloud and return its surface area.

        Returns:
            float: The surface area of the convex hull.
        """
        hull, _ = self._geometry.compute_convex_hull()
        return hull.get_surface_area()

    def get_nearest_neighbor_distance(self) -> np.ndarray:
        """Compute the distance to the nearest neighbor for each point in the point cloud.

        Returns:
            np.ndarray: Array containing the distance to the nearest neighbor for each point.
        """
        return np.asarray(self._geometry.compute_nearest_neighbor_distance())

    def get_average_dNN(self) -> float:
        """Compute the average distance to the nearest neighbor for all points in the point cloud.

        Returns:
            float: The average distance to the nearest neighbor.
        """
        return float(
            np.mean(np.asarray(self._geometry.compute_nearest_neighbor_distance()))
        )

    # %% Data Manipulation
    def scale_axis(
        self,
        scale_factors: Tuple[float, float, float],
        center: Union[np.ndarray, bool] = True,
        inplace: bool = True,
    ) -> "PointCloud":
        """
        Scale the point cloud along the specified axes by the given scale factors.
        Optionally, the point cloud can be centered on the origin before scaling.

        Args:
            scale_factors (Tuple[float, float, float]): Scale factors for the X, Y, and Z axes, respectively.
            center (Union[np.ndarray, bool], optional): If True, center the point cloud on the origin before scaling. If a numpy array is provided, it will be used as the center for translation before scaling. Defaults to True.
            inplace (bool, optional): If True, modify the point cloud in place. If False, return a new PointCloud object with the scaled points. Defaults to True.

        Returns:
            "PointCloud": Either the modified point cloud (if inplace=True) or a new PointCloud object with the scaled points (if inplace=False).
        """

        if isinstance(center, bool) and center:
            self.center_on_origin(inplace=True)
        if isinstance(center, np.ndarray):
            self.translate(-center, inplace=True)

        points = self.get_points()
        scaled_points = points * np.array(scale_factors)

        if inplace:
            self._geometry.points = o3d.utility.Vector3dVector(scaled_points)
            return self
        else:
            new_dataset = o3d.geometry.PointCloud()
            new_dataset.points = o3d.utility.Vector3dVector(scaled_points)
            return PointCloud.from_o3d(new_dataset)

    def downsample_pcd_uniformly(
        self, every_k_point: float, inplace: bool = True
    ) -> "PointCloud":
        """Downsample the point cloud uniformly by keeping every k-th point.

        Args:
            every_k_point (float): The interval at which to keep points.
            inplace (bool, optional): If True, modify the point cloud in place. If False, return a new PointCloud object with the downsampled points. Defaults to True.

        Returns:
            PointCloud: Either the modified point cloud (if inplace=True) or a new PointCloud object with the downsampled points (if inplace=False).
        """
        downsampled_pcd = self._geometry.uniform_down_sample(
            input=self._geometry, every_k_points=every_k_point
        )
        if inplace:
            self._geometry = downsampled_pcd
            return self
        return PointCloud.from_o3d(downsampled_pcd)

    def statistical_outlier_removal(
        self,
        nb_points: int = 20,
        std_ratio: float = 2.0,
        inplace: bool = True,
        display_result: bool = True,
    ) -> "PointCloud":
        """Remove outliers from the point cloud using statistical analysis.

        Args:
            nb_points (int, optional): Number of neighboring points to consider. Defaults to 20.
            std_ratio (float, optional): Standard deviation ratio for outlier detection. Defaults to 2.0.
            inplace (bool, optional): If True, modify the point cloud in place. If False, return a new PointCloud object with the filtered points. Defaults to True.
            display_result (bool, optional): If True, display the inliers and outliers in different colors. Defaults to True.

        Returns:
            PointCloud: Either the modified point cloud (if inplace=True) or a new PointCloud object with the filtered points (if inplace=False).
        """
        # call the open3d function for statistical outlier removal
        cl, ind = self._geometry.remove_statistical_outlier(
            nb_neighbors=nb_points, std_ratio=std_ratio
        )

        # if requested, display the inliers and outliers in different colors
        if display_result:
            self.__display_inlier_outlier(cl, ind)

        if inplace:
            self._geometry = cl
            return self
        else:
            return PointCloud.from_o3d(cl)

    def radius_outlier_removal(
        self,
        nb_points: int = 16,
        radius: float = 0.05,
        inplace: bool = True,
        display_result: bool = True,
    ) -> "PointCloud":
        """
        Remove outliers from the point cloud based on a radius criterion.
        Points that have fewer than a specified number of neighbors within a given radius are considered outliers.

        Args:
            nb_points (int, optional): Number of neighboring points to consider. Defaults to 16.
            radius (float, optional): Radius within which to look for neighbors. Defaults to 0.05.
            inplace (bool, optional): If True, modify the point cloud in place. If False, return a new PointCloud object with the filtered points. Defaults to True.
            display_result (bool, optional): If True, display the inliers and outliers in different colors. Defaults to True.

        Returns:
            PointCloud: Either the modified point cloud (if inplace=True) or a new PointCloud object with the filtered points (if inplace=False).
        """
        # call the open3d function for radius outlier removal
        cl, ind = self._geometry.remove_radius_outlier(
            nb_points=nb_points, radius=radius
        )

        # if requested, display the inliers and outliers in different colors
        if display_result:
            self.__display_inlier_outlier(cl, ind)

        if inplace:
            self._geometry = cl
            return self
        else:
            return PointCloud.from_o3d(cl)

    # %% Measurement plots
    def plot_spread_of_points_around_center(
        self, dpi: int = 100, style: str = "bright", return_fig: bool = False
    ) -> Optional[Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]]:
        """Generate a boxplot showing the spread of points in the point cloud around the center of mass.

        Args:
            dpi (int, optional): Plot resolution. Defaults to 100.
            style (str, optional): Plot style. Can be "bright" or "dark". Defaults to "bright".
            return_fig (bool, optional): If True, return the matplotlib figure and axes. Defaults to False.

        Returns:
            Optional[Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]]: The matplotlib figure and axes if `return_fig` is True, otherwise None.
        """
        center_of_mass = self.get_center_of_mass()
        distance_to_center_of_mass = np.linalg.norm(
            np.asarray(self._geometry.points) - center_of_mass, axis=1
        )

        # apply a stylesheet for better aesthetics
        PltStyler().set_stylesheet(style).set_font(size=12).apply()
        default_boxplot_params = PltStyler().get_default_parameters("boxplot")
        default_boxplot_params.update({"showmeans": False, "vert": False})

        # histogram of distances to center of mass
        fig, ax = plt.subplots(1, 1, figsize=(8, 2), dpi=dpi)
        ax.boxplot(
            distance_to_center_of_mass,
            **default_boxplot_params
            | {"boxprops": {"facecolor": "lightcoral", "color": "black"}},
        )
        ax.scatter(
            np.mean(distance_to_center_of_mass),
            1,
            color="black",
            marker="x",
            s=100,
            zorder=3,
            label="Mean: {:.2e} m".format(np.mean(distance_to_center_of_mass)),
        )

        # layout
        ax.set_xlabel("Distance to Center [m]")
        ax.legend(loc="upper right", fontsize=10)

        fig.tight_layout()

        if return_fig:
            return fig, ax

    def plot_heatmap(
        self, bins=50, dpi=100, cmap="inferno", style="bright", return_fig: bool = False
    ) -> Optional[Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]]:
        """Generate 2D histograms of the point cloud in the XY, XZ, and YZ planes, showing the distribution of points in each plane.

        Args:
            bins (int, optional): Number of bins for the histogram. Defaults to 50.
            dpi (int, optional): Plot resolution. Defaults to 100.
            cmap (str, optional): Colormap for the histogram. Defaults to "inferno".
            style (str, optional): Plot style. Can be "bright" or "dark". Defaults to "bright".
            return_fig (bool, optional): If True, return the matplotlib figure and axes. Defaults to False.

        Returns:
            Optional[Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]]: The matplotlib figure and axes if `return_fig` is True, otherwise None.
        """
        points = np.asarray(self._geometry.points)
        center_of_mass = self.get_center_of_mass()
        axes_key = {0: "X", 1: "Y", 2: "Z"}

        # apply a stylesheet for better aesthetics
        PltStyler().set_stylesheet(style).set_font(size=12).apply()

        # 2d histogram of points in XY plane
        fig, axs = plt.subplots(1, 3, figsize=(15, 5), dpi=dpi)

        for i, ind in enumerate([(0, 1), (0, 2), (1, 2)]):
            axs[i].hist2d(points[:, ind[0]], points[:, ind[1]], bins=bins, cmap=cmap)
            axs[i].scatter(
                center_of_mass[ind[0]],
                center_of_mass[ind[1]],
                color="cyan",
                marker="x",
                s=100,
                label="Center of Mass",
            )

            # layout
            axs[i].set_title(f"{axes_key[ind[0]]} vs {axes_key[ind[1]]}")
            axs[i].set_aspect("equal", adjustable="box")
            axs[i].set_xlabel(f"{axes_key[ind[0]]} [m]")
            axs[i].set_ylabel(f"{axes_key[ind[1]]} [m]")

        # layout
        fig.tight_layout()

        if return_fig:
            return fig, axs

    def plot_dNN(self, **kwargs) -> None:
        """Generate a boxplot showing the distribution of nearest neighbor distances (dNN) for the point cloud."""
        self.plot_dNN_for(self._geometry, **kwargs)

    def plot_dNN_for(
        self,
        *args: Union["PointCloud", o3d.geometry.PointCloud],
        x_lim: Optional[Tuple[float, float]] = None,
        dpi: int = 100,
        style: str = "bright",
        return_fig: bool = False,
    ) -> Optional[Tuple[matplotlib.figure.Figure, List[matplotlib.axes.Axes]]]:
        """
        For all provided point cloud objects, compute the nearest neighbor distances (dNN) and
        generate boxplots to visualize the distribution of dNN values for each point cloud.

        Args:
            *args: Point cloud objects for which to plot dNN.
            x_lim (Optional[Tuple[float, float]], optional): Limits for the x-axis. Defaults to None.
            dpi (int, optional): Plot resolution. Defaults to 100.
            style (str, optional): Plot style. Can be "bright" or "dark". Defaults to "bright".
            return_fig (bool, optional): If True, return the matplotlib figure and axes. Defaults to False.

        Returns:
            Optional[Tuple[matplotlib.figure.Figure, List[matplotlib.axes.Axes]]]: The matplotlib figure and axes if `return_fig` is True, otherwise None.
        """
        # apply a stylesheet for better aesthetics
        PltStyler().set_stylesheet(style).set_font(size=12).apply()
        default_boxplot_params = PltStyler().get_default_parameters("boxplot")
        default_boxplot_params.update({"showmeans": False, "vert": False})

        colors = ["lightblue", "lightcoral", "lightgreen", "lightyellow", "lightgray"]

        if not all(
            isinstance(arg, (o3d.geometry.PointCloud, PointCloud)) for arg in args
        ):
            raise ValueError(
                "All arguments must be of type o3d.geometry.PointCloud or PointCloud."
            )
        dNN = {
            i: np.asarray(
                arg.get_nearest_neighbor_distance()
                if isinstance(arg, PointCloud)
                else arg.compute_nearest_neighbor_distance()
            )
            for i, arg in enumerate(args)
        }

        # build the figure and axes
        fig, axs = plt.subplots(
            len(args), 1, figsize=(8, 2 * len(args)), dpi=dpi, sharex=True
        )
        if len(args) == 1:
            axs = [axs]  # make it iterable

        # populate with boxplots and mean values
        for i in range(len(args)):
            axs[i].boxplot(
                dNN[i],
                **default_boxplot_params
                | {
                    "boxprops": {"facecolor": colors[i % len(colors)], "color": "black"}
                },
            )
            axs[i].scatter(
                np.mean(dNN[i]),
                1,
                color="black",
                marker="x",
                s=100,
                zorder=3,
                label="Mean: {:.2e} m".format(np.mean(dNN[i])),
            )

        # layout
        for i in range(len(args)):
            axs[i].legend(loc="upper right", fontsize="small")
            if x_lim is not None:
                axs[i].set_xlim(x_lim)

        # layout of the lower panel
        axs[-1].set_xlabel("dNN [m]")
        if x_lim is not None:
            axs[-1].set_xlim(x_lim)

        # overall layout
        fig.tight_layout()

        if return_fig:
            return fig, axs
        else:
            plt.show()

    # %% Nearest Neighbor Search
    def get_kNN(self, point: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return the coordinates and indices of the k nearest neighbors of a given point in the point cloud.

        Args:
            point (np.ndarray): 3D array containing the XYZ coordinates of the reference point.
            k (int): Number of nearest neighbors to search for.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the coordinates and indices of the k nearest neighbors.
        """
        [k, idx, _] = o3d.geometry.KDTreeFlann(self._geometry).search_knn_vector_3d(
            point, k
        )
        return np.asarray(self._geometry.points)[idx], idx

    def get_radiusNN(
        self, point: np.ndarray, radius: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return the coordinates and indices of all neighbors within a specified radius of a given point in the point cloud.

        Args:
            point (np.ndarray): 3D array containing the XYZ coordinates of the reference point.
            radius (float): The radius within which to search for neighbors.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the coordinates and indices of the neighbors within the specified radius.
        """
        [_, idx, _] = o3d.geometry.KDTreeFlann(self._geometry).search_radius_vector_3d(
            point, radius
        )
        return np.asarray(self._geometry.points)[idx], idx

    def kNN_search(self, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """For each point in the point cloud, return the indices and distances of its k nearest neighbors.

        Args:
            k (int): Number of nearest neighbors to search for.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the indices and distances of the k nearest neighbors.
        """
        points = np.asarray(self._geometry.points)
        query_points = points.copy()

        # initialize nearest neighbor search
        nns = o3d.core.nns.NearestNeighborSearch(points)
        nns.knn_index()

        # run the search
        indices, distances = nns.knn_search(
            query_points, k + 1
        )  # k+1 because the point itself is included

        # return the results as numpy arrays
        return indices.numpy(), distances.numpy()

    def radiusNN_search(
        self, radius: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """For each point in the point cloud, return the indices, distances, and splits of its neighbors within a specified radius.

        Args:
            radius (float): The radius within which to search for neighbors.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing the indices, distances, and splits of the neighbors within the specified radius.
        """
        points = np.asarray(self._geometry.points)
        query_points = points.copy()

        # initialize nearest neighbor search
        nns = o3d.core.nns.NearestNeighborSearch(points)
        nns.fixed_radius_index(radius=radius)

        # run the search
        neighbors_index, neighbors_distance, neighbors_splits = nns.fixed_radius_search(
            query_points, radius=radius, sort=True
        )

        # return the results as numpy arrays
        return (
            neighbors_index.numpy(),
            neighbors_distance.numpy(),
            neighbors_splits.numpy(),
        )

    # %% IO
    @overrides
    def save(self, file_path: Optional[str] = None) -> None:
        if file_path is None or not os.path.isdir(
            os.path.dirname(os.path.abspath(file_path))
        ):
            if pyfd is None:
                raise RuntimeError(
                    "File dialog is only supported on Windows. Please provide a file path."
                )
            file_path = pyfd.call_save_as_file(
                defaultextension=".ply",
                initialfile="*.ply",
                title="Select Point Cloud PLY File",
                filetypes=[("PLY files", "*.ply")],
            )
            if file_path is None:
                raise ValueError("No file selected. Please provide a valid file path.")

        o3d.io.write_point_cloud(os.path.abspath(file_path), self._geometry)

    @overrides
    def load(self, file_path: Optional[str] = None) -> None:
        if file_path is None or not os.path.isfile(os.path.abspath(file_path)):
            if pyfd is None:
                raise RuntimeError(
                    "File dialog is only supported on Windows. Please provide a file path."
                )
            file_path = pyfd.call_file(
                title="Select Point Cloud PLY File",
                filetypes=[("PLY files", "*.ply")],
            )
            if file_path is None:
                raise ValueError("No file selected. Please provide a valid file path.")

        self._geometry = o3d.io.read_point_cloud(os.path.abspath(file_path))

    @overrides
    def to_dict(self) -> Dict[str, Optional[Any]]:
        return {
            "points": np.asarray(self._geometry.points).tolist(),
            "colors": np.asarray(self._geometry.colors).tolist()
            if self._geometry.colors
            else None,
        }

    # %% Helper functions
    def __display_inlier_outlier(
        self, pcd: o3d.geometry.PointCloud, ind: np.ndarray
    ) -> None:
        """Display inlier and outlier points in the point cloud.

        Args:
            pcd (o3d.geometry.PointCloud): The point cloud to display.
            ind (np.ndarray): The indices of the inlier points.
        """
        inlier_cloud = pcd.select_by_index(ind)
        outlier_cloud = pcd.select_by_index(ind, invert=True)

        # paint them differently
        outlier_cloud.paint_uniform_color([1, 0, 0])
        inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])

        if self.config["verbose"]:
            print("Showing outliers (red) and inliers (gray): ")

        self.visualize_only(inlier_cloud, outlier_cloud)

    # %% Dunder methods
    @overrides
    def __repr__(self) -> str:
        return f"PointCloud with {len(self._geometry.points)} points."

    @overrides
    def __len__(self) -> int:
        return len(self._geometry.points)

    @overrides
    def __add__(self, other: "GeometryBase") -> "PointCloud":
        if not isinstance(other, PointCloud):
            raise ValueError("Can only add another PointCloud object.")
        return PointCloud.from_o3d(self._geometry + other._geometry)

    @overrides
    def __sub__(self, other: "GeometryBase") -> "PointCloud":
        if not isinstance(other, PointCloud):
            raise ValueError("Can only subtract another PointCloud object.")

        # compute the set difference of the points in self and other
        new_points = np.array(
            [point for point in self.points if point not in other.points]
        )
        if new_points.size == 0:
            self._geometry.points = o3d.utility.Vector3dVector(np.empty((0, 3)))
            self._geometry.colors = o3d.utility.Vector3dVector(np.empty((0, 3)))
            return self

        new_pcd = o3d.geometry.PointCloud()
        new_pcd.points = o3d.utility.Vector3dVector(new_points)

        # if colors are present, we need to filter them as well
        new_colors = (
            np.array(
                [
                    color
                    for point, color in zip(self.points, self.colors)
                    if point not in other.points
                ]
            )
            if self._geometry.colors
            else np.array([])
        )
        new_pcd.colors = (
            o3d.utility.Vector3dVector(new_colors)
            if new_colors.size > 0
            else o3d.utility.Vector3dVector(np.empty((0, 3)))
        )

        return PointCloud.from_o3d(new_pcd)

    @overrides
    def __iadd__(self, other: "GeometryBase") -> "PointCloud":
        if not isinstance(other, PointCloud):
            raise ValueError("Can only add another PointCloud object.")
        new_geometry = self._geometry + other._geometry
        self._geometry = new_geometry
        return self

    @overrides
    def __isub__(self, other: "GeometryBase") -> "PointCloud":
        if not isinstance(other, PointCloud):
            raise ValueError("Can only subtract another PointCloud object.")

        # compute the set difference of the points in self and other
        new_points = np.array(
            [point for point in self.points if point not in other.points]
        )
        if new_points.size == 0:
            self._geometry.points = o3d.utility.Vector3dVector(np.empty((0, 3)))
            self._geometry.colors = o3d.utility.Vector3dVector(np.empty((0, 3)))
            return self

        new_pcd = o3d.geometry.PointCloud()
        new_pcd.points = o3d.utility.Vector3dVector(new_points)

        # if colors are present, we need to filter them as well
        new_colors = (
            np.array(
                [
                    color
                    for point, color in zip(self.points, self.colors)
                    if point not in other.points
                ]
            )
            if self._geometry.colors
            else np.array([])
        )
        new_pcd.colors = (
            o3d.utility.Vector3dVector(new_colors)
            if new_colors.size > 0
            else o3d.utility.Vector3dVector(np.empty((0, 3)))
        )

        self._geometry = new_pcd
        return self

    # %% Properties
    @property
    def points(self) -> o3d.utility.Vector3dVector:
        """Get the points of the point cloud.

        Returns:
            o3d.utility.Vector3dVector: The points of the point cloud.
        """
        return self._geometry.points

    @points.setter
    def points(
        self, point_array: Union[np.ndarray, o3d.utility.Vector3dVector]
    ) -> None:
        """Set the points of the point cloud.

        Args:
            point_array (Union[np.ndarray, o3d.utility.Vector3dVector]): The new points for the point cloud, either as a numpy array of shape (N, 3) or as an open3d Vector3dVector.
        """
        if isinstance(point_array, np.ndarray):
            point_array = o3d.utility.Vector3dVector(point_array)
        self._geometry.points = point_array

    @property
    @overrides
    def colors(self) -> Union[np.ndarray, o3d.utility.Vector3dVector]:
        return self._geometry.colors

    @colors.setter
    @overrides
    def colors(
        self, color_array: Union[np.ndarray, o3d.utility.Vector3dVector]
    ) -> None:
        if isinstance(color_array, np.ndarray):
            color_array = o3d.utility.Vector3dVector(color_array)
        self._geometry.colors = color_array
