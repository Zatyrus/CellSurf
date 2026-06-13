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

from typing import Any, List, NoReturn, Tuple, Union

## custom dependencies
from Min3D.core.containers.geometry_base import GeometryBase


## main class implementation - Cell membrane extraction tool
class PointCloud(GeometryBase):
    def __init__(self, geometry: o3d.geometry.PointCloud, **kwargs) -> None:
        super().__init__(geometry=geometry, **kwargs)

    # %% Classmethods
    @classmethod
    @overrides
    def from_ply(cls, file_path: Union[str, None] = None, **kwargs) -> "PointCloud":
        if file_path is None or not os.path.isfile(file_path):
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

        point_cloud = o3d.io.read_point_cloud(file_path)
        return cls(geometry=point_cloud, **kwargs)

    @classmethod
    @overrides
    def from_o3d(cls, geometry: o3d.geometry.PointCloud, **kwargs) -> "PointCloud":
        return cls(geometry=geometry, **kwargs)

    @classmethod
    def from_numpy(cls, table: np.ndarray, **kwargs) -> "PointCloud":
        geometry = o3d.geometry.PointCloud()
        geometry.points = o3d.utility.Vector3dVector(table)
        return cls(geometry=geometry, **kwargs)

    @classmethod
    def from_MinSpt(cls, MinSpt_dataset: Any, **kwargs) -> "PointCloud":
        geometry = o3d.geometry.PointCloud()
        geometry.points = o3d.utility.Vector3dVector(
            MinSpt_dataset.to_table(select_axes=["X", "Y", "Z"])
        )
        return cls(geometry=geometry, **kwargs)

    # %% Utility functions
    def get_point(self, ind: int) -> np.ndarray:
        return np.asarray(self.geometry.points)[ind]

    def get_points(self) -> np.ndarray:
        return np.asarray(self.geometry.points)

    def get_center_of_convex_hull(self) -> np.ndarray:
        hull, _ = self.geometry.compute_convex_hull()
        return hull.get_center()

    # %% Geometric properties
    def get_farthest_point(self, point: np.ndarray) -> np.ndarray:
        points = np.asarray(self.geometry.points)
        distances = np.linalg.norm(points - point, axis=1)
        farthest_point = points[np.argmax(distances)]
        return farthest_point

    def get_farthest_point_from_center(self) -> np.ndarray:
        center_of_mass = self.get_center_of_mass()
        return self.get_farthest_point(center_of_mass)

    def get_average_distance_to_center(self) -> float:
        center_of_mass = self.get_center_of_mass()
        points = np.asarray(self.geometry.points)
        distances = np.linalg.norm(points - center_of_mass, axis=1)
        return np.mean(distances)

    def get_average_distance_to_point(self, point: np.ndarray) -> float:
        points = np.asarray(self.geometry.points)
        distances = np.linalg.norm(points - point, axis=1)
        return np.mean(distances)

    def get_convex_volume(self) -> float:
        hull, _ = self.geometry.compute_convex_hull()
        if not hull.is_watertight():
            raise ValueError("Convex hull mesh must be watertight to compute volume.")
        return hull.get_volume()

    def get_convex_surface_area(self) -> float:
        hull, _ = self.geometry.compute_convex_hull()
        return hull.get_surface_area()

    def get_nearest_neighbor_distance(self) -> np.ndarray:
        return np.asarray(self.geometry.compute_nearest_neighbor_distance())

    def get_average_dNN(self) -> float:
        return float(
            np.mean(np.asarray(self.geometry.compute_nearest_neighbor_distance()))
        )

    # %% Bounding geometry functions
    def get_bounding_sphere(self) -> Tuple[np.ndarray, float]:
        center, radius = self.geometry.get_minimal_bounding_sphere()
        return np.asarray(center), radius

    # %% Data Manipulation
    def scale_axis(
        self,
        scale_factors: Tuple[float, float, float],
        center: Union[np.ndarray, bool] = True,
        inplace: bool = True,
    ) -> Union["PointCloud", None]:
        if isinstance(center, bool) and center:
            self.center_on_origin(inplace=True)
        if isinstance(center, np.ndarray):
            self.translate(-center, inplace=True)

        points = self.get_points()
        scaled_points = points * np.array(scale_factors)

        if inplace:
            self.geometry.points = o3d.utility.Vector3dVector(scaled_points)
        else:
            new_dataset = o3d.geometry.PointCloud()
            new_dataset.points = o3d.utility.Vector3dVector(scaled_points)
            return PointCloud.from_o3d(new_dataset)

    def downsample_pcd_uniformly(
        self, every_k_point: float, inplace: bool = True
    ) -> Union["PointCloud", NoReturn]:
        downsampled_pcd = self.geometry.uniform_down_sample(
            input=self.geometry, every_k_points=every_k_point
        )
        if inplace:
            self.geometry = downsampled_pcd
        return PointCloud.from_o3d(downsampled_pcd)

    def statistical_outlier_removal(
        self,
        nb_points: int = 20,
        std_ratio: float = 2.0,
        inplace: bool = True,
        display_result: bool = True,
    ) -> Union["PointCloud", None]:
        # call the open3d function for statistical outlier removal
        cl, ind = self.geometry.remove_statistical_outlier(
            nb_neighbors=nb_points, std_ratio=std_ratio
        )

        # if requested, display the inliers and outliers in different colors
        if display_result:
            self.__display_inlier_outlier__(cl, ind)

        if inplace:
            self.geometry = cl
        else:
            return PointCloud.from_o3d(cl)

    def radius_outlier_removal(
        self,
        nb_points: int = 16,
        radius: float = 0.05,
        inplace: bool = True,
        display_result: bool = True,
    ) -> Union["PointCloud", None]:
        # call the open3d function for radius outlier removal
        cl, ind = self.geometry.remove_radius_outlier(
            nb_points=nb_points, radius=radius
        )

        # if requested, display the inliers and outliers in different colors
        if display_result:
            self.__display_inlier_outlier__(cl, ind)

        if inplace:
            self.geometry = cl
        else:
            return PointCloud.from_o3d(cl)

    # %% Measurement plots
    def plot_spread_of_points_around_center(
        self, bins=100, dpi: int = 100, style: str = "bright", return_fig: bool = False
    ) -> Union[Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes], None]:
        center_of_mass = self.get_center_of_mass()
        distance_to_center_of_mass = np.linalg.norm(
            np.asarray(self.geometry.points) - center_of_mass, axis=1
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

    def plot_2d_histogram_of_points(
        self, bins=50, dpi=100, cmap="inferno", style="bright", return_fig: bool = False
    ) -> Union[Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes], None]:
        points = np.asarray(self.geometry.points)
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
        self.plot_dNN_for(self.geometry, **kwargs)

    def plot_dNN_for(
        self,
        *args: Union["PointCloud", o3d.geometry.PointCloud],
        x_lim: Union[Tuple[float, float], None] = None,
        dpi: int = 100,
        style: str = "bright",
        return_fig: bool = False,
    ) -> Union[Tuple[matplotlib.figure.Figure, List[matplotlib.axes.Axes]], None]:
        ## Step 1.3 compute nearest neighbor distances and plot boxplots
        # the dNN of the poisson disk sampling should be more consistent (narrower histogram) than the uniform sampling, which may have a wider spread of dNN values.
        # as we derive the wireframe from the poisson disk sampling, the dNN distribution matches the distribution of edge lengths in the wireframe, which is desirable for surface reconstruction.

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
        for i, arg in enumerate(args):
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
        [k, idx, _] = o3d.geometry.KDTreeFlann(self.geometry).search_knn_vector_3d(
            point, k
        )
        return np.asarray(self.geometry.points)[idx], idx

    def get_radiusNN(
        self, point: np.ndarray, radius: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        [k, idx, _] = o3d.geometry.KDTreeFlann(self.geometry).search_radius_vector_3d(
            point, radius
        )
        return np.asarray(self.geometry.points)[idx], idx

    def kNN_search(self, k: int) -> Tuple[np.ndarray, np.ndarray]:
        points = np.asarray(self.geometry.points)
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
        points = np.asarray(self.geometry.points)
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
    def save(self, file_path: Union[str, None] = None) -> None:
        if file_path is None or not os.path.isdir(os.path.dirname(file_path)):
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

        o3d.io.write_point_cloud(file_path, self.geometry)

    @overrides
    def load(self, file_path: Union[str, None] = None) -> None:
        if file_path is None or not os.path.isfile(file_path):
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

        self.geometry = o3d.io.read_point_cloud(file_path)

    # %% Helper functions
    def __display_inlier_outlier__(
        self, pcd: o3d.geometry.PointCloud, ind: np.ndarray
    ) -> None:
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
        return f"PointCloud with {len(self.geometry.points)} points."

    @overrides
    def __len__(self) -> int:
        return len(self.geometry.points)

    # %% Properties
    @property
    def points(self) -> np.ndarray:
        return self.geometry.points
