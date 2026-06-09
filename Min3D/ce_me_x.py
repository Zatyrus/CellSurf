## dependencies
import datetime
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from typing import Callable, Dict, Any, NoReturn, Tuple, Unity

# tqdm for progress bars - automatically selects the right version for notebooks vs. terminal
from IPython import get_ipython

try:
    ipy_str = str(type(get_ipython()))
    if "zmqshell" in ipy_str:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
except ImportError:
    from tqdm import tqdm

## custom dependencies
from Min3D.core.alpha_shape_helper import AlphaShapeHelper
from Min3D.core.graph_container import GraphContainer

## main class implementation - Cell membrane extraction tool
class CeMeX:
    dataset: o3d.geometry.PointCloud
    config: Dict[str, bool]
    
    log: Dict[int, Dict[str, Any]]
    checkpoints: Dict[str, Dict[str, Any]]


    def __init__(self, dataset: o3d.geometry.PointCloud, **kwargs) -> NoReturn:
        self.dataset = dataset

        # config options for decorators - currently a placeholder
        self.config = {"verbose": True,
                       "DEBUG": False}
        
        # update from kwargs if provided
        for key in self.config.keys():
            if key in kwargs:
                self.config[key] = kwargs[key]
                
        # override if DEBUG
        if self.config["DEBUG"]:
            self.config["verbose"] = True
        
        # run post init
        self.__post_init__()
        
    def __post_init__(self) -> NoReturn:
        pass
    
    # %% Classmethods
    @classmethod
    def from_ply(cls, file_path: str, **kwargs) -> "CeMeX":
        dataset = o3d.io.read_point_cloud(file_path)
        return cls(dataset=dataset, **kwargs)
    
    @classmethod
    def from_table(cls, table: np.ndarray, **kwargs) -> "CeMeX":
        dataset = o3d.geometry.PointCloud()
        dataset.points = o3d.utility.Vector3dVector(table)
        return cls(dataset=dataset, **kwargs)
    
    @classmethod
    def from_o3d(cls, dataset: o3d.geometry.PointCloud, **kwargs) -> "CeMeX":
        return cls(dataset=dataset, **kwargs)
    
    @classmethod
    def from_MinSpt(cls, MinSpt_dataset: Any, **kwargs) -> "CeMeX":
        dataset = o3d.geometry.PointCloud()
        dataset.points = o3d.utility.Vector3dVector(MinSpt_dataset.to_table(select_axes = ['X', 'Y', 'Z']))
        return cls(dataset=dataset, **kwargs)
    
    # %% Utility functions
    def get_coordinates(self, ind: int) -> np.ndarray:
        return np.asarray(self.dataset.points)[ind]
    
    def get_table(self) -> np.ndarray:
        return np.asarray(self.dataset.points)

    def get_center_of_mass(self) -> np.ndarray:
        return self.dataset.get_center()
    
    def get_center_of_convex_hull(self) -> np.ndarray:
        hull, _ = self.dataset.compute_convex_hull()
        return hull.get_center()
    
    def get_center_of_bounding_box(self) -> np.ndarray:
        return self.dataset.get_axis_aligned_bounding_box().get_center()
    
    def get_bounding_box(self) -> o3d.geometry.AxisAlignedBoundingBox:
        return self.dataset.get_axis_aligned_bounding_box()
    
    def get_oriented_bounding_box(self) -> o3d.geometry.OrientedBoundingBox:
        return self.dataset.get_oriented_bounding_box()
            
    def get_farthest_point(self, point: np.ndarray) -> np.ndarray:
        points = np.asarray(self.dataset.points)
        distances = np.linalg.norm(points - point, axis=1)
        farthest_point = points[np.argmax(distances)]
        return farthest_point
    
    def get_farthest_point_from_center(self) -> np.ndarray:
        center_of_mass = self.get_center_of_mass()
        return self.get_farthest_point(center_of_mass)
    
    def get_average_distance_to_center(self) -> float:
        center_of_mass = self.get_center_of_mass()
        points = np.asarray(self.dataset.points)
        distances = np.linalg.norm(points - center_of_mass, axis=1)
        return np.mean(distances)
    
    def get_average_distance_to_point(self, point: np.ndarray) -> float:
        points = np.asarray(self.dataset.points)
        distances = np.linalg.norm(points - point, axis=1)
        return np.mean(distances)
    
    def sample_mesh_uniformly(self, mesh: o3d.geometry.TriangleMesh, number_of_points: int) -> "CeMeX":
        sampled_points = mesh.sample_points_uniformly(number_of_points=number_of_points)
        return CeMeX.from_o3d(sampled_points)
    
    def sample_mesh_poisson_disk(self, mesh: o3d.geometry.TriangleMesh, number_of_points: int) -> "CeMeX":
        sampled_points = mesh.sample_points_poisson_disk(number_of_points=number_of_points)
        return CeMeX.from_o3d(sampled_points)
    
    def get_mesh_surface_area(self, mesh: o3d.geometry.TriangleMesh) -> float:
        return mesh.get_surface_area()
    
    def get_mesh_volume(self, mesh: o3d.geometry.TriangleMesh) -> float:
        if not AlphaShapeHelper.check_watertight(mesh):
            raise ValueError("Mesh must be watertight to compute volume.")
        return mesh.get_volume()
    
    def get_convex_volume(self) -> float:
        hull, _ = self.dataset.compute_convex_hull()
        if not AlphaShapeHelper.check_watertight(hull):
            raise ValueError("Convex hull mesh must be watertight to compute volume.")
        return hull.get_volume()
    
    def get_concave_volume(self, alpha: float) -> float:
        concave_hull = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(self.dataset, alpha)
        if not AlphaShapeHelper.check_watertight(concave_hull):
            raise ValueError("Concave hull mesh must be watertight to compute volume.")
        return concave_hull.get_volume()
    
    def get_convex_surface_area(self) -> float:
        hull, _ = self.dataset.compute_convex_hull()
        return hull.get_surface_area()
    
    def get_concave_surface_area(self, alpha: float) -> float:
        concave_hull = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(self.dataset, alpha)
        return concave_hull.get_surface_area()
    
    def get_average_dNN(self) -> float:
        return np.mean(np.asarray(self.dataset.compute_nearest_neighbor_distance()))
    
    # %% Surface reconstruction
    def get_convex_hull(self) -> o3d.geometry.TriangleMesh:
        hull, _ = self.dataset.compute_convex_hull()
        return hull
    
    def get_concave_hull(self, alpha: float, get_raw: bool = False) -> o3d.geometry.TriangleMesh:
        concave_hull = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(self.dataset, alpha)
        
        if self.config["verbose"]:
            print(f"Concave hull created with alpha = {alpha:.2e} is watertight: {AlphaShapeHelper.check_watertight(concave_hull)}.")
        
        if get_raw:
            return concave_hull
        else:
            return AlphaShapeHelper.post_process_mesh(concave_hull)
    
    def get_bounding_sphere(self) -> Tuple[np.ndarray, float]:
        center, radius = self.dataset.get_minimal_bounding_sphere()
        return np.asarray(center), radius
    
    def get_ball_pivoting_mesh(self, radii: list) -> o3d.geometry.TriangleMesh:
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(self.dataset, o3d.utility.DoubleVector(radii))
        return mesh
    
    # %% Data Manipulation
    def center_on_origin(self, inplace:bool = True) -> Unity["CeMeX", NoReturn]:
        center_of_mass = self.get_center_of_mass()
        translation_vector = -center_of_mass
        
        if inplace:
            self.dataset.translate(translation_vector)
            return self
        else:
            new_dataset = self.dataset.translate(translation_vector)
            return CeMeX.from_o3d(new_dataset)
        
    def downsample_pcd_uniformly(self, every_k_point: float, inplace: bool = True) -> Unity["CeMeX", NoReturn]:
        downsampled_pcd = self.dataset.uniform_down_sample(input = self.dataset, every_k_points=every_k_point)
        if inplace:
            self.dataset = downsampled_pcd
        return CeMeX.from_o3d(downsampled_pcd)
    
    def statistical_outlier_removal(self, nb_neighbors: int = 20, std_ratio: float = 2.0, inplace:bool = True, display_result: bool = True) -> Unity["CeMeX", NoReturn]:
        # call the open3d function for statistical outlier removal
        cl, ind = self.dataset.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        
        # if requested, display the inliers and outliers in different colors
        if display_result:
            self.__display_inlier_outlier__(cl, ind)
            
        if inplace:
            self.dataset = cl
        else:
            return CeMeX.from_o3d(cl)
        
    def radius_outlier_removal(self, nb_points: int = 16, radius: float = 0.05, inplace:bool = True, display_result: bool = True) -> Unity["CeMeX", NoReturn]:
        # call the open3d function for radius outlier removal
        cl, ind = self.dataset.remove_radius_outlier(nb_points=nb_points, radius=radius)
        
        # if requested, display the inliers and outliers in different colors
        if display_result:
            self.__display_inlier_outlier__(cl, ind)
            
        if inplace:
            self.dataset = cl
        else:
            return CeMeX.from_o3d(cl)
    
    # %% Transformations
    def make_convex_wireframe(self) -> o3d.geometry.LineSet:
        # compute the convex hull of the point cloud
        hull = self.get_convex_hull()
        
        # make wireframe of the convex hull
        line_set = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
        
        return line_set
    
    def make_kNN_wireframe(self, k: int) -> o3d.geometry.LineSet:
        # # compute the kNN graph of the point cloud
        # pcd_tree = o3d.geometry.KDTreeFlann(self.dataset)
        # lines = []
        # for i in range(len(self.dataset.points)):
        #     [_, idx, _] = pcd_tree.search_knn_vector_3d(self.dataset.points[i], k + 1)  # k+1 because the point itself is included
        #     for j in idx[1:]:  # skip the first index because it's the point itself
        #         lines.append([i, j])
        
        # line_set = o3d.geometry.LineSet(points=self.dataset.points, lines=o3d.utility.Vector2iVector(lines))
        # return line_set
        
        # get kNN indices
        ind, _ = self.kNN_search(k)
        
        # assemble lines from the knn indices
        lines_out = []
        with tqdm(total = len(ind), desc="Creating lines from knn indices") as pbar:
            for lines in ind:
                for j in range(1, len(lines)):
                    lines_out.append([lines[0], lines[j]])
                lines_out.append([lines[-1], lines[0]])
                pbar.update(1)

        # build line set
        kNN_line_set = o3d.geometry.LineSet()
        kNN_line_set.points = o3d.utility.Vector3dVector(self.dataset.points)
        kNN_line_set.lines = o3d.utility.Vector2iVector(np.array(lines_out))
        
        return kNN_line_set
    
    def make_radiusNN_wireframe(self, radius: float) -> o3d.geometry.LineSet:
        # get radiusNN indices
        ind, _, split = self.radiusNN_search(radius)
        ind = [ind[start_i:end_i] for start_i, end_i in zip(split, split[1:])]
        
        # assemble lines from the knn indices
        lines_out = []
        with tqdm(total = len(ind), desc="Creating lines from knn indices") as pbar:
            for lines in ind:
                for j in range(1, len(lines)):
                    lines_out.append([lines[0], lines[j]])
                lines_out.append([lines[-1], lines[0]])
                pbar.update(1)

        # build line set
        kNN_line_set = o3d.geometry.LineSet()
        kNN_line_set.points = o3d.utility.Vector3dVector(self.dataset.points)
        kNN_line_set.lines = o3d.utility.Vector2iVector(np.array(lines_out))
        
        return kNN_line_set
    
    def mesh_to_wireframe(self, mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.LineSet:
        line_set = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
        return line_set
    
    def decimate_mesh(self, mesh: o3d.geometry.TriangleMesh, target_number_of_triangles: int) -> o3d.geometry.TriangleMesh:
        decimated_mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_number_of_triangles)
        
        if self.config["verbose"]:
            print(f"Original mesh had {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles.")
            print(f"Decimated (Simplified) mesh has {len(decimated_mesh.vertices)} vertices and {len(decimated_mesh.triangles)} triangles.")
        
        return decimated_mesh
    
    def subdivide_mesh(self, mesh: o3d.geometry.TriangleMesh, number_of_iterations: int) -> o3d.geometry.TriangleMesh:
        subdivided_mesh = mesh.subdivide_midpoint(number_of_iterations=number_of_iterations)
        
        if self.config["verbose"]:
            print(f"Original mesh had {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles.")
            print(f"Subdivided mesh has {len(subdivided_mesh.vertices)} vertices and {len(subdivided_mesh.triangles)} triangles.")
        
        return subdivided_mesh
    
    # %% Visualization functions
    def visualize(self,  *args) -> NoReturn:
        o3d.visualization.draw_geometries([*args])
    
    def visualize_data(self) -> NoReturn:
        o3d.visualization.draw_geometries([self.dataset])
        
    def visualize_with(self, *args) -> NoReturn:
        o3d.visualization.draw_geometries([self.dataset, *args])
        
    def visual_editor(self, inplace:bool = True, full_screen:bool = False) ->  Unity["CeMeX", NoReturn]:
        # build the visualizer object and add the point cloud to it
        vis = o3d.visualization.VisualizerWithEditing()
        # run the visualizer
        vis.create_window()
        vis.set_full_screen(full_screen)
        vis.add_geometry(self.dataset)
        vis.run()
        # close the visualizer
        vis.destroy_window()

        # chosen points are stored in vis.get_cropped_geometry()
        edited_dataset = vis.get_cropped_geometry()
        
        if inplace:
            self.dataset = edited_dataset
            return self
        else:
            return CeMeX.from_o3d(edited_dataset)
        
    def paint_uniform_color(self, color: Tuple[float, float, float]) -> NoReturn:
        self.dataset.paint_uniform_color(color)
        
    def paint_by_array(self, color_array: np.ndarray, cmap: Unity[str, plt.cm.ScalarMappable] = "viridis") -> NoReturn:
        # select colormap
        if isinstance(cmap, str):
            cmap = plt.cm.get_cmap(cmap)
        
        # apply colormap to the color array
        norm = plt.Normalize(vmin=np.min(color_array), vmax=np.max(color_array))
        color_array = cmap(norm(color_array))[:, :3]  # apply colormap and take only RGB values

        # paint the point cloud with the new colors
        self.dataset.colors = o3d.utility.Vector3dVector(color_array)
        
    def print_visualizer_key_bindings(self)-> NoReturn:
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
        
    def plot_spread_of_points_around_center(self, return_fig: bool = False) -> Unity[Tuple[plt.Figure, plt.Axes], NoReturn]:
        center_of_mass = self.get_center_of_mass()
        distance_to_center_of_mass = np.linalg.norm(np.asarray(self.dataset.points) - center_of_mass, axis=1)

        # histogram of distances to center of mass
        fig, ax = plt.subplots(1,1, figsize=(5,5), dpi = 100)
        ax.hist(distance_to_center_of_mass, bins = 100, color = 'blue', alpha = 0.7)

        # layout    
        ax.set_xlabel('d_Center (nm)')
        ax.set_ylabel('Frequency')

        fig.tight_layout()

        if return_fig:
            return fig, ax
        
    def plot_2d_histogram_of_points(self, return_fig: bool = False) -> Unity[Tuple[plt.Figure, plt.Axes], NoReturn]:
        points = np.asarray(self.dataset.points)
        center_of_mass = self.get_center_of_mass()
        axes_key = {0: 'X', 1: 'Y', 2: 'Z'}

        # 2d histogram of points in XY plane
        fig, axs = plt.subplots(1,3, figsize=(15,5), dpi = 100)
        
        for i, ind in enumerate([(0,1), (0,2), (1,2)]):
            axs[i].hist2d(points[:,ind[0]], points[:,ind[1]], bins = 50, cmap = 'inferno')
            axs[i].scatter(center_of_mass[ind[0]], center_of_mass[ind[1]], color = 'cyan', marker = 'x', s = 100, label = 'Center of Mass')
            
            # layout
            axs[i].set_title(f'{axes_key[ind[0]]} vs {axes_key[ind[1]]}')
            axs[i].set_aspect('equal', adjustable='box')
            axs[i].set_xlabel(f'{axes_key[ind[0]]} (m)')
            axs[i].set_ylabel(f'{axes_key[ind[1]]} (m)')

        # layout    
        fig.tight_layout()

        if return_fig:
            return fig, axs
        
    def plot_dNN(self):
        self.plot_dNN_for(self.dataset)
    
    def plot_dNN_for(self, *args:o3d.geometry.PointCloud, x_lim: Tuple[float, float] = None, return_fig: bool = False)->Unity[Tuple[plt.Figure, plt.Axes], NoReturn]:
        ## Step 1.3 compute nearest neighbor distances and plot boxplots
        # the dNN of the poisson disk sampling should be more consistent (narrower histogram) than the uniform sampling, which may have a wider spread of dNN values.
        # as we derive the wireframe from the poisson disk sampling, the dNN distribution matches the distribution of edge lengths in the wireframe, which is desirable for surface reconstruction.

        default_boxplot_params = {
            "widths": 0.5,
            "vert": False,
            "patch_artist": True,
            "medianprops": dict(color="black", lw=1.5),
            "meanprops": dict(zorder=3, marker="D", ms=10, color="black"),
            "whiskerprops": dict(color="black", lw=2),
            "capprops": dict(color="black", lw=2),
            "showfliers": True,
            "showbox": True,
            "showcaps": True,
            "showmeans": False,
            "zorder": 0,
        }
        
        colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow', 'lightgray']
        
        if not all(isinstance(arg, o3d.geometry.PointCloud) for arg in args):
            raise ValueError("All arguments must be of type o3d.geometry.PointCloud.")
        dNN = {i: np.asarray(arg.compute_nearest_neighbor_distance()) for i, arg in enumerate(args)}

        # build the figure and axes
        fig, axs = plt.subplots(len(args),1, figsize=(10,5), dpi = 100, sharex = True)
    
        # populate with boxplots and mean values
        for i, arg in enumerate(args):
            axs[i].boxplot(dNN[i], boxprops = dict(facecolor=colors[i], color="black", lw=2), **default_boxplot_params)
            axs[i].scatter(np.mean(dNN[i]), 1, color="black", marker="x", s=100, zorder=3, label="Mean: {:.2e} m".format(np.mean(dNN[i])))

        # layout
        for i in range(len(args)):
            axs[i].set_ylabel('Frequency')
            axs[i].legend(loc='upper right', fontsize='small')
            if x_lim is not None:
                axs[i].set_xlim(x_lim)

        # layout of the lower panel
        axs[-1].set_xlabel('dNN (m)')
        axs[-1].set_ylabel('Frequency')
        if x_lim is not None:
            axs[-1].set_xlim(x_lim)
        
        # overall layout
        fig.tight_layout()
        
        if return_fig:
            return fig, axs
        
    # %% Nearest Neighbor Search
    def get_kNN(self, point: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        [k, idx, _] = o3d.geometry.KDTreeFlann(self.dataset).search_knn_vector_3d(point, k)
        return np.asarray(self.dataset.points)[idx], idx
    
    def get_radiusNN(self, point: np.ndarray, radius: float) -> Tuple[np.ndarray, np.ndarray]:
        [k, idx, _] = o3d.geometry.KDTreeFlann(self.dataset).search_radius_vector_3d(point, radius)
        return np.asarray(self.dataset.points)[idx], idx
    
    def kNN_search(self, k:int) -> Tuple[np.ndarray, np.ndarray]:
        points = np.asarray(self.dataset.points)
        query_points = points.copy()
        
        # initialize nearest neighbor search
        nns = o3d.core.nns.NearestNeighborSearch(points)
        nns.knn_index()
        
        # run the search
        indices, distances = nns.knn_search(query_points, k+1)  # k+1 because the point itself is included
        
        # return the results as numpy arrays
        return indices.numpy(), distances.numpy()
    
    def radiusNN_search(self, radius: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        points = np.asarray(self.dataset.points)
        query_points = points.copy()
        
        # initialize nearest neighbor search
        nns = o3d.core.nns.NearestNeighborSearch(points)
        nns.fixed_radius_index(radius=radius)

        # run the search
        neighbors_index, neighbors_distance, neighbors_splits = nns.fixed_radius_search(query_points, radius=radius, sort=True)
        
        # return the results as numpy arrays
        return neighbors_index.numpy(), neighbors_distance.numpy(), neighbors_splits.numpy()
        
    # %% Alpha shapes
    def check_watertight(self, mesh: o3d.geometry.TriangleMesh) -> bool:
        return AlphaShapeHelper.check_watertight(mesh)
    
    def clean_mesh(self, mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
        return AlphaShapeHelper.clean_mesh(mesh)
    
    def cluster_mesh(self, mesh: o3d.geometry.TriangleMesh, cluster_by:str = 'area') -> o3d.geometry.TriangleMesh:
        return AlphaShapeHelper.cluster_mesh(mesh, cluster_by = cluster_by)
    
    def repair_mesh(self, mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
        return AlphaShapeHelper.repair_mesh(mesh)
    
    def make_watertight(self, mesh: o3d.geometry.TriangleMesh, cluster_by:str = 'area') -> o3d.geometry.TriangleMesh:
        return AlphaShapeHelper.make_watertight(mesh, cluster_by = cluster_by)
    
    def iterate_until_watertight(self, pointCloud: o3d.geometry.PointCloud, alpha: float, cluster_by:str = 'area', max_iter: int = 10, alpha_increase_percentage: float = 0.2) -> o3d.geometry.TriangleMesh:
        return AlphaShapeHelper.iterate_until_watertight(pointCloud, alpha, cluster_by, max_iter, alpha_increase_percentage)
    
    def post_process_mesh(self, mesh:o3d.geometry.TriangleMesh, cluster_by:str = 'area') -> o3d.geometry.TriangleMesh:
        return AlphaShapeHelper.post_process_mesh(mesh, cluster_by = cluster_by)
        
    # %% IO
    def save_mesh(self, mesh: o3d.geometry.TriangleMesh, file_path: str) -> NoReturn:
        o3d.io.write_triangle_mesh(file_path, mesh)
        
    def save_wireframe(self, line_set: o3d.geometry.LineSet, file_path: str) -> NoReturn:
        o3d.io.write_line_set(file_path, line_set)
    
    def save_point_cloud(self, file_path: str) -> NoReturn:
        o3d.io.write_point_cloud(file_path, self.dataset)
        
    # %% Helper functions
    def __display_inlier_outlier__(self, pcd: o3d.geometry.PointCloud, ind: np.ndarray) -> NoReturn:
        inlier_cloud = pcd.select_by_index(ind)
        outlier_cloud = pcd.select_by_index(ind, invert=True)

        # paint them differently
        outlier_cloud.paint_uniform_color([1, 0, 0])
        inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
        
        if self.config["verbose"]:
            print("Showing outliers (red) and inliers (gray): ")

        self.visualize_with(inlier_cloud, outlier_cloud)
        
    def __make_wireframe_with_unique_edges__(self, wireframe: o3d.geometry.LineSet) -> o3d.geometry.LineSet:
        # get unique edges from the wireframe
        wireframe_edges = np.asarray(wireframe.lines)
        wireframe_edge_set = set(tuple(sorted(edge)) for edge in wireframe_edges)
        
        if self.config["verbose"]:
            print(f"Number of edges in the wireframe: {wireframe_edges.shape[0]}")
            print(f"Number of unique edges in the wireframe: {len(wireframe_edge_set)}")
            print(f"Number of duplicate edges: {wireframe_edges.shape[0] - len(wireframe_edge_set)}")
        
        # build a new wireframe with the unique edges
        new_wireframe = o3d.geometry.LineSet()
        new_wireframe.points = wireframe.points
        new_wireframe.lines = o3d.utility.Vector2iVector(np.array(list(wireframe_edge_set)))
        
        return new_wireframe
    
    def __make_edge_length_LUT__(self, wireframe: o3d.geometry.LineSet) -> Dict[Tuple[int, int], float]:
        edge_length_LUT = {}
        for edge in np.asarray(wireframe.lines):
            point1 = wireframe.points[edge[0]]
            point2 = wireframe.points[edge[1]]
            edge_length_LUT[(edge[0], edge[1])] =  np.linalg.norm(point1 - point2)
            
        return edge_length_LUT
    
    def __suggest_shortest_path_algorithm__(self, wireframe: o3d.geometry.LineSet) -> Callable:
        ## Step 3.2.1 - Select most efficient shortest path algorithm for surface reconstruction
        E:float = np.asarray(wireframe.lines).shape[0]
        V:float = np.asarray(wireframe.points).shape[0]

        # let's compute the time complexity for a few algorithms:
        def dijkstra_time_complexity(E:float, V:float) -> float:
            return E + V * np.log(V)
        def bellman_ford_time_complexity(E:float, V:float) -> float:
            return V * E
        def floyd_warshall_time_complexity(E:float, V:float) -> float:
            return V**3
        
        complexities = {
            "Dijkstra": dijkstra_time_complexity(E, V),
            "Bellman-Ford": bellman_ford_time_complexity(E, V),
            "Floyd-Warshall": floyd_warshall_time_complexity(E, V)
        }

        if self.config["verbose"]:
            print(f"Dijkstra's time complexity: O({complexities['Dijkstra']:.2e})")
            print(f"Bellman-Ford's time complexity: O({complexities['Bellman-Ford']:.2e})")
            print(f"Floyd-Warshall's time complexity: O({complexities['Floyd-Warshall']:.2e})")

            ## suggest the most efficient algorithm based on the time complexities
            most_efficient = min(complexities, key=complexities.get)
            print(f"\nBased on the time complexities, the {most_efficient} algorithm is the most efficient for this graph.")
            
        return min(complexities, key=complexities.get)