## Dependencies
import open3d as o3d
from typing import Union

__all__ = ["visual_geometry_editor"]


def visual_geometry_editor(
    geometry: Union[
        o3d.geometry.PointCloud, o3d.geometry.TriangleMesh, o3d.geometry.LineSet
    ],
    full_screen: bool = False,
) -> Union[o3d.geometry.PointCloud, o3d.geometry.TriangleMesh, o3d.geometry.LineSet]:
    """
    Spin up the Open3D visualizer in editing mode, which allows the user to interactively select points on the geometry.
    After the user finishes editing and closes the visualizer, the edited geometry can be returned as either a new object or by modifying the original object in place.

    In the future, more callbacks will be added.

    Args:
        geometry (Union[o3d.geometry.PointCloud, o3d.geometry.TriangleMesh, o3d.geometry.LineSet]): The geometry object to be edited.
        full_screen (bool, optional): Whether to open the visualizer in full screen mode. Defaults to False.

    Returns:
        Union[o3d.geometry.PointCloud, o3d.geometry.TriangleMesh, o3d.geometry.LineSet]: The edited geometry object, either the original or a new instance.
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
    vis.add_geometry(geometry)
    vis.run()

    # close the visualizer after editing is done
    vis.destroy_window()

    # chosen points are stored in vis.get_cropped_geometry()
    return vis.get_cropped_geometry()
