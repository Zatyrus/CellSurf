## Dependencies
import numpy as np
import open3d as o3d
from typing import List, Union

# %% Wrapper function for open3d visualization with optional scalebar
def draw_geometries(geometries: List[Union[o3d.geometry.TriangleMesh, o3d.geometry.PointCloud, o3d.geometry.LineSet]], scalebar: Union[float, int, bool] = False) -> None:
    """
    Draw a list of geometries using Open3D's visualization capabilities. 
    Optionally, a scale bar (coordinate frame) can be added to the visualization for reference.

    Args:
        geometries (list): A list of Open3D geometry objects to be visualized.
        scalebar (Union[float, int, bool]): The size of the scale bar or coordiante frame (3D scalebar). No scalebar will be added if set to False. Defaults to False.
    """
    if isinstance(scalebar, (float, int)) and scalebar > 0:
        # gather all points from the geometries to compute the bounding box for positioning the scalebar
        points = np.concatenate([np.asarray(geometries[i].points) if isinstance(geometries[i], (o3d.geometry.LineSet, o3d.geometry.PointCloud)) else np.asarray(geometries[i].vertices) for i in range(len(geometries))])
        
        # make the scalebar
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size = scalebar)
        
        # get axis aligned bounding box of the first geometry to position the scalebar
        aabb = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(points))
        shift = aabb.get_center() - aabb.get_extent()/2 - [scalebar/2, scalebar/2, scalebar/2]  # Shift the coordinate frame away from the bounding box
        coordinate_frame.translate(shift)
        
        # add the scalebar to the geometries list for visualization
        geometries.append(coordinate_frame)
        
    o3d.visualization.draw_geometries(geometries) # type: ignore