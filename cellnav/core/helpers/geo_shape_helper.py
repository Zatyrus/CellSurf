import numpy as np
import open3d as o3d
from typing import List, Union


class GeoShapeHelper:
    """A class to help with the generation of geometrical shapes in open3d."""

    def __init__(self) -> None:
        pass

    @staticmethod
    def generate_lineset_from_edges(
        points: Union[List[np.ndarray], np.ndarray, o3d.utility.Vector3dVector],
        edges: List[tuple],
        color: Union[List[int], np.ndarray] = [0, 0, 0],
    ) -> o3d.geometry.LineSet:
        """Generate a LineSet from a list of points and edges, with an optional color.

        Args:
            points (Union[List[np.ndarray], np.ndarray, o3d.utility.Vector3dVector]): The points to use for the LineSet.
            edges (List[tuple]): The edges to use for the LineSet.
            color (Union[List[int], np.ndarray], optional): The color of the lines. Defaults to [0, 0, 0].

        Returns:
            o3d.geometry.LineSet: The generated LineSet.
        """

        line_set = o3d.geometry.LineSet()
        line_set.points = (
            o3d.utility.Vector3dVector(points)
            if not isinstance(points, o3d.utility.Vector3dVector)
            else points
        )
        line_set.lines = o3d.utility.Vector2iVector(np.array(edges))
        line_set.paint_uniform_color(color)
        return line_set

    @staticmethod
    def generate_outline(
        points: Union[List[np.ndarray], np.ndarray, o3d.utility.Vector3dVector],
        color: Union[List[int], np.ndarray] = [1, 1, 0],
    ) -> o3d.geometry.LineSet:
        """Generate a LineSet representing the outline of a shape defined by a list of points, with an optional color.

        Args:
            points (Union[List[np.ndarray], np.ndarray, o3d.utility.Vector3dVector]): The points to use for the outline.
            color (Union[List[int], np.ndarray], optional): The color of the outline. Defaults to [1, 1, 0].

        Returns:
            o3d.geometry.LineSet: The generated LineSet representing the outline.
        """
        outline = o3d.geometry.LineSet()
        outline.points = (
            o3d.utility.Vector3dVector(points)
            if not isinstance(points, o3d.utility.Vector3dVector)
            else points
        )
        outline.lines = o3d.utility.Vector2iVector(
            np.array([[i, i + 1] for i in range(len(points) - 1)])
        )
        outline.paint_uniform_color(color)
        return outline

    @staticmethod
    def generate_sphere_on_point(
        point: Union[List[float], np.ndarray],
        radius: float = 10e-9,
        color: Union[List[int], np.ndarray] = [1, 0, 0],
        resolution: int = 10,
        make_line: bool = False,
    ) -> Union[o3d.geometry.TriangleMesh, o3d.geometry.LineSet]:
        """
        Generate a sphere on a given point with a specified radius, color, resolution, and an option to make it a line set.
        Making it a line set will create a wireframe representation of the sphere.

        This can be useful for visualizing the structure of the sphere without rendering it as a solid object.
        Will be easier on the GPU when rendering many spheres, as it will reduce the number of triangles to render.

        Args:
            point (Union[List[float], np.ndarray]): The point on which to generate the sphere.
            radius (float, optional): The radius of the sphere. Defaults to 10e-9.
            color (Union[List[int], np.ndarray], optional): The color of the sphere. Defaults to [1, 0, 0].
            resolution (int, optional): The resolution of the sphere. Defaults to 10.
            make_line (bool, optional): Whether to generate a line set instead of a triangle mesh. Defaults to False.

        Returns:
            o3d.geometry.TriangleMesh: The generated sphere.
        """
        sphere = o3d.geometry.TriangleMesh.create_sphere(
            radius=radius, resolution=resolution
        )
        if make_line:
            sphere = o3d.geometry.LineSet.create_from_triangle_mesh(sphere)
        sphere.translate(point, relative=False)
        sphere.paint_uniform_color(color)
        return sphere

    @staticmethod
    def generate_octahedron_on_point(
        point: Union[List[float], np.ndarray],
        radius: float = 10e-9,
        color: Union[List[int], np.ndarray] = [1, 0, 0],
        make_line: bool = False,
    ) -> Union[o3d.geometry.TriangleMesh, o3d.geometry.LineSet]:
        """
        Generate an octahedron on a given point with a specified radius, color, and an option to make it a line set.
        Making it a line set will create a wireframe representation of the octahedron.

        This can be useful for visualizing the structure of the octahedron without rendering it as a solid object.
        Will be easier on the GPU when rendering many octahedrons, as it will reduce the number of triangles to render.

        Args:
            point (Union[List[float], np.ndarray]): The point on which to generate the octahedron.
            radius (float, optional): The radius of the octahedron. Defaults to 10e-9.
            color (Union[List[int], np.ndarray], optional): The color of the octahedron. Defaults to [1, 0, 0].
            make_line (bool, optional): Whether to generate a line set instead of a triangle mesh. Defaults to False.

        Returns:
            o3d.geometry.TriangleMesh: The generated octahedron.
        """
        octahedron = o3d.geometry.TriangleMesh.create_octahedron(radius=radius)
        if make_line:
            octahedron = o3d.geometry.LineSet.create_from_triangle_mesh(octahedron)
        octahedron.translate(point, relative=False)
        octahedron.paint_uniform_color(color)
        return octahedron

    @staticmethod
    def generate_cube_on_point(
        point: Union[List[float], np.ndarray],
        size: float = 10e-9,
        color: Union[List[int], np.ndarray] = [1, 0, 0],
        make_line: bool = False,
    ) -> Union[o3d.geometry.TriangleMesh, o3d.geometry.LineSet]:
        """
        Generate a cube on a given point with a specified size, color, and an option to make it a line set.
        Making it a line set will create a wireframe representation of the cube.

        This can be useful for visualizing the structure of the cube without rendering it as a solid object.
        Will be easier on the GPU when rendering many cubes, as it will reduce the number of triangles to render.

        Args:
            point (Union[List[float], np.ndarray]): The point on which to generate the cube.
            size (float, optional): The size of the cube. Defaults to 10e-9.
            color (Union[List[int], np.ndarray], optional): The color of the cube. Defaults to [1, 0, 0].
            make_line (bool, optional): Whether to generate a line set instead of a triangle mesh. Defaults to False.

        Returns:
            Union[o3d.geometry.TriangleMesh, o3d.geometry.LineSet]: The generated cube.
        """
        cube = o3d.geometry.TriangleMesh.create_box(width=size, height=size, depth=size)
        if make_line:
            cube = o3d.geometry.LineSet.create_from_triangle_mesh(cube)
        cube.translate(point, relative=False)
        cube.paint_uniform_color(color)
        return cube

    @staticmethod
    def generate_cylinder_on_point(
        point: Union[List[float], np.ndarray],
        radius: float = 10e-9,
        height: float = 10e-9,
        color: Union[List[int], np.ndarray] = [1, 0, 0],
        make_line: bool = False,
    ) -> Union[o3d.geometry.TriangleMesh, o3d.geometry.LineSet]:
        """
        Generate a cylinder on a given point with a specified radius, height, color, and an option to make it a line set.
        Making it a line set will create a wireframe representation of the cylinder.

        This can be useful for visualizing the structure of the cylinder without rendering it as a solid object.
        Will be easier on the GPU when rendering many cylinders, as it will reduce the number of triangles to render.

        Args:
            point (Union[List[float], np.ndarray]): The point on which to generate the cylinder.
            radius (float, optional): The radius of the cylinder. Defaults to 10e-9.
            height (float, optional): The height of the cylinder. Defaults to 10e-9.
            color (Union[List[int], np.ndarray], optional): The color of the cylinder. Defaults to [1, 0, 0].
            make_line (bool, optional): Whether to generate a line set instead of a triangle mesh. Defaults to False.

        Returns:
            Union[o3d.geometry.TriangleMesh, o3d.geometry.LineSet]: The generated cylinder.
        """
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(
            radius=radius, height=height
        )
        if make_line:
            cylinder = o3d.geometry.LineSet.create_from_triangle_mesh(cylinder)
        cylinder.translate(point, relative=False)
        cylinder.paint_uniform_color(color)
        return cylinder

    @staticmethod
    def generate_icosahedron_on_point(
        point: Union[List[float], np.ndarray],
        radius: float = 10e-9,
        color: Union[List[int], np.ndarray] = [1, 0, 0],
        make_line: bool = False,
    ) -> Union[o3d.geometry.TriangleMesh, o3d.geometry.LineSet]:
        """
        Generate an icosahedron on a given point with a specified radius, color, and an option to make it a line set.
        Making it a line set will create a wireframe representation of the icosahedron.

        This can be useful for visualizing the structure of the icosahedron without rendering it as a solid object.
        Will be easier on the GPU when rendering many icosahedrons, as it will reduce the number of triangles to render.

        Args:
            point (Union[List[float], np.ndarray]): The point on which to generate the icosahedron.
            radius (float, optional): The radius of the icosahedron. Defaults to 10e-9.
            color (Union[List[int], np.ndarray], optional): The color of the icosahedron. Defaults to [1, 0, 0].
            make_line (bool, optional): Whether to generate a line set instead of a triangle mesh. Defaults to False.

        Returns:
            Union[o3d.geometry.TriangleMesh, o3d.geometry.LineSet]: The generated icosahedron.
        """
        icosahedron = o3d.geometry.TriangleMesh.create_icosahedron(radius=radius)
        if make_line:
            icosahedron = o3d.geometry.LineSet.create_from_triangle_mesh(icosahedron)
        icosahedron.translate(point, relative=False)
        icosahedron.paint_uniform_color(color)
        return icosahedron

    @staticmethod
    def generate_tetrahedron_on_point(
        point: Union[List[float], np.ndarray],
        radius: float = 10e-9,
        color: Union[List[int], np.ndarray] = [1, 0, 0],
        make_line: bool = False,
    ) -> Union[o3d.geometry.TriangleMesh, o3d.geometry.LineSet]:
        """
        Generate a tetrahedron on a given point with a specified radius, color, and an option to make it a line set.
        Making it a line set will create a wireframe representation of the tetrahedron.

        This can be useful for visualizing the structure of the tetrahedron without rendering it as a solid object.
        Will be easier on the GPU when rendering many tetrahedrons, as it will reduce the number of triangles to render.

        Args:
            point (Union[List[float], np.ndarray]): The point on which to generate the tetrahedron.
            radius (float, optional): The radius of the tetrahedron. Defaults to 10e-9.
            color (Union[List[int], np.ndarray], optional): The color of the tetrahedron. Defaults to [1, 0, 0].
            make_line (bool, optional): Whether to generate a line set instead of a triangle mesh. Defaults to False.

        Returns:
            Union[o3d.geometry.TriangleMesh, o3d.geometry.LineSet]: The generated tetrahedron.
        """
        tetrahedron = o3d.geometry.TriangleMesh.create_tetrahedron(radius=radius)
        if make_line:
            tetrahedron = o3d.geometry.LineSet.create_from_triangle_mesh(tetrahedron)
        tetrahedron.translate(point, relative=False)
        tetrahedron.paint_uniform_color(color)
        return tetrahedron
