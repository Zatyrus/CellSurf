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
    ) -> o3d.geometry.TriangleMesh:
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
    ) -> o3d.geometry.TriangleMesh:
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
    ) -> o3d.geometry.TriangleMesh:
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
    ) -> o3d.geometry.TriangleMesh:
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
    ) -> o3d.geometry.TriangleMesh:
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
    ) -> o3d.geometry.TriangleMesh:
        tetrahedron = o3d.geometry.TriangleMesh.create_tetrahedron(radius=radius)
        if make_line:
            tetrahedron = o3d.geometry.LineSet.create_from_triangle_mesh(tetrahedron)
        tetrahedron.translate(point, relative=False)
        tetrahedron.paint_uniform_color(color)
        return tetrahedron
