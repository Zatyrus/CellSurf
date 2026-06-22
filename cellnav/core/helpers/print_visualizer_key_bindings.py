## This file is part of CellNav, a toolbox for 3D cell surface extraction and analysis.
__all__ = ["print_visualizer_key_bindings"]


def print_visualizer_key_bindings() -> None:
    """
    Print the key bindings for the Open3D visualizer.
    This can be useful for users who are not familiar with the Open3D visualizer and want to know how to interact with it.
    """

    print(
        """
        == Mouse view control ==
            Left button + drag         : Rotate.
            Ctrl + left button + drag  : Translate.
            Wheel button + drag        : Translate.
            Shift + left button + drag : Roll.
            Wheel                      : Zoom in/out.

        == Keyboard view control ==
            [/]          : Increase/decrease field of view.
            R            : Reset view point.
            Ctrl/Cmd + C : Copy current view status into the clipboard.
            Ctrl/Cmd + V : Paste view status from clipboard.

        == General control ==
            Q, Esc       : Exit window.
            H            : Print help message.
            P, PrtScn    : Take a screen capture.
            D            : Take a depth capture.
            O            : Take a capture of current rendering settings.
            Alt + Enter  : Toggle between full screen and windowed mode.

        == Render mode control ==
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

        == Color control ==
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
                        
        == Editing control ==
            Y            : Lock Y-Axis when picking points.
            X            : Lock X-Axis when picking points.
            Z            : Lock Z-Axis when picking points.
            F            : Go back to Free-View mode when picking points.
            K            : Lock view and start picking points. Drag to select points, then release to finish picking.
            C            : confirm selection of points when picking points. Press K again to start a new selection.
            S            : Save selected points to a .ply file when picking points.
            
        == Point picking mode ==
            Shift + Mouse left button : Select points by clicking. Get points by calling vis.get_picked_points() after selection.
            Shift + Mouse right button : Deselect points. Get points by calling vis.get_picked_points() after selection.
        """
    )
