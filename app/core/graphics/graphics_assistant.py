import matplotlib.pyplot as plt
import numpy as np
import dataclasses
from typing import Tuple

def display_image(img,size, title):
    plt.figure(figsize=size)
    plt.title(title)
    plt.imshow(img[:, :, ::-1])
    plt.show()


from mpl_toolkits.mplot3d.art3d import Line3DCollection

# Function to generate gradient colors interpolating between two colors
def get_gradient_color(colors, n):
    start_color, end_color = colors
    r = np.linspace(start_color[0], end_color[0], n)
    g = np.linspace(start_color[1], end_color[1], n)
    b = np.linspace(start_color[2], end_color[2], n)
    return list(zip(r, g, b))

def draw_2_way_gradient_line_3d(
    ax_thing, start, end, left_intensity, right_intensity, num_points=100
):
    # Generate points along the line in 3D space
    x = np.linspace(start[0], end[0], num_points)
    y = np.linspace(start[1], end[1], num_points)
    z = np.linspace(start[2], end[2], num_points)
    
    # red to green to red interpolation based on red intensity on both ends 
    colors = get_gradient_color(
        [(left_intensity, 1-left_intensity, 0), (0, 1, 0)],
        num_points // 2,
    ) + get_gradient_color(
        [(0,1,0), (right_intensity, 1-right_intensity, 0)], num_points // 2
    )
    points = np.column_stack([x, y, z])

    line = Line3DCollection(list(zip(points[:-1], points[1:])), colors=colors, linewidths=3,zorder= 2)

    # Add the line to the 3D plot
    ax_thing.add_collection3d(line)

def _normalize_color(color):
    return tuple(v / 255.0 for v in color)

@dataclasses.dataclass
class DrawingSpec:
    # Color for drawing the annotation. Default to the white color.
    color: Tuple[int, int, int] = (224, 224, 224)
    # Thickness for drawing the annotation. Default to 2 pixels.
    thickness: int = 2
    # Circle radius. Default to 2 pixels.
    circle_radius: int = 2


def draw_keypoints_on_3d_graph(
    plottable_landmarks,
    AX,
    landmark_drawing_spec
):
    for key, plottable_landmark in plottable_landmarks.items():
        AX.scatter3D(
            xs=plottable_landmark[0],
            ys=plottable_landmark[1],
            zs=plottable_landmark[2],
            color=_normalize_color(landmark_drawing_spec.color[::-1]),
            linewidth=landmark_drawing_spec.thickness,
            zorder=1,
        )