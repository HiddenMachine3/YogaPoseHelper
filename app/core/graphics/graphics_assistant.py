from mediapipe.python.solutions.drawing_utils import landmark_pb2, Optional, Union, List, Mapping, RED_COLOR, WHITE_COLOR, _BGR_CHANNELS, _VISIBILITY_THRESHOLD, _PRESENCE_THRESHOLD, _normalized_to_pixel_coordinates, cv2
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.pyplot as plt
import numpy as np
import dataclasses
from typing import Tuple
from app.core.math.math_utility import clamp
from math import pi


def display_image(img, size, title):
    plt.figure(figsize=size)
    plt.title(title)
    plt.imshow(img[:, :, ::-1])
    plt.show()


# Function to generate gradient colors interpolating between two colors

def get_gradient_color(colors, n):
    start_color, end_color = colors
    r = np.linspace(start_color[0], end_color[0], n)
    g = np.linspace(start_color[1], end_color[1], n)
    b = np.linspace(start_color[2], end_color[2], n)
    return list(zip(r, g, b))


def draw_2_way_gradient_line_3d(
    ax, start, end, left_intensity, right_intensity, num_points=100
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
        [(0, 1, 0), (right_intensity, 1-right_intensity, 0)], num_points // 2
    )
    points = np.column_stack([x, y, z])

    line = Line3DCollection(
        list(zip(points[:-1], points[1:])), colors=colors, linewidths=3, zorder=2)

    # Add the line to the 3D plot
    ax.add_collection3d(line)


def draw_2_way_gradient_line_2d(
    cv2,img, start, end, left_intensity, right_intensity, num_points=100
):
    mid = ((start[0]+end[0])/2,(start[1]+end[1])/2)#,(start[2]+end[2])/2)
    # calculate gradient
    color1 = [left_intensity, 1-left_intensity, 0]
    color2 = [0,1,0]
    gradient = np.linspace(0, 1, num=100)

    for i in range(len(gradient)):
        x = int(start[0] + gradient[i] * (mid[0] - start[0]))
        y = int(start[1] + gradient[i] * (mid[1] - start[1]))
        color = tuple(int(255*(color1[j] + gradient[i] * (color2[j] - color1[j]))) for j in range(3))
        cv2.circle(img,(x,y), 3, color, -1)

    color1 = [right_intensity, 1-right_intensity, 0]
    for i in range(len(gradient)):
        x = int(end[0] + gradient[i] * (mid[0] - end[0]))
        y = int(end[1] + gradient[i] * (mid[1] - end[1]))
        color = tuple(int(255*(color1[j] + gradient[i] * (color2[j] - color1[j]))) for j in range(3))
        cv2.circle(img,(x,y), 3, color, -1)


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


def draw_error_landmarks_2d(
    arms_and_angles,ideal_arms_and_angles,
    img: np.ndarray,
    landmark_list: landmark_pb2.NormalizedLandmarkList,
    pronounce_error_by = 1,
    connections: Optional[List[Tuple[int, int]]] = None,
    landmark_drawing_spec: Union[DrawingSpec,
                                 Mapping[int, DrawingSpec]] = DrawingSpec(
                                     color=RED_COLOR),
    connection_drawing_spec: Union[DrawingSpec,
                                   Mapping[Tuple[int, int],
                                           DrawingSpec]] = DrawingSpec(),
    verbose:bool = True):

    if not landmark_list:
        return
    if img.shape[2] != _BGR_CHANNELS:
        raise ValueError('Input image must contain three channel bgr data.')
    image_rows, image_cols, _ = img.shape
    idx_to_coordinates = {}
    for idx, landmark in enumerate(landmark_list.landmark):
        if ((landmark.HasField('visibility') and
             landmark.visibility < _VISIBILITY_THRESHOLD) or
            (landmark.HasField('presence') and
             landmark.presence < _PRESENCE_THRESHOLD)):
            continue
        landmark_px = _normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                       image_cols, image_rows)
        if landmark_px:
            idx_to_coordinates[idx] = landmark_px
    if connections:
        num_landmarks = len(landmark_list.landmark)
        # Draws the connections if the start and end landmarks are both visible.
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
                raise ValueError(f'Landmark index is out of range. Invalid connection '
                                 f'from landmark #{start_idx} to landmark #{end_idx}.')
            if start_idx in idx_to_coordinates and end_idx in idx_to_coordinates:

                left, right = 0, 0
                # finding the difference b/w each angle at that vertex:
                if arms_and_angles[start_idx]:
                    for arms, angle in arms_and_angles[start_idx].items():
                        if end_idx in arms:
                            if arms in ideal_arms_and_angles[start_idx]:
                                diff = abs(
                                    ideal_arms_and_angles[start_idx][arms] - angle
                                )
                                left = max(left, diff / (180))

                if arms_and_angles[end_idx]:
                    for arms, angle in arms_and_angles[end_idx].items():
                        if start_idx in arms:
                            if arms in ideal_arms_and_angles[end_idx]:
                                diff = abs(
                                    ideal_arms_and_angles[end_idx][arms] - angle
                                )
                                right = max(right, diff / (180))

                # scaling up to pronounce errors
                left = clamp(left * pronounce_error_by, 0, 1)
                right = clamp(right * pronounce_error_by, 0, 1)
                
                # keypoint_colours[start_idx] = (left, 1 - left, 0)
                # keypoint_colours[end_idx] = (right, 1 - right, 0)

                draw_2_way_gradient_line_2d(
                    cv2=cv2,
                    img=img,
                    start=idx_to_coordinates[start_idx],
                    end=idx_to_coordinates[end_idx],
                    left_intensity=left,
                    right_intensity=right,
                    num_points=100,
                )

                if(verbose):
                    cv2.putText(img, f"{round(left,ndigits=3)}", idx_to_coordinates[start_idx], cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
                    cv2.putText(img, f"{round(right,ndigits=3)}", idx_to_coordinates[end_idx], cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

    # Draws landmark points after finishing the connection lines, which is
    # aesthetically better.
    if landmark_drawing_spec:
        for idx, landmark_px in idx_to_coordinates.items():
            drawing_spec = landmark_drawing_spec[idx] if isinstance(
                landmark_drawing_spec, Mapping) else landmark_drawing_spec
            # White circle border
            circle_border_radius = max(drawing_spec.circle_radius + 1,
                                       int(drawing_spec.circle_radius * 1.2))
            cv2.circle(img, landmark_px, circle_border_radius, WHITE_COLOR,
                       drawing_spec.thickness)
            # Fill color into the circle
            cv2.circle(img, landmark_px, drawing_spec.circle_radius,
                       drawing_spec.color, drawing_spec.thickness)
            
def plot_3d_error_graphics(
        landmark_list: landmark_pb2.NormalizedLandmarkList,
        connections: Optional[List[Tuple[int, int]]] = None,
        elevation: int = 10,
        azimuth: int = 10,
        fig_size=(10, 10),
        arms_and_angles = None,
        ideal_arms_and_angles=None,
        fig_title="",
        pronounce_error_by=10,
        plottable_landmarks=None,
        landmark_drawing_spec = DrawingSpec(color=(150, 150, 150))
    ):
        plt.figure(figsize=fig_size)
        ax = plt.axes(projection="3d")
        ax.set_title(fig_title)
        ax.view_init(elev=elevation, azim=azimuth)

        if (not ideal_arms_and_angles) or(not plottable_landmarks) or (not arms_and_angles):
            print("ideal angles not found")
            return

        # init all keypoint colours to green
        # a precense of red color will result in a gradient line
        keypoint_colours = {key: (0, 1, 0) for key in plottable_landmarks.keys()}

        # GRADIENT ERROR CONNECTORS
        if connections:
            num_landmarks = len(landmark_list.landmark)

            for connection in connections:
                start_idx = connection[0]
                end_idx = connection[1]
                # Draws the connections if the start and end landmarks are both visible.
                if not (
                    0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks
                ):
                    raise ValueError(
                        f"Landmark index is out of range. Invalid connection "
                        f"from landmark #{start_idx} to landmark #{end_idx}."
                    )
                if (
                    start_idx in plottable_landmarks
                    and end_idx in plottable_landmarks
                ):
                    landmark_pair = [
                        plottable_landmarks[start_idx],
                        plottable_landmarks[end_idx],
                    ]

                    left, right = 0, 0
                    # finding the difference b/w each angle at that vertex:
                    if arms_and_angles[start_idx]:
                        for arms, angle in arms_and_angles[start_idx].items():
                            if end_idx in arms:
                                if arms in ideal_arms_and_angles[start_idx]:
                                    diff = abs(
                                        ideal_arms_and_angles[start_idx][arms] - angle
                                    )
                                    left = max(left, diff / (180))

                    if arms_and_angles[end_idx]:
                        for arms, angle in arms_and_angles[end_idx].items():
                            if start_idx in arms:
                                if arms in ideal_arms_and_angles[end_idx]:
                                    diff = abs(
                                        ideal_arms_and_angles[end_idx][arms] - angle
                                    )
                                    right = max(right, diff / (180))

                    # scaling up to pronounce errors
                    left = clamp(left * pronounce_error_by, 0, 1)
                    right = clamp(right * pronounce_error_by, 0, 1)

                    keypoint_colours[start_idx] = (left, 1 - left, 0)
                    keypoint_colours[end_idx] = (right, 1 - right, 0)

                    draw_2_way_gradient_line_3d(
                        ax=ax,
                        start=landmark_pair[0],
                        end=landmark_pair[1],
                        left_intensity=left,
                        right_intensity=right,
                        num_points=100,
                    )

        # KEYPOINTS
        draw_keypoints_on_3d_graph(
            plottable_landmarks=plottable_landmarks,
            AX=ax,
            landmark_drawing_spec=landmark_drawing_spec,
        )

        # # print the keypoint names and number of angles associated with it
        # for idx, pos in plotted_landmarks.items():
        #     label = mp_pose.PoseLandmark(idx).name
        #     ax.text(
        #         pos[0],
        #         pos[1],
        #         pos[2],
        #         s=f"{idx}." + label # + ":" + str((arms_and_angles[idx]))
        #         if arms_and_angles[idx]
        #         else label,
        #         fontsize=font_size,
        #         color="blue",
        #         zorder=5,
        #         zdir="y",
        #     )

        plt.show()
