import math
import cv2
import numpy as np
from time import time
import mediapipe as mp
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
from mediapipe.framework.formats import landmark_pb2


from app.core.math.math_utility import findAngle
from app.core.math.math_utility import clamp
import app.core.graphics.graphics_assistant as graphic
from app.core.graphics.graphics_assistant import DrawingSpec

_PRESENCE_THRESHOLD = 0.5
_VISIBILITY_THRESHOLD = 0.5


class PoseHelperClass:
    def __init__(self, img_path):
        self.img = cv2.imread(img_path)
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True, min_detection_confidence=0.3, model_complexity=2
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.plottable_landmarks = {}

        self.landmark_drawing_spec = DrawingSpec(color=(150,150,150))

    def display_img(self, fig_size, fig_title):
        graphic.display_image(self.img, fig_size, fig_title)

    def detect_keypoints(self, verbose=True):
        self.results = self.pose.process(
            cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        )  # converting image to rgb format
        self.landmarks = []
        if self.results.pose_landmarks:
            for i in range(33):
                norm_landmark = self.results.pose_landmarks.landmark[
                    self.mp_pose.PoseLandmark(i).value
                ]
                if verbose:
                    print(f"{self.mp_pose.PoseLandmark(i).name}:\n{norm_landmark}")
                self.landmarks.append(
                    np.array([norm_landmark.x, norm_landmark.y, norm_landmark.z])
                )

    def plot_keypoints2d(self, fig_title="", figsize=[5, 5]):
        img_copy = self.img.copy()
        if self.results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image=img_copy,
                landmark_list=self.results.pose_landmarks,
                connections=self.mp_pose.POSE_CONNECTIONS,
            )
            fig = plt.figure(figsize=figsize)
            plt.title(fig_title)
            plt.imshow(img_copy[:, :, ::-1])
            plt.show()

    def plot_keypoints3d(self):
        if self.results.pose_landmarks:
            self.mp_drawing.plot_landmarks(
                self.results.pose_world_landmarks, self.mp_pose.POSE_CONNECTIONS
            )

    def calculate_arms_and_angles(
        self,
        landmarks,
        landmark_list: landmark_pb2.NormalizedLandmarkList,
        connections: Optional[List[Tuple[int, int]]] = None,
    ):
        # {i:set() for i in range(len(landmarks))}
        arms_and_angles = [None for _ in range(len(landmarks))]
        connected_points = [set() for _ in range(len(self.landmarks))]

        # # # LOOPING THROUGH EACH LANDMARK
        for idx, landmark in enumerate(landmark_list.landmark):
            if (
                landmark.HasField("visibility")
                and landmark.visibility < _VISIBILITY_THRESHOLD
            ) or (
                landmark.HasField("presence")
                and landmark.presence < _PRESENCE_THRESHOLD
            ):
                continue

            self.plottable_landmarks[idx] = (-landmark.z, landmark.x, -landmark.y)

        if connections:
            num_landmarks = len(landmark_list.landmark)

            for connection in connections:
                start_idx = connection[0]
                end_idx = connection[1]

                if not (
                    0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks
                ):
                    raise ValueError(
                        f"Landmark index is out of range. Invalid connection "
                        f"from landmark #{start_idx} to landmark #{end_idx}."
                    )
                if (
                    start_idx in self.plottable_landmarks
                    and end_idx in self.plottable_landmarks
                ):
                    try:
                        # add each other the other's 'connected points' set
                        # print(start_idx,end_idx,f"{'same!' if start_idx == end_idx else ''}")
                        connected_points[start_idx].add(end_idx)
                        connected_points[end_idx].add(start_idx)
                    except IndexError as e:
                        print(e)
                        print(
                            f"start_idx:{start_idx},end_idx:{end_idx},len(connected_points):{len(connected_points)}"
                        )

        # now we got a set of connected points for each landmark,
        # lets figure out the angles for every set of 3 connected points ( v ^ L \_ )

        """
        looping through every point again to find the
        angle between every group of 3 linearly connected points
        """
        # print(landmarks)
        for vertex in range(len(connected_points)):
            points = list(connected_points[vertex])

            num_connections = len(points)
            if num_connections > 1:
                angles = []
                arms = []
                for i in range(num_connections - 1):
                    for j in range(i + 1, num_connections):
                        try:
                            # print(landmarks[points[i]],landmarks[vertex],landmarks[points[j]])

                            angle = findAngle(
                                landmarks[points[i]],
                                landmarks[vertex],
                                landmarks[points[j]],
                            )
                            angles.append(angle)
                            arms.append(frozenset({points[i], points[j]}))
                        except IndexError as e:
                            print(
                                f"i:{i},j:{j},len(points):{len(points)},len(landmarks):{len(landmarks)},points[i]:{points[i]},points[j]:{points[j]}"
                            )
                # have to convert arms set to a frozenset because keys have to be of immutable type
                arms_and_angles[vertex] = dict(zip(arms, angles))

        return arms_and_angles

    def plot_3d_graphics(
        self,
        landmark_list: landmark_pb2.NormalizedLandmarkList,
        connections: Optional[List[Tuple[int, int]]] = None,
        elevation: int = 10,
        azimuth: int = 10,
        fig_size=(10, 10),
        ideal_arms_and_angles=None,
        fig_title="",
        pronounce_error_by=10
    ):
        plt.figure(figsize=fig_size)
        ax = plt.axes(projection="3d")
        ax.set_title(fig_title)
        ax.view_init(elev=elevation, azim=azimuth)

        if not ideal_arms_and_angles:
            print("ideal angles not found")
            return

        keypoint_colours = {key: (0, 1, 0) for key in self.plottable_landmarks.keys()}

        # GRADIENT ERROR CONNECTORS
        if connections:
            num_landmarks = len(landmark_list.landmark)
            # Draws the connections if the start and end landmarks are both visible.

            for connection in connections:
                start_idx = connection[0]
                end_idx = connection[1]

                if not (
                    0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks
                ):
                    raise ValueError(
                        f"Landmark index is out of range. Invalid connection "
                        f"from landmark #{start_idx} to landmark #{end_idx}."
                    )
                if (
                    start_idx in self.plottable_landmarks
                    and end_idx in self.plottable_landmarks
                ):
                    landmark_pair = [
                        self.plottable_landmarks[start_idx],
                        self.plottable_landmarks[end_idx],
                    ]

                    left, right = 0, 0

                    if self.arms_and_angles[start_idx]:
                        for arms, angle in self.arms_and_angles[start_idx].items():
                            if end_idx in arms:
                                if arms in ideal_arms_and_angles[start_idx]:
                                    diff = abs(
                                        ideal_arms_and_angles[start_idx][arms] - angle
                                    )
                                    left = max(left, diff / (2 * math.pi))

                    if self.arms_and_angles[end_idx]:
                        for arms, angle in self.arms_and_angles[end_idx].items():
                            if end_idx in arms:
                                if arms in ideal_arms_and_angles[end_idx]:
                                    diff = abs(
                                        ideal_arms_and_angles[end_idx][arms] - angle
                                    )
                                    right = max(right, diff / (2 * math.pi))
                    
                    # scaling up to pronounce errors
                    left = clamp(left*pronounce_error_by,0,1)
                    right = clamp(right*pronounce_error_by,0,1)
                   
                    keypoint_colours[start_idx] = (left, 1 - left, 0)
                    keypoint_colours[end_idx] = (right, 1 - right, 0)

                    graphic.draw_2_way_gradient_line_3d(
                        ax_thing=ax,
                        start=landmark_pair[0],
                        end=landmark_pair[1],
                        left_intensity=left,
                        right_intensity=right,
                        num_points=100,
                    )

        # KEYPOINTS
        graphic.draw_keypoints_on_3d_graph(
            plottable_landmarks=self.plottable_landmarks,
            AX=ax,
            landmark_drawing_spec=self.landmark_drawing_spec,
        )

        # # print the keypoint names and number of angles associated with it
        # for idx, pos in self.plotted_landmarks.items():
        #     label = self.mp_pose.PoseLandmark(idx).name
        #     ax.text(
        #         pos[0],
        #         pos[1],
        #         pos[2],
        #         s=f"{idx}." + label # + ":" + str((self.arms_and_angles[idx]))
        #         if self.arms_and_angles[idx]
        #         else label,
        #         fontsize=font_size,
        #         color="blue",
        #         zorder=5,
        #         zdir="y",
        #     )

        plt.show()

    def calculate_angles(self):
        self.arms_and_angles = self.calculate_arms_and_angles(
            self.landmarks,
            self.results.pose_world_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
        )

    def draw3dErrorDetectedSkeleton(self, idealPose, title="", pronounce_error_by=10):
        if self.results.pose_landmarks:
            self.plot_3d_graphics(
                self.results.pose_world_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                ideal_arms_and_angles=idealPose.arms_and_angles,
                fig_title=title,
                pronounce_error_by = pronounce_error_by
            )
