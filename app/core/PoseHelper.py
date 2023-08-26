from __future__ import annotations
import math
import cv2
import numpy as np
from time import time
import mediapipe as mp
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
from mediapipe.framework.formats import landmark_pb2


from app.core.math.math_utility import find_3d_angle
from app.core.math.math_utility import clamp
import app.core.graphics.graphics_assistant as graphic

_PRESENCE_THRESHOLD = 0.2  # for error skeleton calculations only
_VISIBILITY_THRESHOLD = 0.2  # for error skeleton calculations only

class PoseHelper:
    def __init__(self, img_path=None,img=None,mp_pose=None,pose=None,mp_drawing=None):
        self.img = cv2.imread(img_path) if (img is None) else img
        self.mp_pose = mp.solutions.pose if not mp_pose else mp_pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True, min_detection_confidence=0.3, model_complexity=2
        ) if not pose else pose
        self.mp_drawing = mp.solutions.drawing_utils if not mp_drawing else mp_drawing
        self.plottable_landmarks = {}
        self.img_height,self.img_width, _ =self.img.shape

    def display_img(self, fig_size, fig_title):
        graphic.display_image(self.img, fig_size, fig_title)

    def detect_keypoints(self, verbose=True):
        self.results = self.pose.process(
            cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        )  # converting image to rgb format
        self.norm_landmarks = []
        if self.results.pose_landmarks:
            for i in range(33):
                norm_landmark = self.results.pose_landmarks.landmark[
                    self.mp_pose.PoseLandmark(i).value
                ]
                if verbose:
                    print(f"{self.mp_pose.PoseLandmark(i).name}:\n{[norm_landmark.x *self.img_width, norm_landmark.y*self.img_height, norm_landmark.z*self.img_width]}")
                
                #converting normalised landmarks to image coordinates
                self.norm_landmarks.append(
                    np.array([norm_landmark.x, norm_landmark.y, norm_landmark.z])#np.array([norm_landmark.x *self.img_width, norm_landmark.y*self.img_height, norm_landmark.z*self.img_width])
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

        # each index of connected_points is a vertex on the skeleton, and contains a set of which points are connected to it
        connected_points = [set() for _ in range(len(landmarks))]

        # # # LOOPING THROUGH EACH LANDMARK
        for idx, landmark in enumerate(landmark_list.landmark):
            if (
                landmark and
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
        # lets figure out the angles for every set of 3 connected points (1 vertex and 2 arms, ex: L)

        """
        looping through every point again to find the
        angle between every group of arm-vertex-arm
        """
        # print(landmarks)
        for vertex in range(33):#len(connected_points)):
            points = list(connected_points[vertex])

            num_connections = len(points)
            if num_connections > 1:
                angles = []
                arms = []
                for i in range(num_connections - 1):
                    for j in range(i + 1, num_connections):
                        try:
                            # print(landmarks[points[i]],landmarks[vertex],landmarks[points[j]])

                            angle = find_3d_angle(
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

    

    def calculate_angles(self):
        """
        for each vertex, it calculates which sets of 2 arms are connected, and the angles b/w these 2 arms for each set


        arms_and_angles is a list of {vertex,2 arms connected to it, and angle made up by them}
        the index of the list is the vertex number
        at each index, you have a dictionary of:
            {
                {each set of 2 arms are connected to vertex} : angle between arm1,vertex,arm2
            }

        ex: [{{arm1,arm2}:a12, {arm1,arm3}:a13,{arm2,arm3}:a23}, # this is for vertex 0
        {{arm1,arm2}:a12, {arm1,arm3}:a13,{arm2,arm3}:a23}, # this is for vertex 1
        {{arm1,arm2}:a12, {arm1,arm3}:a13,{arm2,arm3}:a23},...] # and so on


        """
        self.arms_and_angles = self.calculate_arms_and_angles(
            self.norm_landmarks,
            self.results.pose_world_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
        )

    def draw3dErrorDetectedSkeleton(self, idealPose, title="", pronounce_error_by=10):
        if self.results.pose_landmarks:
            graphic.plot_3d_error_graphics(
                self.results.pose_world_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                arms_and_angles=self.arms_and_angles,
                ideal_arms_and_angles=idealPose.arms_and_angles,
                fig_title=title,
                pronounce_error_by=pronounce_error_by,
                plottable_landmarks=self.plottable_landmarks
            )
    
    @staticmethod
    def calculate_angle_differences(arms_and_angles_1,arms_and_angles_2,n): # n = len(landmarks)
        arms_and_angles_diff = [{} for _ in range(n)]
        
        for i in range(n):
            if arms_and_angles_1[i] and arms_and_angles_2[i]:
                a1 = arms_and_angles_1[i].keys()
                a2 = arms_and_angles_2[i].keys()

                for arms in a1:
                    if arms in a2:
                        arms_and_angles_diff[i][arms] = abs(arms_and_angles_1[i][arms] - arms_and_angles_2[i][arms])
            
        return arms_and_angles_diff