from __future__ import annotations
import math
import cv2
import numpy as np
from time import time
import mediapipe as mp
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions.pose import PoseLandmark


from app.core.math.math_utility import find_3d_angle
from app.core.math.math_utility import clamp
import app.core.graphics.graphics_assistant as graphic
from vendor.join_angle_calculation.joint_angle_calculator import get_angle_deviations

_PRESENCE_THRESHOLD = 0.1  # for error skeleton calculations only
_VISIBILITY_THRESHOLD = 0.1  # for error skeleton calculations only


class PoseHelper:
    def __init__(
        self, img_path=None, img=None, mp_pose=None, pose=None, mp_drawing=None
    ):
        self.img = cv2.imread(img_path) if (img is None) else img
        self.mp_pose = mp.solutions.pose if not mp_pose else mp_pose
        self.pose = (
            self.mp_pose.Pose(
                static_image_mode=True, min_detection_confidence=0.3, model_complexity=2
            )
            if not pose
            else pose
        )
        self.mp_drawing = mp.solutions.drawing_utils if not mp_drawing else mp_drawing
        self.plottable_landmarks = {}
        self.img_height, self.img_width, _ = self.img.shape

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
                    print(
                        f"{self.mp_pose.PoseLandmark(i).name}:\n{[norm_landmark.x *self.img_width, norm_landmark.y*self.img_height, norm_landmark.z*self.img_width]}"
                    )

                # converting normalised landmarks to image coordinates
                # NOTE: MAKE SURE THE IMAGE IN THE IDEAL AND TEST inputs HAVE THE SAME SIZE(width and height)
                self.landmarks.append(
                    np.array(
                        [
                            norm_landmark.x * self.img_width, # 
                            norm_landmark.y * self.img_height, # 
                            0 # norm_landmark.z, # * self.img_width
                        ]
                    )  # np.array([norm_landmark.x*self.img_width, norm_landmark.y*self.img_height, norm_landmark.z*self.img_width])#
                )
                # self.norm_landmarks.append(np.array())

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
        

        # TODO : REMOVE BELOW COMMENTED CODE
        # # each index of connected_points is a vertex on the skeleton, and contains a set of which points are connected to it
        # connected_points = [set() for _ in range(len(landmarks))]
        # {i:set() for i in range(len(landmarks))}
        # arms_and_angles = [None for _ in range(len(landmarks))]

        # # # LOOPING THROUGH EACH LANDMARK
        
        for idx, landmark in enumerate(landmark_list.landmark):
            if (
                landmark
                and landmark.HasField("visibility")
                and landmark.visibility < _VISIBILITY_THRESHOLD
            ) or (
                landmark.HasField("presence")
                and landmark.presence < _PRESENCE_THRESHOLD
            ):
                continue

            self.plottable_landmarks[idx] = (-landmark.z, landmark.x, -landmark.y)

        
        # TODO : REFACTOR THIS CODE, cause this is a very hacky solution since I didn't have time
        # there is a much more elegant way of doing this
        angles = get_angle_deviations(landmarks)
        
        arms_and_angles = [{},#NOSE = 0
         {},#  LEFT_EYE_INNER = 1
         {},#  LEFT_EYE = 2
         {},#  LEFT_EYE_OUTER = 3
         {},#  RIGHT_EYE_INNER = 4
         {},#  RIGHT_EYE = 5
         {},#  RIGHT_EYE_OUTER = 6
         {},#  LEFT_EAR = 7
         {},#  RIGHT_EAR = 8
         {},#  MOUTH_LEFT = 9
         {},#  MOUTH_RIGHT = 10
         {frozenset({PoseLandmark.LEFT_ELBOW,PoseLandmark.RIGHT_SHOULDER}):np.array([0,0,0])},#  LEFT_SHOULDER = 11
         {frozenset({PoseLandmark.RIGHT_ELBOW,PoseLandmark.LEFT_SHOULDER}):np.array([0,0,0])},#  RIGHT_SHOULDER = 12
         {frozenset({PoseLandmark.LEFT_SHOULDER,PoseLandmark.LEFT_WRIST}):np.array([0,0,0])},#  LEFT_ELBOW = 13
         {frozenset({PoseLandmark.RIGHT_SHOULDER,PoseLandmark.RIGHT_WRIST}):np.array([0,0,0])},#  RIGHT_ELBOW = 14
         {},#actually need, but will implement later#  LEFT_WRIST = 15
         {},#actually need, but will implement later#  RIGHT_WRIST = 16
         {},#  LEFT_PINKY = 17
         {},#  RIGHT_PINKY = 18
         {},#  LEFT_INDEX = 19
         {},#  RIGHT_INDEX = 20
         {},#  LEFT_THUMB = 21
         {},#  RIGHT_THUMB = 22
         {frozenset({PoseLandmark.LEFT_KNEE,PoseLandmark.RIGHT_HIP}):np.array([0,0,0])},#  LEFT_HIP = 23
         {frozenset({PoseLandmark.RIGHT_KNEE,PoseLandmark.LEFT_HIP}):np.array([0,0,0])},#  RIGHT_HIP = 24
         {frozenset({PoseLandmark.LEFT_HIP,PoseLandmark.LEFT_ANKLE}):np.array([0,0,0])},#  LEFT_KNEE = 25
         {frozenset({PoseLandmark.RIGHT_HIP,PoseLandmark.RIGHT_ANKLE}):np.array([0,0,0])},#  RIGHT_KNEE = 26
         {},#  LEFT_ANKLE = 27
         {},#  RIGHT_ANKLE = 28
         {},#  LEFT_HEEL = 29
         {},#  RIGHT_HEEL = 30
         {},#  LEFT_FOOT_INDEX = 31
         {}]#  RIGHT_FOOT_INDEX = 32

        for i in range(33):
            if arms_and_angles[i]:
                arms_and_angles[i][list(arms_and_angles[i].keys())[0]] = angles[i]

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
            self.landmarks,
            self.results.pose_world_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
        )

    def draw3dErrorDetectedSkeleton(
        self, idealPose, title="", pronounce_error_by=10, verbose=False
    ):
        if self.results.pose_landmarks:
            graphic.plot_3d_error_graphics(
                self.results.pose_world_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                arms_and_angles=self.arms_and_angles,
                ideal_arms_and_angles=idealPose.arms_and_angles,
                fig_title=title,
                pronounce_error_by=pronounce_error_by,
                plottable_landmarks=self.plottable_landmarks,
                mp_pose=self.mp_pose,
                verbose=verbose,
            )

    @staticmethod
    def calculate_angle_differences(
        arms_and_angles_1, arms_and_angles_2, n
    ):  # n = len(landmarks)
        arms_and_angles_diff = [{} for _ in range(n)]

        for i in range(n):
            if arms_and_angles_1[i] and arms_and_angles_2[i]:
                a1 = arms_and_angles_1[i].keys()
                a2 = arms_and_angles_2[i].keys()

                for arms in a1:
                    if arms in a2:
                        arms_and_angles_diff[i][arms] = (
                            arms_and_angles_1[i][arms] - arms_and_angles_2[i][arms]
                        )

        return arms_and_angles_diff
