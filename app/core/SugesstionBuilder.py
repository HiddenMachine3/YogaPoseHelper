from mediapipe.python.solutions.pose import PoseLandmark
import numpy as np
import mediapipe as mp


class SuggestionBuilder:
    def __init__(self):
        self.mp_pose = mp.solutions.pose

    def get_suggestions(self, arms_and_angles_diff):
        txt = []
        for i in range(33):
            if arms_and_angles_diff[i]:
                diff = (
                    arms_and_angles_diff[i][list(arms_and_angles_diff[i].keys())[0]]
                    * 180
                    / np.pi
                )
                angle = diff[0]
                landmark_name = self.mp_pose.PoseLandmark(i).name
                if landmark_name.startswith("LEFT"):
                    angle *= -1
                # txt += f"BEND {landmark_name} {'LESS' if angle>0 else 'MORE'} {angle} {list(diff)}\n"
                abs_diff = abs(angle)
                msg = "Very Bad"
                if abs_diff < 10:
                    msg = "Perfect"
                elif abs_diff < 20:
                    msg = "Good"
                elif abs_diff < 30:
                    msg = "Okay"
                elif abs_diff < 40:
                    msg = "Not Good"
                txt.append(((landmark_name + " " + msg),abs_diff))
        return txt