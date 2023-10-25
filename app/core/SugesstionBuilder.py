from mediapipe.python.solutions.pose import PoseLandmark


class SuggestionBuilder:
    def __init__(self, mp_pose):
        self.mp_pose = mp_pose
        self.single_joint_landmarks = (
            {_ for _ in range(11, 33)}
            - {17, 18, 19, 20, 21, 22, 29, 30, 32, 31}
            - {12, 11}
        )
        self.useless_arms_sets = {
            frozenset({12, 23}),
            frozenset({11, 24}),
            frozenset({29, 31}),
            frozenset({30, 32}),
            frozenset({17, 19}),
            frozenset({20, 18}),
        }

        self.armpit_arms_set = {
            frozenset({14, 24}),
            frozenset({13, 23}),
        }
        self.shoulder_neck_arms_set = {frozenset({14, 11}), frozenset({13, 12})}

        self.inner_thigh_arms_set = {frozenset({23, 26}), frozenset({24, 25})}

        self.outer_thigh_arms_set = {frozenset({12, 26}), frozenset({11, 25})}

    def get_suggestions(self, arms_and_angles_diff, angle_error_threshold: float):
        """
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

        angles_list = []
        for (
            landmark
        ) in self.single_joint_landmarks:  # left_shoulder to right_foot index
            if arms_and_angles_diff[landmark]:
                for arms, angle in arms_and_angles_diff[landmark].items():
                    if arms in self.useless_arms_sets:
                        continue

                    angles_list.append((landmark, angle))
        # angles_list.sort(key=lambda x: x[1])
        # construct messages for single joint landmarks
        txt = ""
        for landmark, angle in angles_list:
            
            if abs(angle) > angle_error_threshold:
                # if(mp_pose.PoseLandmark(landmark).name == "RIGHT_ELBOW"):
                txt += f"BEND {self.mp_pose.PoseLandmark(landmark).name} {'LESS' if angle>0 else 'MORE'} {angle}\n"

        # constuct messages for shoulders
        for landmark in {
            PoseLandmark.RIGHT_SHOULDER,
            PoseLandmark.LEFT_SHOULDER,
        }:  # These are shoulder landmarks
            if arms_and_angles_diff[landmark]:
                for arms, angle in arms_and_angles_diff[landmark].items():
                    if abs(angle) > angle_error_threshold:
                        if arms in self.armpit_arms_set:
                            txt += f"BEND {self.mp_pose.PoseLandmark(landmark).name[:4]} arm {'MORE away from' if angle>0 else 'MORE towards'} torso {angle}\n"
                        elif arms in self.shoulder_neck_arms_set:
                            txt += f"BEND {self.mp_pose.PoseLandmark(landmark).name[:4]} arm {'MORE away from' if angle>0 else 'MORE towards'} neck {angle}\n"

        return txt
