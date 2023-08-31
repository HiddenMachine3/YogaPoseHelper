class SuggestionBuilder:
    needed_angles = {_ for _ in range(11, 33)} - {17,18,19,20,21,22,29,30,32,31}

    def __init__(self, mp_pose):
        self.mp_pose = mp_pose

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
        for i in self.needed_angles:  # left_shoulder to right_foot index
            if arms_and_angles_diff[i]:
                for arms, angle in arms_and_angles_diff[i].items():
                    angles_list.append((i, angle))
        angles_list.sort(key=lambda x: x[1])

        txt = ""
        for landmark, angle in angles_list:
            if abs(angle) > angle_error_threshold:
                # if(mp_pose.PoseLandmark(landmark).name == "RIGHT_ELBOW"):
                txt += f"BEND {self.mp_pose.PoseLandmark(landmark).name} {'LESS' if angle<0 else 'MORE'} {angle}\n"
        
        return txt