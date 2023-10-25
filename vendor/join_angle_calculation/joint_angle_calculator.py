import numpy as np
import sys
import vendor.join_angle_calculation.utils as utils
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mediapipe as mp

needed_landmarks = {
    "LEFT_HIP",
    "LEFT_KNEE",
    "LEFT_ANKLE",
    "RIGHT_HIP",
    "RIGHT_KNEE",
    "RIGHT_ANKLE",
    "LEFT_SHOULDER",
    "LEFT_ELBOW",
    "LEFT_WRIST",
    "RIGHT_SHOULDER",
    "RIGHT_ELBOW",
    "RIGHT_WRIST",
}

keypoints_to_index = {
    "lefthip": 6,
    "leftknee": 8,
    "leftfoot": 10,
    "righthip": 7,
    "rightknee": 9,
    "rightfoot": 11,
    "leftshoulder": 0,
    "leftelbow": 2,
    "leftwrist": 4,
    "rightshoulder": 1,
    "rightelbow": 3,
    "rightwrist": 5,
}
mediapipe_landmark_names_map = {
    "LEFT_HIP": "lefthip",
    "LEFT_KNEE": "leftknee",
    "LEFT_ANKLE": "leftfoot",
    "RIGHT_HIP": "righthip",
    "RIGHT_KNEE": "rightknee",
    "RIGHT_ANKLE": "rightfoot",
    "LEFT_SHOULDER": "leftshoulder",
    "LEFT_ELBOW": "leftelbow",
    "LEFT_WRIST": "leftwrist",
    "RIGHT_SHOULDER": "rightshoulder",
    "RIGHT_ELBOW": "rightelbow",
    "RIGHT_WRIST": "rightwrist",
    "lefthip": "LEFT_HIP",
    "leftknee": "LEFT_KNEE",
    "leftfoot": "LEFT_ANKLE",
    "righthip": "RIGHT_HIP",
    "rightknee": "RIGHT_KNEE",
    "rightfoot": "RIGHT_ANKLE",
    "leftshoulder": "LEFT_SHOULDER",
    "leftelbow": "LEFT_ELBOW",
    "leftwrist": "LEFT_WRIST",
    "rightshoulder": "RIGHT_SHOULDER",
    "rightelbow": "RIGHT_ELBOW",
    "rightwrist": "RIGHT_WRIST",
}
index_to_keypoints = {val: key for key, val in keypoints_to_index.items()}

needed_mp_landmark_indices = sorted(
    [mp.solutions.pose.PoseLandmark[name].value for name in needed_landmarks],
    key=lambda x: keypoints_to_index[
        mediapipe_landmark_names_map[mp.solutions.pose.PoseLandmark(x).name]
    ],
)


def landmarks_to_kpts(landmarks: list):
    # selecting the needed 12 features and reshaping the numpy array to
    # be compatible with the rest of the code
    return np.array(landmarks)[needed_mp_landmark_indices].reshape(
        (1, len(needed_mp_landmark_indices), 3)
    )


def read_keypoints(filename):
    num_keypoints = 12
    fin = open(filename, "r")
    kpts = []
    # read only the first line of the file
    line = fin.readline()
    line = line.split()
    line = [float(s) for s in line]
    line = np.reshape(line, (num_keypoints, -1))
    kpts.append(line)
    kpts = np.array(kpts)
    print(f"type(kpts) : {type(kpts)}\nkpts : {kpts}")
    return kpts


def convert_to_dictionary(kpts):
    # its easier to manipulate keypoints by joint name
    global keypoints_to_index

    kpts_dict = {}
    for key, k_index in keypoints_to_index.items():
        kpts_dict[key] = kpts[:, k_index]

    kpts_dict["joints"] = list(keypoints_to_index.keys())

    return kpts_dict


def add_hips_and_neck(kpts):
    # we add two new keypoints which are the mid point between the hips and mid point between the shoulders

    # add hips kpts
    difference = kpts["lefthip"] - kpts["righthip"]
    difference = difference / 2
    hips = kpts["righthip"] + difference
    kpts["hips"] = hips
    kpts["joints"].append("hips")

    # add neck kpts
    difference = kpts["leftshoulder"] - kpts["rightshoulder"]
    difference = difference / 2
    neck = kpts["rightshoulder"] + difference
    kpts["neck"] = neck
    kpts["joints"].append("neck")

    # define the hierarchy of the joints
    hierarchy = {
        "hips": [],
        "lefthip": ["hips"],
        "leftknee": ["lefthip", "hips"],
        "leftfoot": ["leftknee", "lefthip", "hips"],
        "righthip": ["hips"],
        "rightknee": ["righthip", "hips"],
        "rightfoot": ["rightknee", "righthip", "hips"],
        "neck": ["hips"],
        "leftshoulder": ["neck", "hips"],
        "leftelbow": ["leftshoulder", "neck", "hips"],
        "leftwrist": ["leftelbow", "leftshoulder", "neck", "hips"],
        "rightshoulder": ["neck", "hips"],
        "rightelbow": ["rightshoulder", "neck", "hips"],
        "rightwrist": ["rightelbow", "rightshoulder", "neck", "hips"],
    }

    kpts["hierarchy"] = hierarchy
    kpts["root_joint"] = "hips"

    return kpts


# remove jittery keypoints by applying a median filter along each axis
def median_filter(kpts, window_size=3):
    import copy

    filtered = copy.deepcopy(kpts)

    from scipy.signal import medfilt

    # apply median filter to get rid of poor keypoints estimations
    for joint in filtered["joints"]:
        joint_kpts = filtered[joint]
        xs = joint_kpts[:, 0]
        ys = joint_kpts[:, 1]
        zs = joint_kpts[:, 2]
        xs = medfilt(xs, window_size)
        ys = medfilt(ys, window_size)
        zs = medfilt(zs, window_size)
        filtered[joint] = np.stack([xs, ys, zs], axis=-1)

    return filtered


def get_bone_lengths(kpts):
    """
    We have to define an initial skeleton pose(T pose).
    In this case we need to known the length of each bone.
    Here we calculate the length of each bone from data
    """

    bone_lengths = {}
    for joint in kpts["joints"]:
        if joint == "hips":
            continue
        parent = kpts["hierarchy"][joint][0]

        joint_kpts = kpts[joint]
        parent_kpts = kpts[parent]

        _bone = joint_kpts - parent_kpts
        _bone_lengths = np.sqrt(np.sum(np.square(_bone), axis=-1))

        # _bone_length = np.median(_bone_lengths)
        print(
            f"joint : {joint} parent: {parent}, kpts[joint] : {kpts[joint]},  kpts[parent] : {kpts[parent]}"
        )
        bone_lengths[joint] = _bone_lengths  # _bone_length

        # plt.hist(bone_lengths, bins = 25)
        # plt.title(joint)
        # plt.show()

    # print(bone_lengths)
    kpts["bone_lengths"] = bone_lengths
    return


# Here we define the T pose and we normalize the T pose by the length of the hips to neck distance.
def get_base_skeleton(kpts, normalization_bone="neck"):
    # this defines a generic skeleton to which we can apply rotations to
    body_lengths = kpts["bone_lengths"]

    # define skeleton offset directions
    offset_directions = {}
    offset_directions["lefthip"] = np.array([1, 0, 0])
    offset_directions["leftknee"] = np.array([0, -1, 0])
    offset_directions["leftfoot"] = np.array([0, -1, 0])

    offset_directions["righthip"] = np.array([-1, 0, 0])
    offset_directions["rightknee"] = np.array([0, -1, 0])
    offset_directions["rightfoot"] = np.array([0, -1, 0])

    offset_directions["neck"] = np.array([0, 1, 0])

    offset_directions["leftshoulder"] = np.array([1, 0, 0])
    offset_directions["leftelbow"] = np.array([1, 0, 0])
    offset_directions["leftwrist"] = np.array([1, 0, 0])

    offset_directions["rightshoulder"] = np.array([-1, 0, 0])
    offset_directions["rightelbow"] = np.array([-1, 0, 0])
    offset_directions["rightwrist"] = np.array([-1, 0, 0])

    # set bone normalization length. Set to 1 if you dont want normalization
    normalization = kpts["bone_lengths"][normalization_bone]
    # print(f"normalization : {normalization}")
    # normalization = 1

    # base skeleton set by multiplying offset directions by measured bone lengths. In this case we use the average of two sided limbs. E.g left and right hip averaged
    base_skeleton = {"hips": np.array([0, 0, 0])}

    def _set_length(joint_type):
        base_skeleton["left" + joint_type] = offset_directions["left" + joint_type] * (
            (body_lengths["left" + joint_type] + body_lengths["right" + joint_type])
            / (2 * normalization)
        )
        base_skeleton["right" + joint_type] = offset_directions[
            "right" + joint_type
        ] * (
            (body_lengths["left" + joint_type] + body_lengths["right" + joint_type])
            / (2 * normalization)
        )

    _set_length("hip")
    _set_length("knee")
    _set_length("foot")
    _set_length("shoulder")
    _set_length("elbow")
    _set_length("wrist")
    base_skeleton["neck"] = offset_directions["neck"] * (
        body_lengths["neck"] / normalization
    )

    kpts["offset_directions"] = offset_directions
    kpts["base_skeleton"] = base_skeleton
    kpts["normalization"] = normalization

    return


# calculate the rotation of the root joint with respect to the world coordinates
def get_hips_position_and_rotation(
    frame_pos, root_joint="hips", root_define_joints=["lefthip", "neck"]
):
    # root position is saved directly
    root_position = frame_pos[root_joint]

    # calculate unit vectors of root joint
    root_u = frame_pos[root_define_joints[0]] - frame_pos[root_joint]
    root_u = root_u / np.sqrt(np.sum(np.square(root_u)))
    root_v = frame_pos[root_define_joints[1]] - frame_pos[root_joint]
    root_v = root_v / np.sqrt(np.sum(np.square(root_v)))
    root_w = np.cross(root_u, root_v)

    # Make the rotation matrix
    C = np.array([root_u, root_v, root_w]).T
    thetaz, thetay, thetax = utils.Decompose_R_ZXY(C)
    root_rotation = np.array([thetaz, thetax, thetay])

    return root_position, root_rotation


# calculate the rotation matrix and joint angles input joint
def get_joint_rotations(
    joint_name, joints_hierarchy, joints_offsets, frame_rotations, frame_pos
):
    _invR = np.eye(3)
    for i, parent_name in enumerate(joints_hierarchy[joint_name]):
        if i == 0:
            continue
        _r_angles = frame_rotations[parent_name]
        R = (
            utils.get_R_z(_r_angles[0])
            @ utils.get_R_x(_r_angles[1])
            @ utils.get_R_y(_r_angles[2])
        )
        _invR = _invR @ R.T

    b = _invR @ (frame_pos[joint_name] - frame_pos[joints_hierarchy[joint_name][0]])

    _R = utils.Get_R2(joints_offsets[joint_name], b)
    tz, ty, tx = utils.Decompose_R_ZXY(_R)
    joint_rs = np.array([tz, tx, ty])
    # print(np.degrees(joint_rs))

    return joint_rs


# helper function that composes a chain of rotation matrices
def get_rotation_chain(joint, hierarchy, frame_rotations):
    hierarchy = hierarchy[::-1]

    # this code assumes ZXY rotation order
    R = np.eye(3)
    for parent in hierarchy:
        angles = frame_rotations[parent]
        _R = (
            utils.get_R_z(angles[0])
            @ utils.get_R_x(angles[1])
            @ utils.get_R_y(angles[2])
        )
        R = R @ _R

    return R


# calculate the joint angles frame by frame.
def calculate_joint_angles(kpts):
    # set up emtpy container for joint angles
    for joint in kpts["joints"]:
        kpts[joint + "_angles"] = []

    for framenum in range(kpts["hips"].shape[0]):
        # get the keypoints positions in the current frame
        frame_pos = {}
        for joint in kpts["joints"]:
            frame_pos[joint] = kpts[joint][framenum]

        root_position, root_rotation = get_hips_position_and_rotation(frame_pos)

        frame_rotations = {"hips": root_rotation}

        # center the body pose
        for joint in kpts["joints"]:
            frame_pos[joint] = frame_pos[joint] - root_position

        # get the max joints connectsion
        max_connected_joints = 0
        for joint in kpts["joints"]:
            if len(kpts["hierarchy"][joint]) > max_connected_joints:
                max_connected_joints = len(kpts["hierarchy"][joint])

        depth = 2
        while depth <= max_connected_joints:
            for joint in kpts["joints"]:
                if len(kpts["hierarchy"][joint]) == depth:
                    joint_rs = get_joint_rotations(
                        joint,
                        kpts["hierarchy"],
                        kpts["offset_directions"],
                        frame_rotations,
                        frame_pos,
                    )
                    parent = kpts["hierarchy"][joint][0]
                    frame_rotations[parent] = joint_rs
            depth += 1

        # for completeness, add zero rotation angles for endpoints. This is not necessary as they are never used.
        for _j in kpts["joints"]:
            if _j not in list(frame_rotations.keys()):
                frame_rotations[_j] = np.array([0.0, 0.0, 0.0])

        # update dictionary with current angles.
        for joint in kpts["joints"]:
            kpts[joint + "_angles"].append(frame_rotations[joint])

    # convert joint angles list to numpy arrays.
    for joint in kpts["joints"]:
        kpts[joint + "_angles"] = np.array(kpts[joint + "_angles"])
        # print(joint, kpts[joint+'_angles'].shape)

    return


# draw the pose from original data
def draw_skeleton_from_joint_coordinates(kpts):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    connections = [
        ["hips", "lefthip"],
        ["lefthip", "leftknee"],
        ["leftknee", "leftfoot"],
        ["hips", "righthip"],
        ["righthip", "rightknee"],
        ["rightknee", "rightfoot"],
        ["hips", "neck"],
        ["neck", "leftshoulder"],
        ["leftshoulder", "leftelbow"],
        ["leftelbow", "leftwrist"],
        ["neck", "rightshoulder"],
        ["rightshoulder", "rightelbow"],
        ["rightelbow", "rightwrist"],
    ]

    for framenum in range(kpts["lefthip"].shape[0]):
        print(framenum)
        if framenum % 2 == 0:
            continue  # skip every 2nd frame

        for _j in kpts["joints"]:
            if _j == "hips":
                continue
            _p = kpts["hierarchy"][_j][0]  # get the name of the parent joint
            r1 = kpts[_p][framenum]
            r2 = kpts[_j][framenum]
            plt.plot(
                xs=[r1[0], r2[0]], ys=[r1[1], r2[1]], zs=[r1[2], r2[2]], color="blue"
            )

        # ax.set_axis_off()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        ax.set_xlim3d(-10, 10)
        ax.set_xlabel("x")
        ax.set_ylim3d(-10, 10)
        ax.set_ylabel("y")
        ax.set_zlim3d(-10, 10)
        ax.set_zlabel("z")
        plt.pause(5)
        ax.cla()
    plt.close()


# recalculate joint positions from calculated joint angles and draw
def draw_skeleton_from_joint_angles(kpts):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for framenum in range(kpts["hips"].shape[0]):
        # get a dictionary containing the rotations for the current frame
        frame_rotations = {}
        for joint in kpts["joints"]:
            frame_rotations[joint] = kpts[joint + "_angles"][framenum]

        # for plotting
        for _j in kpts["joints"]:
            if _j == "hips":
                continue

            # get hierarchy of how the joint connects back to root joint
            hierarchy = kpts["hierarchy"][_j]

            # get the current position of the parent joint
            r1 = kpts["hips"][framenum] / kpts["normalization"]
            for parent in hierarchy:
                if parent == "hips":
                    continue
                R = get_rotation_chain(
                    parent, kpts["hierarchy"][parent], frame_rotations
                )
                r1 = r1 + R @ kpts["base_skeleton"][parent]

            # get the current position of the joint. Note: r2 is the final position of the joint. r1 is simply calculated for plotting.
            r2 = (
                r1
                + get_rotation_chain(hierarchy[0], hierarchy, frame_rotations)
                @ kpts["base_skeleton"][_j]
            )
            plt.plot(
                xs=[r1[0], r2[0]], ys=[r1[1], r2[1]], zs=[r1[2], r2[2]], color="red"
            )

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.azim = 90
        ax.elev = -85
        ax.set_title("Pose from joint angles")
        ax.set_xlim3d(-4, 4)
        ax.set_xlabel("x")
        ax.set_ylim3d(-4, 4)
        ax.set_ylabel("y")
        ax.set_zlim3d(-4, 4)
        ax.set_zlabel("z")
        plt.pause(10)
        ax.cla()
    plt.close()


def get_angle_deviations(landmarks: list):
    global needed_mp_landmark_indices

    kpts = landmarks_to_kpts(landmarks)

    # if kpts.ndim != 3:
    #     kpts = keypoints[np.newaxis,:,:]

    # rotate to orient the pose better
    R = utils.get_R_z(np.pi / 2)
    for framenum in range(kpts.shape[0]):
        for kpt_num in range(kpts.shape[1]):
            kpts[framenum, kpt_num] = R @ kpts[framenum, kpt_num]

    kpts = convert_to_dictionary(kpts)
    add_hips_and_neck(kpts)
    # print("kpts :", kpts)
    filtered_kpts = kpts  # median_filter(kpts)
    # print("filtered_kpts :", filtered_kpts)
    get_bone_lengths(filtered_kpts)
    get_base_skeleton(filtered_kpts)

    calculate_joint_angles(filtered_kpts)
    # draw_skeleton_from_joint_angles(filtered_kpts)
    
    # for key in needed_landmarks.keys():
    #     print(filtered_kpts[mediapipe_landmark_names_map[key] + "_angles"][0])

    angles = [np.array([0, 0, 0]) for _ in range(33)]
    for i in range(33):
        keypoint_name = mediapipe_landmark_names_map.get(
            mp.solutions.pose.PoseLandmark(i).name, None
        )
        if keypoint_name:
            angles[i] = filtered_kpts[keypoint_name + "_angles"][0]
    return angles


def calc_example_angles():
    filename = "vendor/join_angle_calculation/kpts_3d.dat"  # sys.argv[1]
    kpts = read_keypoints(filename)
    print(np.shape(kpts))

    # rotate to orient the pose better
    R = utils.get_R_z(np.pi / 2)
    for framenum in range(kpts.shape[0]):
        for kpt_num in range(kpts.shape[1]):
            kpts[framenum, kpt_num] = R @ kpts[framenum, kpt_num]

    kpts = convert_to_dictionary(kpts)
    add_hips_and_neck(kpts)
    print("kpts :", kpts)
    filtered_kpts = kpts  # median_filter(kpts)
    print("filtered_kpts :", filtered_kpts)
    get_bone_lengths(filtered_kpts)
    get_base_skeleton(filtered_kpts)

    calculate_joint_angles(filtered_kpts)
    # draw_skeleton_from_joint_coordinates(filtered_kpts)
    draw_skeleton_from_joint_angles(filtered_kpts)
    for key in needed_landmarks.keys():
        print(filtered_kpts[mediapipe_landmark_names_map[key] + "_angles"][0])


if __name__ == "__main__":
    ideal_landmarks = [
        np.array([143.73686379, 106.64321381, -118.98162383]),
        np.array([151.67757928, 95.94112271, -94.31096902]),
        np.array([155.61486709, 96.53626353, -94.51458403]),
        np.array([161.20302039, 97.06428748, -94.44981048]),
        np.array([136.99421456, 94.93629462, -92.11133316]),
        np.array([132.98771688, 95.11934221, -92.36583927]),
        np.array([129.48359382, 95.65558201, -92.52392724]),
        np.array([164.92247337, 102.4761588, 33.3375616]),
        np.array([123.14177316, 100.63422346, 42.92056304]),
        np.array([153.13434339, 116.90352058, -69.12819827]),
        np.array([136.49337494, 116.46072042, -68.54784235]),
        np.array([202.20917052, 167.90898108, 85.01194018]),
        np.array([89.65446699, 171.84392512, 103.63114205]),
        np.array([244.33146691, 240.83276457, -31.3092615]),
        np.array([54.43311021, 249.55312622, -36.58009605]),
        np.array([164.88345742, 221.21436125, -234.9220953]),
        np.array([135.4505122, 217.55800742, -264.73584664]),
        np.array([151.55946875, 201.09956765, -281.54922408]),
        np.array([140.53550652, 197.1583147, -318.17035604]),
        np.array([153.39623773, 194.74706233, -268.39659858]),
        np.array([141.92497528, 191.07645106, -299.97637403]),
        np.array([155.90400344, 202.34812653, -232.42249376]),
        np.array([142.96275228, 199.53235519, -263.59810919]),
        np.array([177.18692517, 330.08618975, 2.31146962]),
        np.array([108.17027575, 330.19825315, -2.51229954]),
        np.array([166.92942476, 465.86726189, -13.37592138]),
        np.array([114.44653004, 466.41104031, -28.72384132]),
        np.array([163.20486075, 576.21557212, 197.78414118]),
        np.array([117.12939522, 577.27557647, 147.40326655]),
        np.array([158.32305974, 588.71684408, 211.49430847]),
        np.array([122.16791788, 589.24675715, 159.52594185]),
        np.array([161.66291267, 619.78735065, 45.59251876]),
        np.array([123.13942006, 620.72477674, -9.09259264]),
    ]
    # print(landmarks)
    # kpts = landmarks_to_kpts(landmarks)
    # print(kpts)
    import numpy as np
    ideal_angles = get_angle_deviations(ideal_landmarks)
    i = 0
    for angle in ideal_angles:
        print(angle, np.linalg.norm(angle), mp.solutions.pose.PoseLandmark(i).name)
        i += 1

    my_landmarks = [
        [518.1345851123333, 361.7545419037342, -721.1206722259521],
        [525.1857960820198, 330.5317285656929, -690.0183438658714],
        [537.4582443535328, 326.1027023792267, -690.8573042750359],
        [545.3601765334606, 322.52295757830143, -690.6868599057198],
        [498.82453629374504, 344.32356759905815, -670.1895634531975],
        [490.23590594530106, 350.86841148138046, -671.1691648960114],
        [482.7797674536705, 357.1609453558922, -671.6552314758301],
        [559.9777704179287, 328.6468712091446, -468.86010906100273],
        [479.19254863262177, 375.16625383496284, -367.61929455399513],
        [548.7409008145332, 378.78357124328613, -626.142048060894],
        [518.2067203521729, 391.2405175715685, -603.2130925655365],
        [676.5358592271805, 419.50811073184013, -411.2072313427925],
        [491.9283794462681, 500.0430399775505, -181.4697713404894],
        [564.0351420938969, 241.13860648870468, -577.2764430642128],
        [408.0545339882374, 320.5461936444044, -285.8633704483509],
        [409.1245574951172, 108.29712986946106, -616.2808431982994],
        [360.42733785510063, 143.3809090256691, -554.2517609298229],
        [369.6570540070534, 82.00796273350716, -700.9933698773384],
        [339.5870587527752, 97.55534380674362, -655.1539551019669],
        [363.81811290979385, 81.17845671251416, -694.5430067181587],
        [338.21125441789627, 89.07868266105652, -673.8660526871681],
        [375.4521974623203, 94.12408688664436, -618.2925334572792],
        [350.5855016708374, 106.22836571931839, -581.8830127418041],
        [725.8644977807999, 837.3280994296074, -52.978343173861504],
        [589.7537107467651, 835.5649726390839, 52.56902716308832],
        [682.9557210803032, 1146.706988453865, -4.242753429571167],
        [602.9121277332306, 1144.9053760766983, 30.598164716735482],
        [654.5659848451614, 1403.9254536032677, 433.255592495203],
        [607.0612825155258, 1420.2737749814987, 340.2444072663784],
        [647.5916553139687, 1437.8804158568382, 456.47005277872086],
        [613.6146661043167, 1452.3218747377396, 355.54287126660347],
        [666.2721348404884, 1502.10920804739, 81.6930805966258],
        [621.7734513878822, 1518.5699897408485, -38.35371723771095],
    ]

    my_angles = get_angle_deviations(my_landmarks)
    i = 0
    for angle in my_angles:
        print(angle, np.linalg.norm(angle), mp.solutions.pose.PoseLandmark(i).name)
        i += 1

    deviations = [ideal_angles[i] - my_angles[i] for i in range(33)]
    for i in range(33):
        print(deviations[i], np.linalg.norm(deviations[i]),np.linalg.norm(deviations[i])*180/np.pi, mp.solutions.pose.PoseLandmark(i).name)

    # else:
    #     angles[i] = np.array([0,0,0])

    # for key in needed_landmarks.keys():
    #     print(filtered_kpts[mediapipe_landmark_names_map[key] + "_angles"][0])
