import csv
import cv2
import mediapipe as mp
import numpy as np
import os

pose = mp.solutions.pose.Pose(
    static_image_mode=True, min_detection_confidence=0.3, model_complexity=2
)

#functions to write and append rows to our csv file
def write_row(row, mode="w"):
    with open(
        "C:\\Users\\cosmo\\Projects\\YogaPoseHelper\\app\\data\\yogaposes.csv",
        mode=mode,
        newline="",
    ) as f:
        writer = csv.writer(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(row)


def append_row(row):
    write_row(row, mode="a")


def extract_coords_as_row(frame):
    """
    returns pose coordinates in the form : [x0,y0,z0,x1,y1,z1,x2,y2,z2,x3,y3....]
    where x0,y0,z0,x1,y1,z1,x2,y2,z2,x3,y3 are the coordinates of landmarks 0,1,2,3...
    """
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose.process(img)

    if results.pose_landmarks:
        try:
            landmarks = results.pose_landmarks.landmark
            row = list(
                np.array(
                    [[landmark.x, landmark.y, landmark.z] for landmark in landmarks]
                ).flatten()
            )

            return row

        except Exception as e:
            print(e)
            print("Error while exporting coords")

    return None


def collect_coords_to_csv(frame, class_name: str):
    """
    returns pose coordinates in the form : ["<class_name>",x0,y0,z0,x1,y1,z1,x2,y2,z2,x3,y3....]
    """
    row = extract_coords_as_row(frame)
    if row:
        append_row([class_name] + row)


if __name__ == "__main__":
    # to initialize yogaposes.csv
    num_coords = 33

    columns = ["class"]
    for i in range(num_coords):
        columns += [f"x{i}", f"y{i}", f"z{i}"]

    write_row(columns)

    # each directory in kaggle\\DATASET\\TRAIN has its name=the pose's name, and images of that pose are stored under it
    # following code loops through each image file, extracts landmarks of the pose in the image, and stores the coords in a csv
    dataset_path = (
        "C:\\Users\\cosmo\\Projects\\YogaPoseHelper\\app\\data\\kaggle\\DATASET\\TRAIN"
    )
    for pose_name in os.listdir(dataset_path):
        pose_dir_path = os.path.join(dataset_path, pose_name)
        print(pose_dir_path)
        for filename in os.listdir(pose_dir_path):
            file_path = os.path.join(pose_dir_path, filename)
            print(file_path)
            collect_coords_to_csv(
                cv2.imread(file_path),
                pose_name,
            )
