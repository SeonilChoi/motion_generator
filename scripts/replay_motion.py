import os
import time
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import FramesViewer.utils as fv_utils

from scipy.spatial.transform import Rotation as R


def main(args):
    fv = None
    if not args.no_viewer:
        from FramesViewer.viewer import Viewer

        fv = Viewer()
        fv.start()

    file_name = f"../data/{args.robot}/{args.index}.json"
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)

    with open(file_path, "r") as f:
        episode = json.load(f)

    frame_duration = episode["frame_duration"]
    frame_offset = episode["frame_offset"][0] 
    frame_size = episode["frame_size"]
    frames = episode["frames"]

    root_position_slice = slice(frame_offset["root_position"], frame_offset["root_orientation_quat"])
    root_orientation_quat_slice = slice(frame_offset["root_orientation_quat"], frame_offset["joints_positions"])
    left_toe_position_slice = slice(frame_offset["left_toe_position"], frame_offset["right_toe_position"])
    right_toe_position_slice = slice(frame_offset["right_toe_position"], frame_offset["world_linear_velocity"])
    linear_velocity_slice = slice(frame_offset["world_linear_velocity"], frame_offset["world_angular_velocity"])
    angular_velocity_slice = slice(frame_offset["world_angular_velocity"], frame_offset["joints_velocities"])
    joints_velocities_slice = slice(frame_offset["joints_velocities"], frame_offset["left_toe_linear_velocity"])

    pose = np.eye(4)
    
    velocities = {}
    velocities["linear"] = []
    velocities["angular"] = []
    velocities["joints"] = []

    for i, frame in enumerate(frames):
        pose[:3, 3] = frame[root_position_slice]
        pose[:3, :3] = R.from_quat(frame[root_orientation_quat_slice]).as_matrix()

        if fv is not None:
            fv.push_frame(pose, "aze")

        velocities["linear"].append(frame[linear_velocity_slice])
        velocities["angular"].append(frame[angular_velocity_slice])
        velocities["joints"].append(frame[joints_velocities_slice])

        if fv is not None:
            fv.push_frame(
                fv_utils.make_pose(np.array(frame[left_toe_position_slice]), [0.0, 0.0, 0.0]),
                "left_toe",
            )
            fv.push_frame(
                fv_utils.make_pose(np.array(frame[right_toe_position_slice]), [0.0, 0.0, 0.0]),
                "right_toe",
            )
            time.sleep(frame_duration)

    x_linear_velocity = np.array(velocities["linear"])[:, 0]
    y_linear_velocity = np.array(velocities["linear"])[:, 1]
    z_linear_velocity = np.array(velocities["linear"])[:, 2]

    x_angular_velocity = np.array(velocities["angular"])[:, 0]
    y_angular_velocity = np.array(velocities["angular"])[:, 1]
    z_angular_velocity = np.array(velocities["angular"])[:, 2]

    joints_velocities = np.array(velocities["joints"])

    plt.plot(x_linear_velocity, label="x_linear_velocity")
    plt.plot(y_linear_velocity, label="y_linear_velocity")
    plt.plot(z_linear_velocity, label="z_linear_velocity")
    plt.plot(x_angular_velocity, label="x_angular_velocity")
    plt.plot(y_angular_velocity, label="y_angular_velocity")
    plt.plot(z_angular_velocity, label="z_angular_velocity")
    plt.plot(joints_velocities, label="joints_velocities")
    
    plt.legend()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--robot", type=str, required=True, choices=["bdx"])

    parser.add_argument("--index", type=int, required=True)

    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Skip FramesViewer (OpenGL/GLUT). Use when glutInit fails or there is no display.",
    )

    args = parser.parse_args()

    main(args)
