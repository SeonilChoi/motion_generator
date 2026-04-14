import os
import time
import json
import argparse
import threading
import webbrowser

from placo_utils.visualization import robot_viz, robot_frame_viz, footsteps_viz
from scipy.spatial.transform import Rotation as R

from motion_generator.motion_engine import MotionEngine


def open_browser():
    webbrowser.open("http://127.0.0.1:7000/static/")


def main(args):
    ISAACSIM_FPS = 30
    MESHCAT_FPS = 20

    # Load gait config
    gait_config_file = f"../../config/{args.robot}/gait.json"
    gait_config_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), gait_config_file)

    if not os.path.isfile(gait_config_file_path):
        raise FileNotFoundError(f"Gait config file not found: {gait_config_file_path}")

    with open(gait_config_file_path, "r") as f:
        gait_parameters = json.load(f)

    gait_parameters["dx"] = args.dx
    gait_parameters["dy"] = args.dy
    gait_parameters["dth"] = args.dth

    # Set robot path
    robot_folder = f"../../robots/{args.robot}"
    robot_folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), robot_folder)

    # Run motion engine
    motion_engine = MotionEngine(robot_folder_path, gait_parameters)

    first_joints_positions = list(motion_engine.get_angles().values())
    first_T_world_fbase = motion_engine.robot.get_T_world_fbase()
    first_T_world_left_foot = motion_engine.robot.get_T_world_left()
    first_T_world_right_foot = motion_engine.robot.get_T_world_right()

    motion_engine.set_trajectory(gait_parameters["dx"], gait_parameters["dy"], gait_parameters["dth"])

    viz = robot_viz(motion_engine.robot)
    threading.Timer(1.0, open_browser).start()

    dt = 0.001
    start = time.time()

    skip_warmup = 0.0 # TODO

    last_record = 0.0
    last_meshcat_display = 0.0
    
    prev_root_position = [0.0, 0.0, 0.0]
    prev_root_orientation_quat = None
    prev_root_orientation_euler = [0.0, 0.0, 0.0]
    prev_left_toe_position = [0.0, 0.0, 0.0]
    prev_right_toe_position = [0.0, 0.0, 0.0]
    prev_joints_positions = None

    i = 0

    is_prev_initialized = False
    is_added = False

    average_x_linear_velocity = []
    average_y_linear_velocity = []
    average_z_angular_velocity = []

    while True:
        motion_engine.tick(dt)
        if motion_engine.t <= 0 + skip_warmup:
            start = motion_engine.t
            last_record = motion_engine.t + 1 / ISAACSIM_FPS
            last_meshcat_display = motion_engine.t + 1 / MESHCAT_FPS
            continue

        if motion_engine.t - last_record >= 1 / ISAACSIM_FPS:
            if args.stand:
                T_world_fbase = first_T_world_fbase
            else:
                T_world_fbase = motion_engine.robot.get_T_world_fbase()
            
            root_position = list(T_world_fbase[:3, 3])
            root_orientation_quat = list(R.from_matrix(T_world_fbase[:3, :3]).as_quat())

            if args.stand:
                joints_positions = first_joints_positions
                T_world_left_foot = first_T_world_left_foot
                T_world_right_foot = first_T_world_right_foot
            else:
                joints_positions = list(motion_engine.get_angles().values())
                T_world_left_foot = motion_engine.robot.get_T_world_left()
                T_world_right_foot = motion_engine.robot.get_T_world_right()

            T_body_left_foot = np.linalg.inv(T_world_fbase) @ T_world_left_foot
            T_body_right_foot = np.linalg.inv(T_world_fbase) @ T_world_right_foot

            left_toe_position = list(T_body_left_foot[:3, 3])
            right_toe_position = list(T_body_right_foot[:3, 3])

            if not is_prev_initialized:
                #TODO

        if motion_engine.t - last_meshcat_display >= 1 / MESHCAT_FPS:
            last_meshcat_display = motion_engine.t
            viz.display(motion_engine.robot.state.q)

            robot_frame_viz(motion_engine.robot, "trunk")
            robot_frame_viz(motion_engine.robot, "left_foot")
            robot_frame_viz(motion_engine.robot, "right_foot")

            footsteps_viz(motion_engine.get_supports())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--index", type=int, required=True)

    parser.add_argument("--robot", type=str, required=True, choices=["bdx"])

    parser.add_argument("--dx", type=float, required=True)

    parser.add_argument("--dy", type=float, required=True)

    parser.add_argument("--dth", type=float, required=True)

    parser.add_argument("--stand", type=bool, required=True)

    args = parser.parse_args()

    main(args)
