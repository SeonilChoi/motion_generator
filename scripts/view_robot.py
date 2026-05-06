import os
import placo
import argparse

from ischedule import schedule, run_loop
from placo_utils.visualization import robot_viz


def main(args):
    urdf_file = f"../robots/{args.robot}/{args.robot}.urdf"
    urdf_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), urdf_file)

    robot = placo.RobotWrapper(urdf_file_path)
    
    viz = robot_viz(robot)

    @schedule(interval=0.01)
    def loop():

        robot.update_kinematics()

        viz.display(robot.state.q)

    run_loop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--robot", type=str, required=True, choices=["bdx", "olaf"])
    
    args = parser.parse_args()

    main(args)