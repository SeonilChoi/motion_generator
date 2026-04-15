import os
import json
import argparse
import subprocess
import numpy as np

from concurrent.futures import ThreadPoolExecutor


def main(args):
    # Load motion config
    motion_config_file = f"../config/{args.robot}/motion.json"
    motion_config_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), motion_config_file)

    if not os.path.isfile(motion_config_file_path):
        raise FileNotFoundError(f"Motion config file not found: {motion_config_file_path}")

    with open(motion_config_file_path, "r") as f:
        motion_config = json.load(f)

    dx_bounds = motion_config["dx_bounds"]
    dy_bounds = motion_config["dy_bounds"]
    dth_bounds = motion_config["dth_bounds"]

    # Load and run source file
    source_file = f"../src/motion_generator/gait_generator.py"
    source_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), source_file)

    num_references = args.num

    commands = []
    for i in range(num_references):
        dx = round(
            np.random.uniform(dx_bounds[0], dx_bounds[1]) * np.random.choice([-1, 1]),
            2
        )
        dy = round(
            np.random.uniform(dy_bounds[0], dy_bounds[1]) * np.random.choice([-1, 1]),
            2
        )
        dth = round(
            np.random.uniform(dth_bounds[0], dth_bounds[1]) * np.random.choice([-1, 1]),
            2
        )

        cmd = [
            "python",
            source_file_path,
            "--index", str(i),
            "--robot", args.robot,
            "--dx", str(dx),
            "--dy", str(dy),
            "--dth", str(dth),
            "--stand", str(args.stand),
            "--duration", str(args.duration)
        ]

        commands.append(cmd)

    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        executor.map(subprocess.run, commands)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--robot", type=str, required=True, choices=["bdx"])

    parser.add_argument("--num", type=int, required=True)

    parser.add_argument("--jobs", type=int, required=True)

    parser.add_argument("--stand", type=bool, required=True)

    parser.add_argument("--duration", type=int, required=True)

    args = parser.parse_args()

    main(args)