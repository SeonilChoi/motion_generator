"""Open Duck style command sweep (auto_waddle --sweep): one clip per (dx, dy, dth) grid point.

    .venv/bin/python scripts/sweep_motions.py --robot arti_d1 --gait_config gait_duck.json --out_dir <dir> --jobs 8 \
        --dx -0.03 0.04 0.01 --dy -0.02 0.02 0.01 --dth -0.2 0.2 0.1
Grids are (min, max, step), per-step placo values. Writes <out_dir>/<index>.json and <out_dir>/sweep.json (index -> cmd).
"""
import argparse, json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--robot", required=True)
p.add_argument("--gait_config", default="gait.json")
p.add_argument("--out_dir", required=True)
p.add_argument("--jobs", type=int, default=8)
p.add_argument("--dx", type=float, nargs=3, required=True)
p.add_argument("--dy", type=float, nargs=3, required=True)
p.add_argument("--dth", type=float, nargs=3, required=True)
a = p.parse_args()
grid = lambda r: np.round(np.arange(r[0], r[1] + r[2] / 2, r[2]), 4)
gen = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../src/motion_generator/gait_generator.py")
os.makedirs(a.out_dir, exist_ok=True)
cmds, table = [], {}
for i, (dx, dy, dth) in enumerate((x, y, t) for x in grid(a.dx) for y in grid(a.dy) for t in grid(a.dth)):
    table[i] = [float(dx), float(dy), float(dth)]
    cmds.append([sys.executable, gen, "--index", str(i), "--robot", a.robot, "--dx", str(dx), "--dy", str(dy),
                 "--dth", str(dth), "--stand", "false", "--visual", "false", "--gait_config", a.gait_config,
                 "--out_dir", os.path.abspath(a.out_dir)])
json.dump({"gait_config": a.gait_config, "grid": {"dx": a.dx, "dy": a.dy, "dth": a.dth}, "clips": table},
          open(os.path.join(a.out_dir, "sweep.json"), "w"), indent=1)
print(f"{len(cmds)} clips -> {a.out_dir}")
with ThreadPoolExecutor(max_workers=a.jobs) as ex:
    list(ex.map(lambda c: subprocess.run(c, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL), cmds))
