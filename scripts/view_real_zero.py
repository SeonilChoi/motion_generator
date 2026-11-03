"""ARTI-D1: URDF zero vs the real robot's servo zero ("real zero"), side by side in Meshcat.

Real zero (user, 2026-11-03): the hip link (roll_to_pitch) is level, the leg is stretched out straight and points
backward at 90 deg to the upper body (horizontal), and the sole is perpendicular to the ground. In URDF joint angles:
  hip_roll   = +/-0.15 rad   (the URDF hip_roll frame is rolled 0.15 rad inward; this cancels it)
  hip_pitch  = HIP_PITCH     (thigh line hip_pitch -> knee horizontal, pointing backward)
  knee       = KNEE          (shin line knee -> ankle in line with the thigh)
  ankle      = ANKLE         (foot pitched 90 deg: sole vertical, facing backward; the foot link z axis is the sole
                              normal, flat on the ground at the URDF stance pose hip_pitch + knee + ankle = 0)
They come from the URDF link geometry (computed below), so they follow URDF edits.

    python scripts/view_real_zero.py                      # Meshcat (left: URDF zero, right: real zero)
    python scripts/view_real_zero.py --png out.png        # also a front / side render (matplotlib)
    python scripts/view_real_zero.py --leg_deg 90 --foot_deg 90   # leg angle from straight down / foot pitch
"""
import argparse
import math
import os
import struct

import numpy as np
import placo

HERE = os.path.dirname(os.path.abspath(__file__))
URDF = os.path.join(HERE, "..", "robots", "arti_d1", "arti_d1.urdf")


def sagittal_angle(v):
    """Angle of v from straight down, + = pointing backward (-x), in the x-z plane."""
    return math.atan2(-v[0], -v[2])


def real_zero(robot, leg_deg=90.0, foot_deg=90.0):
    """URDF joint angles of the real servo zero."""
    m = robot.model
    origin = lambda j: m.jointPlacements[m.getJointId(j)].translation
    thigh = origin("left_knee")             # hip_pitch -> knee, in the thigh frame
    shin = origin("left_ankle")             # knee -> ankle, in the shin frame
    # rotation about +y by a turns a "down" vector backward by a (R_y(a) (0,0,-1) = (-sin a, 0, -cos a))
    hip_pitch = math.radians(leg_deg) - sagittal_angle(thigh)
    knee = sagittal_angle(thigh) - sagittal_angle(shin)
    ankle = math.radians(foot_deg) - hip_pitch - knee
    roll = 0.15
    q = {}
    for side, s in (("left", 1.0), ("right", -1.0)):
        q[f"{side}_hip_roll"] = s * roll
        q[f"{side}_hip_pitch"] = hip_pitch
        q[f"{side}_knee"] = knee
        q[f"{side}_ankle"] = ankle
    return q


def set_pose(robot, q, y):
    for name in robot.joint_names():
        robot.set_joint(name, q.get(name, 0.0))
    T = np.eye(4)
    T[1, 3] = y
    T[2, 3] = 0.45
    robot.set_T_world_fbase(T)
    robot.update_kinematics()


def read_stl(path):
    with open(path, "rb") as f:
        data = f.read()
    n = struct.unpack("<I", data[80:84])[0]
    if 84 + n * 50 == len(data):
        a = np.frombuffer(data[84:], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]), count=n)
        return a["v"].astype(np.float64)
    tris, cur = [], []
    for line in data.decode(errors="ignore").splitlines():
        p = line.split()
        if p and p[0] == "vertex":
            cur.append([float(x) for x in p[1:4]])
            if len(cur) == 3:
                tris.append(cur)
                cur = []
    return np.array(tris)


def render_png(robots, out):
    import pinocchio as pin
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(16, 8))
    views = [("front", 0, 0), ("side (forward = right)", 0, -90), ("3/4", 15, -60)]
    for k, (title, elev, azim) in enumerate(views):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        for robot, color in robots:
            data = robot.model.createData()
            gdata = pin.GeometryData(robot.visual_model)
            pin.forwardKinematics(robot.model, data, robot.state.q)
            pin.updateGeometryPlacements(robot.model, data, robot.visual_model, gdata)
            for i, g in enumerate(robot.visual_model.geometryObjects):
                tri = read_stl(g.meshPath) * np.asarray(g.meshScale)
                M = gdata.oMg[i]
                tri = tri @ M.rotation.T + M.translation
                ax.add_collection3d(Poly3DCollection(tri[::2], facecolor=color, edgecolor="none", alpha=0.35))
        ax.set_xlim(-0.45, 0.25); ax.set_ylim(-0.25, 0.65); ax.set_zlim(0.0, 0.9)
        ax.set_box_aspect((0.7, 0.9, 0.9))
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title + "  | grey: URDF zero, orange: real zero")
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    plt.tight_layout()
    plt.savefig(out, dpi=90)
    print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leg_deg", type=float, default=90.0, help="straight leg angle from down, + = backward (deg)")
    ap.add_argument("--foot_deg", type=float, default=90.0, help="foot pitch, + = toes up / sole facing back (deg)")
    ap.add_argument("--png", type=str, default=None, help="save a side / front render instead of / besides Meshcat")
    ap.add_argument("--no-viewer", action="store_true")
    args = ap.parse_args()

    urdf_zero = placo.RobotWrapper(URDF)
    real = placo.RobotWrapper(URDF)
    q = real_zero(real, args.leg_deg, args.foot_deg)
    set_pose(urdf_zero, {}, 0.0)
    set_pose(real, q, 0.4)

    print("real zero in URDF joint angles (rad / deg); offset = URDF angle at the real zero:")
    for name in sorted(q):
        print(f"  {name:18s} {q[name]:+.4f} rad  {math.degrees(q[name]):+7.2f} deg")

    if args.png:
        render_png([(urdf_zero, "grey"), (real, "orange")], args.png)
    if args.no_viewer:
        return

    from ischedule import run_loop, schedule
    from placo_utils.visualization import robot_viz

    viz_a = robot_viz(urdf_zero, "urdf_zero")
    viz_b = robot_viz(real, "real_zero")
    print("Meshcat: open http://127.0.0.1:7000/static/  (left: URDF zero, right y+0.4: real zero)")

    @schedule(interval=0.1)
    def loop():
        viz_a.display(urdf_zero.state.q)
        viz_b.display(real.state.q)

    run_loop()


if __name__ == "__main__":
    main()
