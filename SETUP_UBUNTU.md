# Ubuntu setup log

Setup of `motion_generator` on Ubuntu after previous use on Windows (conda `env_placo`).

- Machine: Ubuntu 24.04.5 LTS, RTX 5070 Ti, NVIDIA driver 595.99.02, Python 3.12.16, uv 0.13.6
- Location: `/opt/isaac/projects/motion_generator`, branch `ubuntu` (not pushed; upstream unset)
- Date: 2026-10-30

## Quick start

```bash
cd /opt/isaac/projects/motion_generator
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-ubuntu.lock.txt
uv pip install --python .venv/bin/python -e .
source .venv/bin/activate
```

Test run (writes to `out/test/<robot>/`, git-ignored):

```bash
scripts/generate_motions.sh arti_h1 4 4 out/test
scripts/generate_motions.sh arti_d1 4 4 out/test
```

Regenerate the clip sets in the sibling Isaac Lab projects. Both commands use the sibling `gen_motions.py`, which
produced the existing sets (distinct grid commands, fixed seed). It writes to `data/<robot>/` here; copy the result
over afterwards:

```bash
cd /opt/isaac/projects/motion_generator
MOTION_GENERATOR=$PWD .venv/bin/python ../arti_h1_local/scripts/tuning/gen_motions.py arti_h1 1000 16
MOTION_GENERATOR=$PWD .venv/bin/python ../arti_d1_local/scripts/tuning/gen_motions.py arti_d1 1000 16
```

Or generate random commands straight into the project folders (`scripts/main.py` sampling):

```bash
OVERWRITE=1 scripts/generate_motions.sh arti_h1 1000 16   # -> ../arti_h1_local/source/arti_h1/arti_h1/assets/motions/arti_h1/
OVERWRITE=1 scripts/generate_motions.sh arti_d1 1000 16   # -> ../arti_d1_local/source/arti_d1/arti_d1/assets/motions/arti_d1/
```

Targets can be overridden with `ARTI_H1_MOTIONS_DIR` / `ARTI_D1_MOTIONS_DIR` or a 4th argument (motions root).

The nominal gait tables (`arti_*_nominal_gait_command.npy`) are **not** produced by this repo. They come from
`scripts/tuning/ilc_nominal.py` in each Isaac Lab project, which runs in the Isaac Lab env. The stand clip (`1000.json`)
comes from `scripts/tuning/make_stand_clip.py` there.

## Log

### 1. Clone
- The repo was already cloned at the target location over HTTPS (`main` @ `6d60787`, clean).
- SSH to GitHub fails (`Permission denied (publickey)`; no key in `~/.ssh`). HTTPS fetch also fails (no
  credentials). The SSH key was declined for now; work continued from the local `origin/develop` ref.
- Git LFS: not used (no `.gitattributes`; STL/USD are plain blobs).

### 2. Inspection
- `main` only supports `bdx`/`olaf`, and its output has no `gait_phase` / `phase_rate` / `periodic_valid`. The H1/D1
  code (configs, URDFs, newer generator and engine) is only on `origin/develop` @ `f2a59fa`
  ("WIP: ARTI-D1/H1 configs ... (pre-Ubuntu move)"). Branch `ubuntu` was created from `origin/develop`
  (approved by the owner).
- Pure-Python, CPU-only PlaCo gait generator. No torch / CUDA dependency, so no cu128 / sm_120 concerns.
- Output: one JSON episode per `(dx, dy, dth)` command, `<out>/<index>.json`, 60 fps, 15 s (900 frames).

### 3. Windows-specific issues
| Check | Result |
|---|---|
| CRLF line endings | none found |
| Exec bits / .bat / .ps1 | no shell scripts existed. Added `scripts/generate_motions.sh` (+x) |
| Hard-coded Windows paths | none in this repo. (Siblings' `gen_motions.py` default `C:/motion_generator`; set `MOTION_GENERATOR`) |
| Filename case (URDF mesh refs vs files) | all `arti_h1` (`.STL`) and `arti_d1` (`.stl`) refs resolve exactly |
| Bare `"python"` in `main.py` subprocess | Ubuntu has no `python` on PATH. Fixed: `sys.executable` |
| Output dir hard-coded, not created | Fixed: optional `--out_dir`, `os.makedirs` (default unchanged) |
| Windows-only conda pin `coal=3.0.2=h1ea47a8_1` | left untouched; Ubuntu uses pip wheels (`requirements-ubuntu.lock.txt`) |
| `LD_LIBRARY_PATH` / `cmeel.prefix` workaround (README) | not needed with uv + pip wheels |

### 4. Environment
- `uv venv --python 3.12 .venv`, then `uv pip install placo numpy scipy matplotlib ischedule FramesViewer` and
  `uv pip install -e .`
- Key versions: placo 0.10.2, pin 3.8.0, coal 3.0.2, eigenpy 3.12.0, numpy 2.3.5, scipy 1.18.1, meshcat 0.3.2.
  The full pin list is in `requirements-ubuntu.lock.txt`.
- No apt packages were installed. `freeglut3-dev` is only needed for the optional FramesViewer window in
  `replay_motion.py`; use `--no-viewer` otherwise.

### 5. Output paths
- `gait_generator.py` / `main.py`: new `--out_dir` (default `data/<robot>/`, as before).
- `scripts/generate_motions.sh`: defaults to the sibling project motion folders, with an overwrite guard
  (`OVERWRITE=1`).

### 6. Smoke test (out/test/)
- 4 random-command clips per robot via `generate_motions.sh` (~3-5 s per batch, 4 jobs). Top-level keys and order,
  `fps` 60, `frame_duration` 0.0167, `joints`, `frame_offset`, `frame_size`, and `gait_parameters` keys all match the
  existing project clips. Frames are (900, 54) for H1 and (900, 58) for D1.
- Same-command regeneration of shipped clips 4, 565, 999 (both robots): **bit-identical** (max |diff| = 0.0,
  identical `gait_parameters`). The Ubuntu/pip PlaCo build reproduces the Windows output.
- Sibling `gen_motions.py` command list vs shipped sets: D1 1000/1000 match (`1000.json` = stand clip). H1 matches
  if `(0,0,0)` is kept at index 202. The shipped H1 set was made with an older `gen_motions.py` that kept it, and
  `202.json` was then deleted. The script then dropped `(0,0,0)`, so a full H1 regeneration had the same 999
  commands but indices 203-999 shifted down by one. **Fixed** in `arti_h1_local/scripts/tuning/gen_motions.py`
  (commit `5ad9865` on `isaaclab3`): the `(0,0,0)` slot keeps its index and is skipped. The fixed command list matches every
  shipped H1 clip (indices 0-999 minus 202), and a 3-clip run is identical to the shipped `0-2.json`.
- D1 joint order in the JSON is the PlaCo/URDF order (legs L, legs R, arms), not Dynamixel ID 1-14. This is the
  same as the shipped clips, and `MotionLoader` remaps by name to `robot.data.joint_names`, which `arti_d1_env.py`
  checks is the Dynamixel order. No change needed.

## Open issues
- `../arti_d1_local/README.md` contains unresolved git merge-conflict markers (not touched).
- `config/arti_d1/` has several variants (`gait.json`, `gait_15mm_*`, `gait_22mm_*`, `gait_fast`). The shipped D1 set
  matches `gait.json` (same-command regeneration is identical).
- GitHub SSH key not set up; nothing pushed.
