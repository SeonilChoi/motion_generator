# Motion Generator

Small toolkit for **humanoid gait / motion generation** with [PlaCo](https://github.com/Rhoban/placo): it plans walking patterns, records time-series episodes (poses, velocities, contacts) for simulation or datasets, and provides helpers to visualize the URDF or replay saved JSON.

## Requirements

### Python

- **Python** 3.9+
- Install the local package (declared deps include **numpy**):

  ```bash
  pip install -e .
  ```

- **Runtime stack** (not all are listed in `pyproject.toml`; use your PlaCo environment as needed):
  - **placo**, **placo_utils** (Meshcat / IK / walk pipeline)
  - **scipy** (rotations, used by gait export and replay)
  - **matplotlib** (replay velocity plots)
  - **ischedule** (used by `view_robot.py`)
  - **FramesViewer** (optional 3D replay in `replay_motion.py` when the viewer is enabled)

### OpenGL / GLUT (optional, for `replay_motion.py` 3D viewer)

If you see `glutInit` / `NullFunctionError`, install native GLUT (Debian/Ubuntu example):

```bash
sudo apt install freeglut3-dev
```

Or run replay **without** the OpenGL window using `--no-viewer` (see below).

---

## Scripts

Run these from the **repository root** (`motion_generator/`) unless you adjust paths.

### `scripts/main.py` — batch gait generation

Spawns multiple runs of `src/motion_generator/gait_generator.py` with random `(dx, dy, dθ)` from `config/<robot>/motion.json`, in parallel.

```bash
python scripts/main.py --robot bdx --num <N> --jobs <workers> --stand <true|false> --duration <seconds>
```

Example:

```bash
python scripts/main.py --robot bdx --num 4 --jobs 2 --stand false --duration 10
```

Episode JSON files are written under `data/<robot>/` (e.g. `data/bdx/0.json`).

### `scripts/replay_motion.py` — replay a saved episode

Loads `data/<robot>/<index>.json`, optionally drives the FramesViewer 3D view, then plots linear/angular/joint velocities with Matplotlib.

```bash
python scripts/replay_motion.py --robot bdx --index 0
```

Without OpenGL / headless:

```bash
python scripts/replay_motion.py --robot bdx --index 0 --no-viewer
```

### `scripts/view_robot.py` — static URDF + Meshcat

Loads `robots/<robot>/<robot>.urdf` via PlaCo and displays it in Meshcat (simple joint animation loop).

```bash
python scripts/view_robot.py --robot bdx
```

---

### Direct gait run (not under `scripts/`)

For a single episode with explicit velocities:

```bash
python src/motion_generator/gait_generator.py --index 0 --robot bdx --dx 0.0 --dy 0.0 --dth 0.0 --stand true --duration 10
```

Use `PYTHONPATH=src` if the package is not installed editable.

## References

- [Open Duck reference motion generator](https://github.com/apirrone/Open_Duck_reference_motion_generator) — related PlaCo-based reference motions for imitation learning (Open Duck / BDX-style workflows).
