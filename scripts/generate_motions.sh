#!/usr/bin/env bash
# Batch-generate reference motions headlessly (Linux equivalent of running scripts/main.py by hand).
#
#   scripts/generate_motions.sh <robot> <num> <jobs> [motions_root]
#
# Clips are written to <motions_root>/<robot>/<index>.json. Default motions_root per robot:
#   arti_h1: $ARTI_H1_MOTIONS_DIR (default ../arti_h1_local/source/arti_h1/arti_h1/assets/motions)
#   arti_d1: $ARTI_D1_MOTIONS_DIR (default ../arti_d1_local/source/arti_d1/arti_d1/assets/motions)
#   other:   data/  (repo-local, git-ignored)
# Relative defaults are resolved against the repo root. Refuses to write into a folder that already
# holds .json clips unless OVERWRITE=1. Extra main.py options via env: STAND (false), START_INDEX (0).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-$REPO/.venv/bin/python}"

if [ $# -lt 3 ]; then
    sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
fi

ROBOT="$1"; NUM="$2"; JOBS="$3"

case "$ROBOT" in
    arti_h1) DEFAULT_ROOT="${ARTI_H1_MOTIONS_DIR:-../arti_h1_local/source/arti_h1/arti_h1/assets/motions}" ;;
    arti_d1) DEFAULT_ROOT="${ARTI_D1_MOTIONS_DIR:-../arti_d1_local/source/arti_d1/arti_d1/assets/motions}" ;;
    *)       DEFAULT_ROOT="data" ;;
esac
ROOT="${4:-$DEFAULT_ROOT}"
case "$ROOT" in /*) ;; *) ROOT="$REPO/$ROOT" ;; esac
OUT_DIR="$ROOT/$ROBOT"

if [ "${OVERWRITE:-0}" != "1" ] && compgen -G "$OUT_DIR/*.json" > /dev/null; then
    echo "error: $OUT_DIR already contains .json clips; set OVERWRITE=1 to write into it" >&2
    exit 1
fi

echo "robot=$ROBOT num=$NUM jobs=$JOBS -> $OUT_DIR"
"$PY" "$REPO/scripts/main.py" --robot "$ROBOT" --num "$NUM" --jobs "$JOBS" \
    --start_index "${START_INDEX:-0}" --stand "${STAND:-false}" --visual false --out_dir "$OUT_DIR"
echo "done: $(find "$OUT_DIR" -maxdepth 1 -name '*.json' | wc -l) clips in $OUT_DIR"
