#!/bin/bash
# Batch render SMPL orientation videos (transform ON, position OFF) for all episodes.
# Output: /hdd/gmr_sonic_out/viz_hq/videos/smpl_orient/<episode_id>.mp4
#
# Usage:
#   bash gear_sonic/scripts/render_smpl_orient_batch.sh [NUM_PARALLEL]
#
# NUM_PARALLEL defaults to 8.

set -e

SMPL_DIR="/hdd/gmr_sonic_out/viz_hq/smpl"
OUT_DIR="/hdd/gmr_sonic_out/viz_hq/videos/smpl_orient"
NUM_PARALLEL="${1:-8}"

mkdir -p "$OUT_DIR"

npz_files=("$SMPL_DIR"/*.npz)
total=${#npz_files[@]}
echo "Found $total episodes. Rendering with $NUM_PARALLEL parallel workers..."

render_one() {
    local npz="$1"
    local ep_id
    ep_id=$(basename "$npz" .npz)
    local out="$OUT_DIR/${ep_id}.mp4"

    if [ -f "$out" ]; then
        return 0
    fi

    conda run -n mujoco_vis python gear_sonic/visualize_smpl_orient.py \
        --npz "$npz" \
        --comotion \
        --no-position \
        --output "$out" \
        2>/dev/null

    echo "  done: $ep_id"
}

export -f render_one

printf '%s\n' "${npz_files[@]}" | \
    xargs -P "$NUM_PARALLEL" -I{} bash -c 'render_one "$@"' _ {}

echo "All done -> $OUT_DIR"
