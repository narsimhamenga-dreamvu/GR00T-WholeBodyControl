#!/usr/bin/env python3
"""Extract per-episode SMPL data from source PT files using a CSV episode index.

The CSV must have columns: episode_id, file, start_frame, end_frame, track_id.
This matches the format of GMR's episodes_viz.csv.

Usage:
    conda run -n h2r python gear_sonic/data_process/extract_smpl_per_episode.py \
        --csv /home/ubuntu/menga/GMR/input_data/episodes_viz.csv \
        --output /hdd/gmr_sonic_out/viz_hq/smpl
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Cache loaded PT files — many episodes share the same source file
_pt_cache: dict[str, dict] = {}


def load_pt(path: str) -> dict:
    if path not in _pt_cache:
        data = torch.load(path, map_location="cpu", weights_only=False)
        _pt_cache[path] = {k: v.numpy() if hasattr(v, "numpy") else v
                           for k, v in data.items()}
    return _pt_cache[path]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",    required=True, type=Path, help="episodes_viz.csv")
    parser.add_argument("--output", required=True, type=Path, help="Output directory for NPZ files")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    with open(args.csv) as f:
        rows = list(csv.DictReader(f))

    skipped = 0
    for row in tqdm(rows, desc="Episodes"):
        episode_id  = row["episode_id"]
        source_file = row["file"]
        start_frame = int(row["start_frame"])
        end_frame   = int(row["end_frame"])
        track_id    = int(row["track_id"])

        out_path = args.output / f"{episode_id}.npz"
        if out_path.exists():
            continue  # already done

        try:
            pt = load_pt(source_file)
        except Exception as e:
            tqdm.write(f"  SKIP {episode_id}: cannot load {source_file}: {e}")
            skipped += 1
            continue

        sl = slice(start_frame, end_frame)

        # Filter to the specific track_id using the 'id' field
        pose      = pt.get("pose")
        trans     = pt.get("trans")
        ids       = pt.get("id")
        frame_idx = pt.get("frame_idx")
        betas     = pt.get("betas")

        if pose is None or len(pose) == 0:
            tqdm.write(f"  SKIP {episode_id}: PT has no pose data")
            skipped += 1
            continue

        # start_frame/end_frame are video frame numbers stored in frame_idx,
        # NOT row indices into the PT array.  Filter by frame_idx value + track_id.
        if frame_idx is not None and ids is not None:
            mask = (frame_idx >= start_frame) & (frame_idx < end_frame) & (ids == track_id)
        elif ids is not None:
            # No frame_idx — fall back to row-index slice then track filter
            mask_ids = ids[sl] == track_id
            pose_sl  = pose[sl][mask_ids]
            trans_sl = trans[sl][mask_ids] if trans is not None else None
            mask     = None
        else:
            mask = None

        if mask is not None:
            if mask.sum() == 0:
                tqdm.write(f"  SKIP {episode_id}: track_id {track_id} not found in "
                           f"frame range [{start_frame}:{end_frame}]")
                skipped += 1
                continue
            pose_sl      = pose[mask]
            trans_sl     = trans[mask]     if trans     is not None else None
            ids_sl       = ids[mask]
            frame_idx_sl = frame_idx[mask] if frame_idx is not None else None
            betas_sl     = betas[mask]     if betas     is not None else None
        else:
            ids_sl       = None
            frame_idx_sl = None
            betas_sl     = betas[sl] if betas is not None else None

        if len(pose_sl) == 0:
            tqdm.write(f"  SKIP {episode_id}: empty result for track {track_id} "
                       f"in frames [{start_frame}:{end_frame}]")
            skipped += 1
            continue

        out = {"pose": pose_sl, "episode_id": episode_id, "track_id": track_id,
               "start_frame": start_frame, "end_frame": end_frame}
        if trans_sl     is not None: out["trans"]     = trans_sl
        if betas_sl     is not None: out["betas"]     = betas_sl
        if frame_idx_sl is not None: out["frame_idx"] = frame_idx_sl

        np.savez(out_path, **out)

    total = len(rows)
    print(f"\nSaved {total - skipped}/{total} episodes -> {args.output}")
    if skipped:
        print(f"Skipped {skipped} (unreadable source or empty slice)")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
