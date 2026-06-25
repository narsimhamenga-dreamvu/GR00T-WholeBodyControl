#!/usr/bin/env python3
"""Batch render SONIC policy robot_states_env*.npz files to MP4 videos.

Each robot_states_env{i:03d}.npz is mapped to its motion key via the motion lib
PKL (sorted key order matches env index order from eval).

Usage:
    conda run -n mujoco_vis python gear_sonic/scripts/render_robot_states_batch.py \
        --states_dir /mnt/bucket-processed-raw-mount/Retargeting_Tests/sonic_inference/yaw_postik_eval \
        --motion_lib gear_sonic/data/gmr_yaw_postik_motion_lib.pkl \
        --out_dir    /hdd/gmr_sonic_out/yaw_postik_eval/videos/sonic_out \
        --workers 8
"""

import argparse
import os
import re
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio
import joblib
import mujoco
import numpy as np

MJCF_PATH = "gear_sonic/data/assets/robot_description/mjcf/g1_29dof_rev_1_0.xml"
CAM_DISTANCE = 4.0
CAM_ELEVATION = -15
CAM_AZIMUTH = 90


def render_one(args):
    npz_path, out_path, width, height = args
    try:
        d = np.load(npz_path)
        root_pos = d["root_pos"]          # (T, 3)
        root_quat = d["root_quat_wxyz"]   # (T, 4) wxyz
        dof = d["joint_pos_mjcf"]         # (T, 29)
        fps = float(d["fps"])
        T = len(root_pos)

        model = mujoco.MjModel.from_xml_path(MJCF_PATH)
        mjdata = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=height, width=width)

        cam = mujoco.MjvCamera()
        cam.distance = CAM_DISTANCE
        cam.elevation = CAM_ELEVATION
        cam.azimuth = CAM_AZIMUTH

        frames = []
        for t in range(T):
            mjdata.qpos[0:3] = root_pos[t]
            mjdata.qpos[3:7] = root_quat[t]
            mjdata.qpos[7:7 + dof.shape[1]] = dof[t]
            mujoco.mj_forward(model, mjdata)
            cam.lookat[:] = root_pos[t] + np.array([0.0, 0.0, 0.8])
            renderer.update_scene(mjdata, camera=cam)
            frames.append(renderer.render().copy())
        renderer.close()

        out_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(str(out_path), frames, fps=int(fps), quality=8)
        return npz_path.name, True, None
    except Exception as e:
        return npz_path.name, False, str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--states_dir", required=True,
                        help="Directory containing robot_states_env*.npz files")
    parser.add_argument("--motion_lib",  required=True,
                        help="Motion lib PKL used for the eval (to resolve episode IDs)")
    parser.add_argument("--out_dir",    required=True,
                        help="Output directory for MP4 files")
    parser.add_argument("--width",   type=int, default=640)
    parser.add_argument("--height",  type=int, default=480)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    states_dir = Path(args.states_dir)
    out_dir = Path(args.out_dir)

    # Map env index → episode ID via sorted motion lib keys
    print(f"Loading motion lib keys from {args.motion_lib} ...")
    motion_data = joblib.load(args.motion_lib)
    episode_ids = sorted(motion_data.keys())  # sorted order matches eval env assignment
    print(f"  {len(episode_ids)} episodes")

    # Collect NPZ files sorted by env index
    npz_files = sorted(states_dir.glob("robot_states_env*.npz"),
                       key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)))
    print(f"  {len(npz_files)} NPZ files in {states_dir}")

    if len(npz_files) != len(episode_ids):
        print(f"WARNING: {len(npz_files)} NPZ files but {len(episode_ids)} episode IDs — "
              f"using min({len(npz_files)}, {len(episode_ids)})")

    tasks = []
    for i, npz in enumerate(npz_files):
        if i >= len(episode_ids):
            break
        ep_id = episode_ids[i]
        out_path = out_dir / f"{ep_id}.mp4"
        if out_path.exists():
            continue
        tasks.append((npz, out_path, args.width, args.height))

    total = min(len(npz_files), len(episode_ids))
    skipped = total - len(tasks)
    print(f"To render: {len(tasks)}  already done: {skipped}  workers: {args.workers}")

    if not tasks:
        print("All done.")
        return

    done = skipped
    errors = []
    with Pool(processes=args.workers) as pool:
        for name, ok, err in pool.imap_unordered(render_one, tasks):
            done += 1
            if ok:
                print(f"  [{done}/{total}] {name}", flush=True)
            else:
                errors.append((name, err))
                print(f"  [{done}/{total}] FAIL: {name} — {err}", flush=True)

    print(f"\nDone. {done - len(errors)}/{total} saved -> {out_dir}")
    if errors:
        for name, err in errors:
            print(f"  {name}: {err}")


if __name__ == "__main__":
    main()
