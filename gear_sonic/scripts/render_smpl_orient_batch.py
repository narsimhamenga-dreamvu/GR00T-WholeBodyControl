#!/usr/bin/env python3
"""Batch render SMPL orientation videos for all episodes.

Transform: Y-up Z-forward (CoMotion world) → Z-up X-forward (MuJoCo), position fixed at origin.
Output: /hdd/gmr_sonic_out/viz_hq/videos/smpl_orient/<episode_id>.mp4

Usage:
    conda run -n mujoco_vis python gear_sonic/scripts/render_smpl_orient_batch.py \
        --smpl_dir /hdd/gmr_sonic_out/viz_hq/smpl \
        --out_dir  /hdd/gmr_sonic_out/viz_hq/videos/smpl_orient \
        --workers 8
"""

import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


# CoMotion world frame: X=right, Y=up, Z=forward
# MuJoCo frame:         X=forward, Y=right, Z=up
WORLD_R = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
WORLD_R_scipy = Rotation.from_matrix(WORLD_R)
# SMPL body (CoMotion): X=left, Y=down, Z=forward. Remap to X=forward, Y=right, Z=up.
_SMPL_BODY_FIX = Rotation.from_matrix(np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=np.float64))

MJCF = """
<mujoco model="smpl_orient">
  <option gravity="0 0 0" timestep="0.04"/>
  <visual>
    <rgba haze="0.15 0.15 0.15 1"/>
  </visual>
  <worldbody>
    <light pos="0 2 3" dir="0 -0.5 -1" diffuse="1 1 1" specular="0.3 0.3 0.3"/>
    <geom type="plane" size="3 3 0.1" rgba="0.25 0.25 0.25 1" pos="0 0 -0.1"/>

    <!-- World reference axes (faint) -->
    <geom type="cylinder" fromto="0 0 0 0.3 0 0" size="0.005" rgba="1 0 0 0.2"/>
    <geom type="cylinder" fromto="0 0 0 0 0.3 0" size="0.005" rgba="0 1 0 0.2"/>
    <geom type="cylinder" fromto="0 0 0 0 0 0.3" size="0.005" rgba="0 0 1 0.2"/>

    <body name="smpl_root" pos="0 0 0">
      <freejoint name="root"/>

      <!-- Body axes: red=X(forward), green=Y(right), blue=Z(up) -->
      <geom type="cylinder" fromto="0 0 0 0.5 0 0" size="0.012" rgba="1 0.1 0.1 1"/>
      <geom type="cylinder" fromto="0 0 0 0 0.5 0" size="0.012" rgba="0.1 1 0.1 1"/>
      <geom type="cylinder" fromto="0 0 0 0 0 0.5" size="0.012" rgba="0.1 0.1 1 1"/>

      <!-- Pelvis sphere (white) -->
      <geom type="sphere" size="0.06" rgba="0.85 0.75 0.65 1"/>

      <!-- Torso: capsule leaning slightly forward along X -->
      <geom type="capsule" fromto="-0.05 0 0.05  0.05 0 0.28" size="0.07" rgba="0.85 0.75 0.65 1"/>

      <!-- Head at the top (blue/up = +Z body direction) -->
      <geom type="sphere" pos="0 0 0.46" size="0.11" rgba="0.85 0.75 0.65 1"/>

      <!-- Nose: offset toward forward (+X) from head center — shows which way is front -->
      <geom type="sphere" pos="0.11 0 0.46" size="0.035" rgba="0.65 0.45 0.35 1"/>

      <!-- Shoulder stubs: extend along +/-Y (green/right) -->
      <geom type="capsule" fromto="0  0.12 0.22  0  0.28 0.12" size="0.04" rgba="0.85 0.75 0.65 1"/>
      <geom type="capsule" fromto="0 -0.12 0.22  0 -0.28 0.12" size="0.04" rgba="0.85 0.75 0.65 1"/>
    </body>
  </worldbody>
</mujoco>
"""


def render_episode(args):
    npz_path, out_path, fps, width, height = args
    try:
        data = np.load(npz_path)
        global_orient_aa = data["pose"][:, :3]
        T = len(global_orient_aa)

        R_body = Rotation.from_rotvec(global_orient_aa)
        R_world = WORLD_R_scipy * R_body * _SMPL_BODY_FIX
        quats_xyzw = R_world.as_quat()
        quats_wxyz = quats_xyzw[:, [3, 0, 1, 2]]
        trans = np.zeros((T, 3))

        model = mujoco.MjModel.from_xml_string(MJCF)
        mjdata = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=height, width=width)

        cam = mujoco.MjvCamera()
        cam.distance  = 3.0
        cam.elevation = -20
        cam.azimuth   = 45

        frames = []
        for t in range(T):
            mjdata.qpos[0:3] = trans[t]
            mjdata.qpos[3:7] = quats_wxyz[t]
            mujoco.mj_forward(model, mjdata)
            renderer.update_scene(mjdata, camera=cam)
            frames.append(renderer.render().copy())
        renderer.close()

        imageio.mimwrite(str(out_path), frames, fps=int(fps), quality=7)
        return npz_path.stem, True, None
    except Exception as e:
        return npz_path.stem, False, str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smpl_dir", default="/hdd/gmr_sonic_out/viz_hq/smpl")
    parser.add_argument("--out_dir",  default="/hdd/gmr_sonic_out/viz_hq/videos/smpl_orient")
    parser.add_argument("--fps",      type=float, default=25.0)
    parser.add_argument("--width",    type=int,   default=640)
    parser.add_argument("--height",   type=int,   default=480)
    parser.add_argument("--workers",  type=int,   default=8)
    args = parser.parse_args()

    smpl_dir = Path(args.smpl_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(smpl_dir.glob("*.npz"))
    tasks = []
    for npz in npz_files:
        out = out_dir / (npz.stem + ".mp4")
        if out.exists():
            continue
        tasks.append((npz, out, args.fps, args.width, args.height))

    total   = len(npz_files)
    skipped = total - len(tasks)
    print(f"Episodes: {total}  already done: {skipped}  to render: {len(tasks)}  workers: {args.workers}")

    if not tasks:
        print("All done.")
        return

    done = skipped
    errors = []
    with Pool(processes=args.workers) as pool:
        for ep_id, ok, err in pool.imap_unordered(render_episode, tasks):
            done += 1
            if ok:
                print(f"  [{done}/{total}] done: {ep_id}", flush=True)
            else:
                errors.append((ep_id, err))
                print(f"  [{done}/{total}] FAIL: {ep_id} — {err}", flush=True)

    print(f"\nFinished. {done - len(errors)}/{total} saved -> {out_dir}")
    if errors:
        print(f"{len(errors)} failures:")
        for ep_id, err in errors:
            print(f"  {ep_id}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
