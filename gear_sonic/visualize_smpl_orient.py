#!/usr/bin/env python3
"""Visualize SMPL root orientation as an animated coordinate axis in MuJoCo.

Live viewer (default):
    conda run -n mujoco_vis python gear_sonic/visualize_smpl_orient.py \
        --npz /hdd/gmr_sonic_out/viz_hq/smpl/000024.npz

Headless video output:
    conda run -n mujoco_vis python gear_sonic/visualize_smpl_orient.py \
        --npz /hdd/gmr_sonic_out/viz_hq/smpl/000024.npz \
        --output /hdd/gmr_sonic_out/viz_hq/videos/smpl_orient/000024.mp4
"""

import argparse
import os
import time

import mujoco
import mujoco.viewer
import numpy as np
import torch
from scipy.spatial.transform import Rotation

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


def load_pt(path: str):
    data = torch.load(path, map_location="cpu", weights_only=False)
    global_orient_aa = data["pose"][:, :3].numpy()
    trans = data["trans"].numpy()
    return global_orient_aa, trans


def load_npz(path: str):
    data = np.load(path)
    global_orient_aa = data["pose"][:, :3]
    trans = data["trans"]
    return global_orient_aa, trans


# CoMotion world frame: X=right, Y=up, Z=forward
# MuJoCo frame:         X=forward, Y=right, Z=up
WORLD_R = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
WORLD_R_scipy = Rotation.from_matrix(WORLD_R)

# SMPL body convention (from CoMotion data): X=left, Y=down, Z=forward (right-handed).
# Remap to MuJoCo body: X (red)=forward, Y (green)=right, Z (blue)=up.
#   new X → SMPL +Z (forward),  new Y → SMPL -X (right=-left),  new Z → SMPL -Y (up=-down)
_SMPL_BODY_FIX = Rotation.from_matrix(np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=np.float64))


def apply_comotion_transform(global_orient_aa: np.ndarray, trans: np.ndarray):
    """Re-express CoMotion world-space (Y-up, Z-forward) in MuJoCo frame (Z-up, X-forward).

    After transform: blue (body Z) points up, red (body X) points roughly forward.
    """
    R_body = Rotation.from_rotvec(global_orient_aa)
    R_world = WORLD_R_scipy * R_body * _SMPL_BODY_FIX
    trans_world = trans @ WORLD_R.T
    return R_world, trans_world


def xyzw_to_wxyz(q):
    return q[[3, 0, 1, 2]]


def main():
    parser = argparse.ArgumentParser()
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--pt",  help="Path to full SMPL .pt file")
    src.add_argument("--npz", help="Path to per-episode SMPL .npz")
    parser.add_argument("--fps",      type=float, default=25.0)
    parser.add_argument("--loop",     action="store_true", default=True)
    parser.add_argument("--transform", action="store_true", default=False,
                        help="Apply coordinate transform: Y-up Z-fwd (CoMotion) → Z-up X-fwd (MuJoCo).")
    parser.add_argument("--translate", action="store_true", default=False,
                        help="Show root translation movement. Off by default (fixed at origin).")
    parser.add_argument("--output", default=None,
                        help="Save video to this path instead of opening live viewer "
                             "(requires imageio[ffmpeg]). Sets headless EGL rendering.")
    parser.add_argument("--width",  type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--yaw-only", action="store_true", default=False,
                        help="Show only yaw (rotation around Z/up axis); fix pitch and roll.")
    args = parser.parse_args()

    src_path = args.pt or args.npz
    print(f"Loading {src_path}")
    global_orient_aa, trans = load_pt(args.pt) if args.pt else load_npz(args.npz)
    T = len(global_orient_aa)

    if args.transform:
        print("  Transform: Y-up Z-fwd → Z-up X-fwd (MuJoCo)")
        R_all, trans = apply_comotion_transform(global_orient_aa, trans)
    else:
        print("  Transform: none (raw SMPL data)")
        R_all = Rotation.from_rotvec(global_orient_aa)

    if args.yaw_only:
        # Decompose into ZYX euler, keep only the Z (yaw) component
        yaws = R_all.as_euler('ZYX')[:, 0]   # (T,) yaw angles around world Z
        R_all = Rotation.from_euler('Z', yaws)
        print("  Yaw-only: X/Y rotations zeroed, Z (yaw) preserved")

    # Re-centre translation so it starts at origin; zero out if position display is off
    if args.translate:
        trans = trans - trans[0]
    else:
        trans = np.zeros_like(trans)

    print(f"  Frames: {T}  FPS: {args.fps}")
    print(f"  Translation range — X:[{trans[:,0].min():.2f}, {trans[:,0].max():.2f}]"
          f"  Y:[{trans[:,1].min():.2f}, {trans[:,1].max():.2f}]"
          f"  Z:[{trans[:,2].min():.2f}, {trans[:,2].max():.2f}]")

    quats_wxyz = np.array([xyzw_to_wxyz(q) for q in R_all.as_quat()])  # (T, 4) wxyz

    if args.output:
        os.environ.setdefault("MUJOCO_GL", "egl")

    model = mujoco.MjModel.from_xml_string(MJCF)
    data  = mujoco.MjData(model)

    if args.output:
        try:
            import imageio
        except ImportError:
            raise SystemExit("Run: pip install imageio[ffmpeg]")

        renderer = mujoco.Renderer(model, height=args.height, width=args.width)
        cam = mujoco.MjvCamera()
        cam.distance  = 3.0
        cam.elevation = -20
        cam.azimuth   = 45

        frames = []
        for t in range(T):
            data.qpos[0:3] = trans[t]
            data.qpos[3:7] = quats_wxyz[t]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render().copy())
        renderer.close()

        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        imageio.mimwrite(args.output, frames, fps=int(args.fps), quality=7)
        print(f"Saved {len(frames)} frames -> {args.output}")
    else:
        dt = 1.0 / args.fps
        t  = 0

        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.distance = 3.0
            viewer.cam.elevation = -20
            viewer.cam.azimuth   = 45

            print("Viewer open. Press Ctrl+C to quit.")
            while viewer.is_running():
                frame_start = time.time()

                data.qpos[0:3] = trans[t]
                data.qpos[3:7] = quats_wxyz[t]
                mujoco.mj_forward(model, data)
                viewer.sync()

                t = (t + 1) % T if args.loop else min(t + 1, T - 1)

                elapsed = time.time() - frame_start
                sleep   = max(0.0, dt - elapsed)
                time.sleep(sleep)


if __name__ == "__main__":
    main()
