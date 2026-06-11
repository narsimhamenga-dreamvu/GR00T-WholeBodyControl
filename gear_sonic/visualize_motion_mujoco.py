#!/usr/bin/env python3
"""Visualize G1 robot motion in MuJoCo and save an MP4.

Two modes:
  1. Reference motion from a motion_lib PKL (--motion):
       python gear_sonic/visualize_motion_mujoco.py --motion gear_sonic/data/gmr_test10_3_motion_lib.pkl

  2. Actual SONIC policy output logged during eval (--states):
       python gear_sonic/visualize_motion_mujoco.py --states /tmp/robot_states.npz

For mode 2, first run the eval with ++log_robot_states=/tmp/robot_states.npz
"""
import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")  # headless GPU rendering via EGL

import mujoco
import numpy as np

MJCF_PATH = "gear_sonic/data/assets/robot_description/mjcf/g1_29dof_rev_1_0.xml"
CAMERA_OFFSET = np.array([2.5, 2.5, 1.2])


def xyzw_to_wxyz(q):
    return q[[3, 0, 1, 2]]


def load_from_pkl(motion_path, key):
    import joblib
    motions = joblib.load(motion_path)
    k = key or list(motions.keys())[0]
    m = motions[k]
    print(f"Reference motion '{k}'  T={m['dof'].shape[0]}  fps={m['fps']}")
    return (
        m["root_trans_offset"],           # (T,3)
        np.array([xyzw_to_wxyz(r) for r in m["root_rot"]]),  # (T,4) wxyz
        m["dof"],                          # (T,29) MJCF order
        float(m["fps"]),
    )


def load_from_npz(states_path):
    d = np.load(states_path)
    print(f"Policy output  T={d['joint_pos_mjcf'].shape[0]}  fps={float(d['fps'])}")
    return (
        d["root_pos"],          # (T,3)
        d["root_quat_wxyz"],    # (T,4) wxyz
        d["joint_pos_mjcf"],    # (T,29) MJCF order
        float(d["fps"]),
    )


def render_frames(root_pos, root_quat_wxyz, dof, model, width, height):
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    frames = []
    for t in range(len(root_pos)):
        data.qpos[0:3] = root_pos[t]
        data.qpos[3:7] = root_quat_wxyz[t]
        data.qpos[7:7 + dof.shape[1]] = dof[t]
        mujoco.mj_forward(model, data)

        eye = root_pos[t] + CAMERA_OFFSET
        target = root_pos[t] + np.array([0.0, 0.0, 0.5])
        renderer.update_scene(data)
        cam = renderer.scene.camera[0]
        cam.pos[:] = eye
        fwd = target - eye
        cam.forward[:] = fwd / (np.linalg.norm(fwd) + 1e-9)
        cam.up[:] = [0.0, 0.0, 1.0]
        frames.append(renderer.render().copy())

    renderer.close()
    return frames


def main():
    parser = argparse.ArgumentParser()
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--motion", help="motion_lib PKL (reference motion)")
    src.add_argument("--states", help="robot_states.npz logged during SONIC eval (policy output)")
    parser.add_argument("--key",    default=None, help="motion key for --motion mode")
    parser.add_argument("--output", default="/tmp/sonic_mujoco.mp4")
    parser.add_argument("--width",  type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    try:
        import imageio
    except ImportError:
        raise SystemExit("Run: pip install imageio[ffmpeg]")

    if args.states:
        root_pos, root_quat, dof, fps = load_from_npz(args.states)
    else:
        root_pos, root_quat, dof, fps = load_from_pkl(args.motion, args.key)

    model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    frames = render_frames(root_pos, root_quat, dof, model, args.width, args.height)

    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    print(f"Rendering {len(frames)} frames -> {out_path}")
    imageio.mimwrite(out_path, frames, fps=int(fps), quality=8)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"Done. File size: {size_mb:.1f} MB")
