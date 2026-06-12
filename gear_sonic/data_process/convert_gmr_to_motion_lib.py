#!/usr/bin/env python3
"""Convert GMR-retargeted G1 PKL files to SONIC motion_lib format.

Usage:
    conda run -n h2r python gear_sonic/data_process/convert_gmr_to_motion_lib.py \
        --input  /home/ubuntu/menga/GMR/outputs/test_10/pkls \
        --output gear_sonic/data/gmr_test10_motion_lib.pkl
"""

import argparse
import pickle
from pathlib import Path

import joblib
import numpy as np
from scipy.spatial.transform import Rotation

NUM_DOF = 29
NUM_BODIES = 30  # pelvis + 29 joints

# GMR produces corrupted root_pos in the first few frames of each clip
# (tracking instability), causing 4-5m teleportation jumps.  Any inter-frame
# jump larger than this threshold is treated as a bad frame boundary.
JUMP_THRESHOLD_M = 1.0   # metres per frame at 25 fps
MIN_STABLE_FRAMES = 30   # clips shorter than this after trimming are dropped

# GMR also produces bad root orientation (large tilt) in the first few frames
# while the IK solver converges.  We skip ahead until the robot is roughly
# upright and stays that way for a short settling window.
MAX_TILT_DEG = 20.0      # max acceptable tilt from vertical (degrees)
TILT_SETTLE_FRAMES = 10  # consecutive frames that must be below threshold

# Joint limits from the G1 URDF, in DOF order (lower, upper) in radians.
G1_JOINT_LIMITS = np.array([
    [-2.5307,  2.8798], [-0.5236,  2.9671], [-2.7576,  2.7576],  # left hip pitch/roll/yaw
    [-0.0873,  2.8798], [-0.8727,  0.5236], [-0.2618,  0.2618],  # left knee, ankle pitch/roll
    [-2.5307,  2.8798], [-2.9671,  0.5236], [-2.7576,  2.7576],  # right hip pitch/roll/yaw
    [-0.0873,  2.8798], [-0.8727,  0.5236], [-0.2618,  0.2618],  # right knee, ankle pitch/roll
    [-2.618,   2.618],  [-0.52,    0.52],   [-0.52,    0.52],    # waist yaw/roll/pitch
    [-3.0892,  2.6878], [-1.5882,  2.2515], [-2.618,   2.618],   # left shoulder pitch/roll/yaw
    [-1.0472,  2.0944], [-1.9722,  1.9722], [-1.6057,  1.6057], [-1.9722, 1.9722],  # left elbow, wrist r/p/y
    [-2.6878,  3.0892], [-2.2515,  1.5882], [-2.618,   2.618],   # right shoulder pitch/roll/yaw
    [-2.0944,  1.0472], [-1.9722,  1.9722], [-1.6057,  1.6057], [-1.9722, 1.9722],  # right elbow, wrist r/p/y
], dtype=np.float32)

DOF_AXIS = np.array([
    [0, 1, 0], [1, 0, 0], [0, 0, 1],        # left hip pitch/roll/yaw
    [0, 1, 0], [0, 1, 0], [1, 0, 0],        # left knee, ankle pitch/roll
    [0, 1, 0], [1, 0, 0], [0, 0, 1],        # right hip pitch/roll/yaw
    [0, 1, 0], [0, 1, 0], [1, 0, 0],        # right knee, ankle pitch/roll
    [0, 0, 1], [1, 0, 0], [0, 1, 0],        # waist yaw/roll/pitch
    [0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],  # left arm
    [0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],  # right arm
], dtype=np.float32)


def find_stable_start(root_pos: np.ndarray, threshold: float = JUMP_THRESHOLD_M) -> int:
    """Return first stable frame index (after last inter-frame jump > threshold)."""
    diffs = np.linalg.norm(np.diff(root_pos, axis=0), axis=1)
    bad = np.where(diffs > threshold)[0]
    return int(bad[-1]) + 1 if len(bad) > 0 else 0


def find_tilt_start(
    root_quat_xyzw: np.ndarray,
    max_tilt_deg: float = MAX_TILT_DEG,
    settle_frames: int = TILT_SETTLE_FRAMES,
) -> int:
    """Return first frame where tilt from vertical stays below max_tilt_deg
    for settle_frames consecutive frames.  Returns -1 if it never converges."""
    R = Rotation.from_quat(root_quat_xyzw)
    # Robot Z-axis in world frame; tilt = angle between it and world Z [0,0,1]
    robot_z = R.apply(np.array([0.0, 0.0, 1.0]))
    tilt_deg = np.degrees(np.arccos(np.clip(robot_z[:, 2], -1.0, 1.0)))
    below = tilt_deg < max_tilt_deg
    for i in range(len(below) - settle_frames + 1):
        if below[i : i + settle_frames].all():
            return i
    return -1  # tilt never converges


def compute_start(motion: dict) -> tuple[int, int, int]:
    """Return (start, jump_start, tilt_start).

    start        = frame index to use after all filtering
    jump_start   = frame chosen by the position-jump detector
    tilt_start   = frame chosen by the tilt-convergence detector (-1 = never)
    """
    raw_pos = motion["root_pos"].astype(np.float32)
    raw_quat = motion["root_rot"].astype(np.float32)
    jump_start = find_stable_start(raw_pos)
    tilt_start = find_tilt_start(raw_quat)
    if tilt_start == -1:
        return -1, jump_start, tilt_start  # clip never reaches good orientation
    start = max(jump_start, tilt_start)
    return start, jump_start, tilt_start


def convert_gmr_to_sonic(motion: dict) -> dict | None:
    start, _, _ = compute_start(motion)
    if start == -1:
        return None
    raw_pos = motion["root_pos"].astype(np.float32)
    stable_frames = len(raw_pos) - start
    if stable_frames < MIN_STABLE_FRAMES:
        return None

    root_trans_offset = raw_pos[start:].copy()
    root_trans_offset[:, :2] -= root_trans_offset[0, :2]  # re-centre XY to origin

    root_quat_xyzw = motion["root_rot"].astype(np.float32)[start:]
    dof = motion["dof_pos"].astype(np.float32)[start:]
    dof = np.clip(dof, G1_JOINT_LIMITS[:, 0], G1_JOINT_LIMITS[:, 1])

    T = root_trans_offset.shape[0]
    pose_aa = np.zeros((T, NUM_BODIES, 3), dtype=np.float32)
    pose_aa[:, 0, :] = Rotation.from_quat(root_quat_xyzw).as_rotvec()
    pose_aa[:, 1:, :] = DOF_AXIS[None, :, :] * dof[:, :, None]

    return {
        "root_trans_offset": root_trans_offset,
        "pose_aa": pose_aa,
        "dof": dof,
        "root_rot": root_quat_xyzw,
        "smpl_joints": np.zeros((T, 24, 3), dtype=np.float32),
        "fps": float(motion["fps"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    pkl_files = sorted(args.input.glob("*.pkl"))
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl files found in {args.input}")

    all_motions = {}
    for pkl_path in pkl_files:
        with open(pkl_path, "rb") as f:
            motion = pickle.load(f)

        start, jump_start, tilt_start = compute_start(motion)
        raw_pos = motion["root_pos"].astype(np.float32)
        total_frames = len(raw_pos)

        if start == -1:
            print(f"  SKIP    {pkl_path.name}  (tilt never converges below {MAX_TILT_DEG}°)")
            continue

        stable_frames = total_frames - start
        if stable_frames < MIN_STABLE_FRAMES:
            print(f"  SKIP    {pkl_path.name}  (only {stable_frames} stable frames after trimming {start})")
            continue

        converted = convert_gmr_to_sonic(motion)
        if converted is None:
            continue

        raw_dof = motion["dof_pos"].astype(np.float32)[start:]
        n_clipped = int(
            np.any(
                (raw_dof < G1_JOINT_LIMITS[:, 0]) | (raw_dof > G1_JOINT_LIMITS[:, 1]),
                axis=1,
            ).sum()
        )
        clip_note = f"  [{n_clipped}/{converted['dof'].shape[0]} frames clipped]" if n_clipped else ""
        trim_parts = []
        if jump_start > 0:
            trim_parts.append(f"jump={jump_start}f")
        if tilt_start > 0:
            trim_parts.append(f"tilt={tilt_start}f")
        trim_note = f"  [trimmed {start}f: {', '.join(trim_parts)}]" if start > 0 else ""
        all_motions[pkl_path.stem] = converted
        print(f"  OK  {pkl_path.name}  T={converted['dof'].shape[0]}  fps={converted['fps']}{trim_note}{clip_note}")

    if not all_motions:
        raise RuntimeError("No motions survived filtering — check JUMP_THRESHOLD_M / MIN_STABLE_FRAMES")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(all_motions, args.output)
    print(f"\nSaved {len(all_motions)} motions -> {args.output}")


if __name__ == "__main__":
    main()
