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

DOF_AXIS = np.array([
    [0, 1, 0], [1, 0, 0], [0, 0, 1],        # left hip pitch/roll/yaw
    [0, 1, 0], [0, 1, 0], [1, 0, 0],        # left knee, ankle pitch/roll
    [0, 1, 0], [1, 0, 0], [0, 0, 1],        # right hip pitch/roll/yaw
    [0, 1, 0], [0, 1, 0], [1, 0, 0],        # right knee, ankle pitch/roll
    [0, 0, 1], [1, 0, 0], [0, 1, 0],        # waist yaw/roll/pitch
    [0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],  # left arm
    [0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],  # right arm
], dtype=np.float32)


def convert_gmr_to_sonic(motion: dict) -> dict:
    root_trans_offset = motion["root_pos"].astype(np.float32)
    root_quat_xyzw = motion["root_rot"].astype(np.float32)
    dof = motion["dof_pos"].astype(np.float32)

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
        key = pkl_path.stem
        all_motions[key] = convert_gmr_to_sonic(motion)
        print(f"  Converted {pkl_path.name}  T={all_motions[key]['dof'].shape[0]}  fps={all_motions[key]['fps']}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(all_motions, args.output)
    print(f"\nSaved {len(all_motions)} motions -> {args.output}")


if __name__ == "__main__":
    main()
