#!/usr/bin/env python3
"""Convert SOMA retargeter HU_D04 CSV data to motion_lib format for SONIC training.

Like convert_soma_csv_to_motion_lib.py but for HU_D04 (31 DOF, 32 bodies).
Joint axes extracted from the MJCF to handle oblique hip-pitch axes.

Usage:
    python gear_sonic/data_process/convert_hu_d04_csv_to_motion_lib.py \
        --input /path/to/hu_d04_csvs/ \
        --output data/hu_d04_motions/robot \
        --fps 30 --fps_source 120 --individual --num_workers 4
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import joblib
import numpy as np
from scipy.spatial import transform

NUM_DOF = 31
NUM_BODIES = 32  # base_link + 31 actuated links

# HU_D04 joint axes in MuJoCo DFS order (from hu_d04.xml).
# Most are cardinal axes except hip_pitch joints which are oblique.
DOF_AXIS = np.array(
    [
        [0, 0.90631, -0.42262],  # left_hip_pitch (oblique)
        [1, 0, 0],                # left_hip_roll
        [0, 0, 1],                # left_hip_yaw
        [0, 1, 0],                # left_knee
        [0, 1, 0],                # left_ankle_pitch
        [1, 0, 0],                # left_ankle_roll
        [0, 0.90631, 0.42262],   # right_hip_pitch (oblique, mirrored)
        [1, 0, 0],                # right_hip_roll
        [0, 0, 1],                # right_hip_yaw
        [0, 1, 0],                # right_knee
        [0, 1, 0],                # right_ankle_pitch
        [1, 0, 0],                # right_ankle_roll
        [0, 0, 1],                # waist_yaw
        [1, 0, 0],                # waist_roll
        [0, 1, 0],                # waist_pitch
        [0, 0, 1],                # head_yaw
        [0, 1, 0],                # head_pitch
        [0, 1, 0],                # left_shoulder_pitch
        [1, 0, 0],                # left_shoulder_roll
        [0, 0, 1],                # left_shoulder_yaw
        [0, 1, 0],                # left_elbow
        [0, 0, 1],                # left_wrist_yaw
        [0, 1, 0],                # left_wrist_pitch
        [1, 0, 0],                # left_wrist_roll
        [0, 1, 0],                # right_shoulder_pitch
        [1, 0, 0],                # right_shoulder_roll
        [0, 0, 1],                # right_shoulder_yaw
        [0, 1, 0],                # right_elbow
        [0, 0, 1],                # right_wrist_yaw
        [0, 1, 0],                # right_wrist_pitch
        [1, 0, 0],                # right_wrist_roll
    ],
    dtype=np.float32,
)

# Normalize oblique axes
for i in range(NUM_DOF):
    n = np.linalg.norm(DOF_AXIS[i])
    if n > 0:
        DOF_AXIS[i] /= n


def load_bones_csv(csv_path: str) -> dict:
    """Load a single HU_D04 flat CSV motion file (from SOMA retargeter).

    CSV format: Frame, root_translate{X,Y,Z}, root_rotate{X,Y,Z}, 31 joint DOFs.
    All angles in degrees, positions in centimeters.
    """
    import pandas as pd

    data = pd.read_csv(csv_path)
    T = len(data)

    root_pos = (
        np.stack(
            [
                data["root_translateX"].values,
                data["root_translateY"].values,
                data["root_translateZ"].values,
            ],
            axis=1,
        ).astype(np.float32)
        / 100.0
    )

    euler_deg = np.stack(
        [
            data["root_rotateX"].values,
            data["root_rotateY"].values,
            data["root_rotateZ"].values,
        ],
        axis=1,
    ).astype(np.float64)
    root_quat_xyzw = (
        transform.Rotation.from_euler("xyz", euler_deg, degrees=True).as_quat().astype(np.float32)
    )
    root_quat_wxyz = root_quat_xyzw[:, [3, 0, 1, 2]]

    joint_cols = [c for c in data.columns if c.endswith("_dof")]
    assert len(joint_cols) == NUM_DOF, (
        f"Expected {NUM_DOF} DOF columns, got {len(joint_cols)}: {joint_cols}"
    )
    joint_pos_mj = np.deg2rad(data[joint_cols].values).astype(np.float32)

    body_pos_w = np.zeros((T, 14, 3), dtype=np.float32)
    body_pos_w[:, 0, :] = root_pos
    body_quat_w = np.zeros((T, 14, 4), dtype=np.float32)
    body_quat_w[:, :, 0] = 1.0
    body_quat_w[:, 0, :] = root_quat_wxyz

    return {
        "joint_pos": joint_pos_mj,
        "body_pos_w": body_pos_w,
        "body_quat_w": body_quat_w,
        "joint_order": "mj",
    }


def convert_sequence(seq_data: dict, fps: int) -> dict:
    """Convert a single sequence to motion_lib format."""
    joint_pos = seq_data["joint_pos"]  # (T, 31)
    body_pos_w = seq_data["body_pos_w"]
    body_quat_w = seq_data["body_quat_w"]

    T = joint_pos.shape[0]

    root_trans_offset = body_pos_w[:, 0, :].copy()
    root_quat_wxyz = body_quat_w[:, 0, :]
    root_quat_xyzw = root_quat_wxyz[:, [1, 2, 3, 0]]

    dof_mj = joint_pos[:, :NUM_DOF]

    pose_aa = np.zeros((T, NUM_BODIES, 3), dtype=np.float32)
    pose_aa[:, 1:NUM_BODIES, :] = DOF_AXIS[None, :, :] * dof_mj[:, :, None]
    pose_aa[:, 0, :] = transform.Rotation.from_quat(root_quat_xyzw).as_rotvec()

    return {
        "root_trans_offset": root_trans_offset.astype(np.float32),
        "pose_aa": pose_aa.astype(np.float32),
        "dof": dof_mj.astype(np.float32),
        "root_rot": root_quat_wxyz.astype(np.float32),
        "fps": fps,
    }


def downsample(data: dict, factor: int) -> dict:
    """Downsample motion data by an integer factor."""
    return {
        "joint_pos": data["joint_pos"][::factor],
        "body_pos_w": data["body_pos_w"][::factor],
        "body_quat_w": data["body_quat_w"][::factor],
        "joint_order": data.get("joint_order", "mj"),
    }


def process_single_csv(args):
    """Process one CSV file -> one PKL. Called by parallel workers."""
    csv_path, output_dir, fps, fps_source = args
    name = Path(csv_path).stem
    try:
        raw = load_bones_csv(csv_path)
        factor = max(1, round(fps_source / fps))
        if factor > 1:
            raw = downsample(raw, factor)
        result = convert_sequence(raw, fps)
        out = {name: result}
        out_path = os.path.join(output_dir, f"{name}.pkl")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        joblib.dump(out, out_path)
        return name, True
    except Exception as e:
        print(f"  FAILED {name}: {e}", file=sys.stderr)
        return name, False


def main():
    print(f"HU_D04 {NUM_DOF} DOFs, {NUM_BODIES} bodies")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV dir or parent of session dirs")
    parser.add_argument("--output", required=True, help="Output directory for PKL files")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS")
    parser.add_argument("--fps_source", type=int, default=120, help="Source CSV FPS")
    parser.add_argument("--individual", action="store_true", help="One PKL per motion")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    input_path = Path(args.input)
    csv_files = sorted(input_path.rglob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {input_path}")
        return

    # Group by parent directory (session)
    sessions = {}
    for f in csv_files:
        session = f.parent.name
        sessions.setdefault(session, []).append(str(f))

    print(f"\nBatch converting {len(sessions)} sessions with {args.num_workers} workers")
    print(f"Output: {args.output}")

    total_ok = 0
    total_fail = 0
    for session_name, files in sessions.items():
        out_dir = os.path.join(args.output, session_name)
        os.makedirs(out_dir, exist_ok=True)

        tasks = [(f, out_dir, args.fps, args.fps_source) for f in files]
        ok = fail = 0
        with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
            futures = [pool.submit(process_single_csv, t) for t in tasks]
            for future in as_completed(futures):
                _, success = future.result()
                if success:
                    ok += 1
                else:
                    fail += 1
        print(f"  {session_name}: {ok}/{ok + fail} converted")
        total_ok += ok
        total_fail += fail

    print(f"\nDone: {total_ok} motions converted, {total_fail} failed, {total_ok + total_fail} total CSVs")


if __name__ == "__main__":
    main()
