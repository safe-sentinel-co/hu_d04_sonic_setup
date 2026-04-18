#!/usr/bin/env python3
"""Render an HU_D04 motion to MP4 (headless) and upload to S3."""
import os
os.environ["MUJOCO_GL"] = "osmesa"

import glob
import joblib
import numpy as np
import mujoco
import cv2
import boto3
from pathlib import Path
import sys


MJCF = "/workspace/GR00T-WholeBodyControl/gear_sonic/data/assets/robot_description/mjcf/hu_d04.xml"
PKL_DIR = "/workspace/hu_d04_motions/robot"
OUT = "/workspace/walk_test.mp4"
W, H = 640, 480
FPS = 30

# AWS config from env
AWS_KEY = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET = os.environ["AWS_SECRET_ACCESS_KEY"]
BUCKET = os.environ.get("S3_BUCKET", "safe-sentinel-inc")
S3_KEY = os.environ.get("S3_KEY", "hu_d04_retargeting/walk_test.mp4")


def pick_walking_pkl(target_frames_min=90, target_frames_max=300):
    """Find a walking motion that's a reasonable length for preview."""
    candidates = glob.glob(f"{PKL_DIR}/**/*walk*.pkl", recursive=True)
    print(f"found {len(candidates)} walk-themed pkls")
    # Rank by length, pick one in target range
    best = None
    for p in candidates:
        try:
            d = joblib.load(p)
            m = next(iter(d.values()))
            T = m["dof"].shape[0]
            if target_frames_min <= T <= target_frames_max:
                best = (p, m, T)
                break
        except Exception:
            continue
    if best is None:
        # fallback: any walk pkl
        p = candidates[0]
        d = joblib.load(p); m = next(iter(d.values())); T = m["dof"].shape[0]
        best = (p, m, T)
    return best


def render(m, out_path):
    model = mujoco.MjModel.from_xml_path(MJCF)
    data = mujoco.MjData(model)
    # freejoint: qpos[0:7] = xyz + quat (wxyz). PKL root_rot is quat; align order.
    renderer = mujoco.Renderer(model, height=H, width=W)

    # tracking side-camera
    cam = mujoco.MjvCamera()
    cam.lookat[:] = m["root_trans_offset"][0]
    cam.distance = 3.0
    cam.azimuth = 90
    cam.elevation = -10

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_path, fourcc, FPS, (W, H))
    if not vw.isOpened():
        print("ERR: VideoWriter failed to open"); return False

    T = m["dof"].shape[0]
    for t in range(T):
        # qpos layout for MuJoCo freejoint: [xyz(3), quat_wxyz(4), joint_q(31)]
        # Verified: converter stores root_rot in (w, x, y, z) order (see
        # convert_hu_d04_csv_to_motion_lib.py:151 — "root_quat_wxyz"), which matches
        # MuJoCo's freejoint convention directly.
        data.qpos[:3] = m["root_trans_offset"][t]
        data.qpos[3:7] = m["root_rot"][t]
        data.qpos[7:] = m["dof"][t]
        mujoco.mj_forward(model, data)
        cam.lookat[:] = data.qpos[:3]  # follow the root
        renderer.update_scene(data, camera=cam)
        frame = renderer.render()
        vw.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if t == 0 or t == T // 2 or t == T - 1:
            print(f"  frame {t}/{T}: root_z={data.qpos[2]:.3f}")
    vw.release()
    return True


def upload(path, bucket, key):
    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_KEY,
        aws_secret_access_key=AWS_SECRET,
    )
    size = Path(path).stat().st_size
    print(f"uploading {path} ({size/1e6:.1f} MB) -> s3://{bucket}/{key}")
    s3.upload_file(path, bucket, key, ExtraArgs={"ContentType": "video/mp4"})
    # generate a presigned URL (7 days) so user can view
    url = s3.generate_presigned_url(
        "get_object", Params={"Bucket": bucket, "Key": key},
        ExpiresIn=7 * 24 * 3600,
    )
    return url


def main():
    p, m, T = pick_walking_pkl()
    print(f"motion: {p}")
    print(f"frames: {T}  duration: {T/FPS:.1f}s  dof shape: {m['dof'].shape}")
    print("rendering...")
    if not render(m, OUT):
        print("render failed"); return 1
    print(f"rendered -> {OUT}  ({Path(OUT).stat().st_size/1e6:.1f} MB)")
    url = upload(OUT, BUCKET, S3_KEY)
    print()
    print("=" * 70)
    print("S3 URL (7-day presigned):")
    print(url)
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
