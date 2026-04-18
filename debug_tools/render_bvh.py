#!/usr/bin/env python3
"""Render source BVH (SOMA human skeleton) as a stick-figure MP4."""
import os
os.environ.setdefault("MPLBACKEND", "Agg")
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
import boto3

import warp as wp
import soma_retargeter.assets.bvh as bvh_utils

BVH = os.environ.get("BVH", "/workspace/bones-seed/soma_uniform/bvh/240918/injured_torso_grab_walk_ff_180_002__A549.bvh")
OUT = os.environ.get("OUT", "/workspace/walk_source.mp4")
W, H = 640, 480
STRIDE = 4        # 120 fps BVH → 30 fps output
OUT_FPS = 30

AWS_KEY = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET = os.environ["AWS_SECRET_ACCESS_KEY"]
BUCKET = os.environ.get("S3_BUCKET", "safesentinel-inc")
S3_KEY = os.environ.get("S3_KEY", "hu_d04_retargeting/walk_source.mp4")


def main():
    print(f"loading BVH: {BVH}")
    skel, anim = bvh_utils.load_bvh(BVH)
    T_src = anim.num_frames
    parents = [int(p) for p in skel.parent_indices]
    J = skel.num_joints
    print(f"  frames: {T_src}  joints: {J}  source_fps: {anim.sample_rate}")

    # Stride-decimate
    frames = list(range(0, T_src, STRIDE))
    T = len(frames)
    gpos = np.zeros((T, J, 3), dtype=np.float32)
    for i, f in enumerate(frames):
        tx = anim.compute_global_transforms(f)   # (J, 7): xyz + xyzw
        gpos[i] = tx[:, :3]

    print(f"rendered frame count: {T}  output fps: {OUT_FPS}")
    print(f"root trajectory range:")
    root = gpos[:, 1]   # joint 1 is Hips; joint 0 is "Root" (all zeros marker)
    print(f"  x: [{root[:,0].min():.2f}, {root[:,0].max():.2f}]")
    print(f"  y: [{root[:,1].min():.2f}, {root[:,1].max():.2f}]")
    print(f"  z: [{root[:,2].min():.2f}, {root[:,2].max():.2f}]")

    # SOMA skeleton: y is vertical (BVH convention), but Newton re-expresses
    # as Y-up with Z-forward. Check: hip (joint 1) sits near y≈1 when standing.
    # Plot will use ax.view_init(elev=15) and set Y as vertical.

    fig = plt.figure(figsize=(W/100, H/100), dpi=100)
    ax = fig.add_subplot(111, projection="3d")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(OUT, fourcc, OUT_FPS, (W, H))
    if not vw.isOpened():
        raise RuntimeError("VideoWriter open failed")

    # camera-following (root) with fixed scale
    radius = 1.2  # m; SOMA skeleton is ~2m tall
    for t in range(T):
        ax.cla()
        pts = gpos[t]
        cx, cy, cz = root[t]
        for j in range(J):
            pj = parents[j]
            if pj < 0: continue
            ax.plot([pts[pj, 0], pts[j, 0]],
                    [pts[pj, 2], pts[j, 2]],   # swap so Z (forward) is plotted as X
                    [pts[pj, 1], pts[j, 1]],   # Y (up) is plotted as Z
                    color="royalblue", linewidth=1.4)
        ax.scatter(pts[:, 0], pts[:, 2], pts[:, 1], s=5, c="crimson")
        ax.set_xlim(cx - radius, cx + radius)
        ax.set_ylim(cz - radius, cz + radius)   # forward axis
        ax.set_zlim(0, 2.2)                     # height fixed so ground is visible
        ax.view_init(elev=10, azim=-75)
        ax.set_xlabel("X (lat)"); ax.set_ylabel("Z (fwd)"); ax.set_zlabel("Y (up)")
        ax.set_title(f"{os.path.basename(BVH)}\nframe {t}/{T}")
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        img = img[:, :, [1, 2, 3]]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if img.shape[:2] != (H, W):
            img = cv2.resize(img, (W, H))
        vw.write(img)
        if t in (0, T // 4, T // 2, 3 * T // 4, T - 1):
            print(f"  frame {t}/{T}  hip=({cx:.2f},{cy:.2f},{cz:.2f})")
    vw.release()
    plt.close(fig)

    size = os.path.getsize(OUT) / 1e6
    print(f"rendered → {OUT} ({size:.1f} MB)")

    s3 = boto3.client("s3", aws_access_key_id=AWS_KEY, aws_secret_access_key=AWS_SECRET)
    s3.upload_file(OUT, BUCKET, S3_KEY, ExtraArgs={"ContentType": "video/mp4"})
    url = s3.generate_presigned_url("get_object",
        Params={"Bucket": BUCKET, "Key": S3_KEY}, ExpiresIn=7 * 24 * 3600)
    print("\n7-day URL:"); print(url)


if __name__ == "__main__":
    sys.exit(main())
