#!/usr/bin/env python3
"""Diagnostic: show SOMA T-pose joint positions vs HU_D04 T-pose body positions.

Produces a human-readable table + a PNG showing both skeletons side-by-side
at T-pose, with joint correspondences marked. Upload to S3 for inspection.

What "wrong" looks like:
  - Large position distance between SOMA joint and its mapped HU_D04 body
  - Arms pointing different directions
  - Left/right swapped
"""
import os
os.environ["MUJOCO_GL"] = "osmesa"
os.environ.setdefault("MPLBACKEND", "Agg")
import sys, json, time
import numpy as np, cv2, boto3, mujoco
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

import warp as wp
import soma_retargeter.assets.bvh as bvh_utils

MJCF = "/workspace/GR00T-WholeBodyControl/gear_sonic/data/assets/robot_description/mjcf/hu_d04.xml"
OUT_PNG = "/workspace/tpose_alignment.png"
OUT_JSON = "/workspace/tpose_alignment.json"

AWS_KEY = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET = os.environ["AWS_SECRET_ACCESS_KEY"]
BUCKET = "safesentinel-inc"

# SOMA ↔ HU_D04 body mapping (from ik_map in retargeter_config)
MAPPING = [
    ("Hips",         "base_link"),
    ("Chest",        "waist_pitch_link"),
    ("Neck1",        "head_yaw_link"),
    ("LeftArm",      "left_shoulder_roll_link"),
    ("LeftForeArm",  "left_elbow_link"),
    ("LeftHand",     "left_wrist_roll_link"),
    ("RightArm",     "right_shoulder_roll_link"),
    ("RightForeArm", "right_elbow_link"),
    ("RightHand",    "right_wrist_roll_link"),
    ("LeftLeg",      "left_hip_roll_link"),
    ("LeftShin",     "left_knee_link"),
    ("LeftFoot",     "left_ankle_roll_link"),
    ("RightLeg",     "right_hip_roll_link"),
    ("RightShin",    "right_knee_link"),
    ("RightFoot",    "right_ankle_roll_link"),
]


def get_soma_tpose(apply_spaceconverter=True, scale=0.80):
    """Return SOMA T-pose joint positions.
    If apply_spaceconverter is True, rotates by the current SpaceConverter
    (wp.quat_from_axis_angle((1,0,0), 90°)) so positions are in the "MuJoCo frame".
    """
    skel, anim = bvh_utils.load_bvh(
        "/workspace/soma-retargeter/soma_retargeter/configs/soma/soma_zero_frame0.bvh")
    tx = anim.compute_global_transforms(0)  # (J, 7): xyz + xyzw
    positions = tx[:, :3].copy()
    if apply_spaceconverter:
        R_conv = R.from_quat([0.5, 0.5, 0.5, 0.5])
        positions = positions @ R_conv.as_matrix().T
    # Scale human positions to roughly robot-sized
    hips_pos = positions[skel.joint_index("Hips")].copy()
    positions = (positions - hips_pos) * scale + hips_pos
    return skel, positions


def get_robot_tpose():
    """Return HU_D04 REST pose body positions (arms down at sides).
    The SOMA zero-pose reference BVH also has arms-down, so compare zero-to-zero.
    """
    model = mujoco.MjModel.from_xml_path(MJCF)
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    def _sj(n, d):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        if jid >= 0: data.qpos[model.jnt_qposadr[jid]] = np.deg2rad(d)
    _sj("left_elbow_joint", -90); _sj("right_elbow_joint", -90)
    mujoco.mj_forward(model, data)
    body_pos = {}
    for _, robot_name in MAPPING:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, robot_name)
        body_pos[robot_name] = data.xpos[bid].copy()
    return body_pos, model, data


def build_table(skel, soma_pos, robot_pos):
    """Return list of dict rows."""
    hips_soma = soma_pos[skel.joint_index("Hips")]
    base_robot = robot_pos["base_link"]
    rows = []
    for soma_name, robot_name in MAPPING:
        if soma_name not in skel.joint_names:
            continue
        sp = soma_pos[skel.joint_index(soma_name)] - hips_soma
        rp = robot_pos[robot_name] - base_robot
        dist = float(np.linalg.norm(sp - rp))
        rows.append({
            "soma_joint": soma_name,
            "robot_body": robot_name,
            "soma_pos_rel_hips": sp.round(3).tolist(),
            "robot_pos_rel_base": rp.round(3).tolist(),
            "distance_m": round(dist, 3),
        })
    return rows


def render_png(skel, soma_pos, robot_pos, rows, out_path):
    """Three views: (1) overlay, (2) front, (3) side. Both skeletons in every plot
    so misalignment is obvious.
    """
    fig = plt.figure(figsize=(18, 8), dpi=110)
    parents = [int(p) for p in skel.parent_indices]

    hips = soma_pos[skel.joint_index("Hips")]
    base = robot_pos["base_link"]
    soma = soma_pos - hips + np.array([0, 0, 1.0])
    CHAIN = [
        ("base_link","waist_pitch_link"), ("waist_pitch_link","head_yaw_link"),
        ("waist_pitch_link","left_shoulder_roll_link"),("left_shoulder_roll_link","left_elbow_link"),("left_elbow_link","left_wrist_roll_link"),
        ("waist_pitch_link","right_shoulder_roll_link"),("right_shoulder_roll_link","right_elbow_link"),("right_elbow_link","right_wrist_roll_link"),
        ("base_link","left_hip_roll_link"),("left_hip_roll_link","left_knee_link"),("left_knee_link","left_ankle_roll_link"),
        ("base_link","right_hip_roll_link"),("right_hip_roll_link","right_knee_link"),("right_knee_link","right_ankle_roll_link"),
    ]
    robot_xyz = {k: (robot_pos[k] - base + np.array([0,0,1.0])) for k in robot_pos}

    def draw_both(ax):
        # SOMA (blue)
        for j in range(skel.num_joints):
            pj = parents[j]
            if pj < 0: continue
            ax.plot([soma[pj,0], soma[j,0]], [soma[pj,1], soma[j,1]], [soma[pj,2], soma[j,2]],
                    color="royalblue", linewidth=1.6, alpha=0.85, label="_nolegend_")
        for soma_name, robot_name in MAPPING:
            if soma_name not in skel.joint_names: continue
            p = soma[skel.joint_index(soma_name)]
            ax.scatter([p[0]], [p[1]], [p[2]], s=40, c="blue", zorder=6, marker='o')
        # HU_D04 (green)
        for a, b in CHAIN:
            pa, pb = robot_xyz[a], robot_xyz[b]
            ax.plot([pa[0],pb[0]], [pa[1],pb[1]], [pa[2],pb[2]],
                    color="forestgreen", linewidth=1.6, alpha=0.85)
        for _, robot_name in MAPPING:
            p = robot_xyz[robot_name]
            ax.scatter([p[0]], [p[1]], [p[2]], s=40, c="green", zorder=6, marker='^')
        # Lines showing the mismatch
        for soma_name, robot_name in MAPPING:
            if soma_name not in skel.joint_names: continue
            ps = soma[skel.joint_index(soma_name)]
            pr = robot_xyz[robot_name]
            ax.plot([ps[0], pr[0]], [ps[1], pr[1]], [ps[2], pr[2]],
                    color="red", linewidth=0.8, linestyle=":", alpha=0.7)

    # --- FRONT view ---
    ax1 = fig.add_subplot(131, projection="3d")
    draw_both(ax1)
    ax1.view_init(elev=0, azim=0)    # look down +X
    ax1.set_title("FRONT view (from +X)")
    ax1.set_xlabel("X"); ax1.set_ylabel("Y (left→+)"); ax1.set_zlabel("Z (up)")
    ax1.set_xlim(-0.5,0.5); ax1.set_ylim(-0.9,0.9); ax1.set_zlim(0,2.0)
    ax1.set_xticks([]); ax1.set_box_aspect([0.3,1.8,2.0])

    # --- SIDE view ---
    ax2 = fig.add_subplot(132, projection="3d")
    draw_both(ax2)
    ax2.view_init(elev=0, azim=-90)  # look down +Y
    ax2.set_title("SIDE view (from +Y)")
    ax2.set_xlabel("X (fwd→+)"); ax2.set_ylabel("Y"); ax2.set_zlabel("Z (up)")
    ax2.set_xlim(-0.6,0.6); ax2.set_ylim(-0.3,0.3); ax2.set_zlim(0,2.0)
    ax2.set_yticks([]); ax2.set_box_aspect([1.8,0.3,2.0])

    # --- TOP view ---
    ax3 = fig.add_subplot(133, projection="3d")
    draw_both(ax3)
    ax3.view_init(elev=89, azim=-90)
    ax3.set_title("TOP view (from +Z)")
    ax3.set_xlabel("X (fwd→+)"); ax3.set_ylabel("Y (left→+)"); ax3.set_zlabel("Z")
    ax3.set_xlim(-0.6,0.6); ax3.set_ylim(-0.9,0.9); ax3.set_zlim(0,2.0)
    ax3.set_zticks([]); ax3.set_box_aspect([1.2,1.8,0.3])

    # Legend (manually, via proxy lines)
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0],[0], color="royalblue", lw=2, label="SOMA human"),
        Line2D([0],[0], color="forestgreen", lw=2, label="HU_D04 robot"),
        Line2D([0],[0], color="red", lw=1, linestyle=":", label="mismatch"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.98))

    # Table
    txt = f"{'soma':14s} {'robot':28s} {'|diff|':>7s}\n" + "-"*56 + "\n"
    for r in rows:
        txt += f"{r['soma_joint']:14s} {r['robot_body']:28s} {r['distance_m']:>7.3f} m\n"
    fig.text(0.01, 0.01, txt, fontsize=7, family="monospace", verticalalignment="bottom")

    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def upload(png, json_path):
    s3 = boto3.client("s3", aws_access_key_id=AWS_KEY, aws_secret_access_key=AWS_SECRET)
    for fp, key, ct in [(png, "hu_d04_retargeting/tpose_alignment.png", "image/png"),
                        (json_path, "hu_d04_retargeting/tpose_alignment.json", "application/json")]:
        s3.upload_file(fp, BUCKET, key, ExtraArgs={"ContentType": ct})
    urls = {
        "png": s3.generate_presigned_url("get_object",
            Params={"Bucket": BUCKET, "Key": "hu_d04_retargeting/tpose_alignment.png"},
            ExpiresIn=7*24*3600),
        "json": s3.generate_presigned_url("get_object",
            Params={"Bucket": BUCKET, "Key": "hu_d04_retargeting/tpose_alignment.json"},
            ExpiresIn=7*24*3600),
    }
    return urls


def main():
    skel, soma_pos = get_soma_tpose()
    robot_pos, _, _ = get_robot_tpose()
    rows = build_table(skel, soma_pos, robot_pos)
    print(f"{'soma':14s} {'robot':28s} {'soma_rel_hips':>24s} {'robot_rel_base':>24s} {'|diff|':>7s}")
    print("-" * 110)
    for r in rows:
        print(f"{r['soma_joint']:14s} {r['robot_body']:28s} {str(r['soma_pos_rel_hips']):>24s} {str(r['robot_pos_rel_base']):>24s} {r['distance_m']:>7.3f}")
    json.dump(rows, open(OUT_JSON, "w"), indent=2)
    render_png(skel, soma_pos, robot_pos, rows, OUT_PNG)
    urls = upload(OUT_PNG, OUT_JSON)
    print("\nPNG URL:"); print(urls["png"])
    print("\nJSON URL:"); print(urls["json"])


if __name__ == "__main__":
    main()
