#!/usr/bin/env python3
"""Render source BVH (left) and retargeted HU_D04 (right) side-by-side to MP4."""
import os
os.environ["MUJOCO_GL"] = "osmesa"
os.environ.setdefault("MPLBACKEND", "Agg")
import sys, glob, numpy as np, cv2, boto3, joblib
import matplotlib.pyplot as plt
import warp as wp
import soma_retargeter.assets.bvh as bvh_utils
import mujoco

BVH = os.environ.get("BVH", "/workspace/bones-seed/soma_uniform/bvh/240918/injured_torso_grab_walk_ff_180_002__A549.bvh")
PKL = os.environ.get("PKL", "")
OUT = os.environ.get("OUT", "/workspace/side_by_side.mp4")
MJCF = "/workspace/GR00T-WholeBodyControl/gear_sonic/data/assets/robot_description/mjcf/hu_d04.xml"
W, H = 640, 480
FPS = 30

AWS_KEY = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET = os.environ["AWS_SECRET_ACCESS_KEY"]
BUCKET = os.environ.get("S3_BUCKET", "safesentinel-inc")
S3_KEY = os.environ.get("S3_KEY", "hu_d04_retargeting/side_by_side.mp4")


def render_bvh_frames(bvh_path, stride=4):
    """Return list of (joint_positions, parents) per frame."""
    skel, anim = bvh_utils.load_bvh(bvh_path)
    frames = list(range(0, anim.num_frames, stride))
    parents = [int(p) for p in skel.parent_indices]
    gpos = np.zeros((len(frames), skel.num_joints, 3), dtype=np.float32)
    for i, f in enumerate(frames):
        tx = anim.compute_global_transforms(f)
        gpos[i] = tx[:, :3]
    return gpos, parents, anim.sample_rate / stride


def render_pkl_frames(pkl_path):
    m = next(iter(joblib.load(pkl_path).values()))
    return m


def main():
    print(f"BVH: {BVH}")
    bvh_gpos, parents, bvh_fps = render_bvh_frames(BVH)
    print(f"  bvh frames: {bvh_gpos.shape[0]} @ {bvh_fps} fps")

    pkl_path = PKL
    if not pkl_path:
        # Pick the matching test_v6 PKL (if exists) else a walking pkl
        pkl_path = glob.glob('/workspace/hu_d04_motions/_test_v6/pkl_out/**/*.pkl', recursive=True)
        if not pkl_path:
            pkl_path = glob.glob('/workspace/hu_d04_motions/robot/**/injured_torso_grab_walk_ff_180_002__A549.pkl', recursive=True)
        pkl_path = pkl_path[0]
    print(f"PKL: {pkl_path}")
    m = render_pkl_frames(pkl_path)
    T_pkl = m["dof"].shape[0]
    T = min(bvh_gpos.shape[0], T_pkl)
    print(f"  aligned frames: {T}")

    # MuJoCo setup for right panel
    model = mujoco.MjModel.from_xml_path(MJCF)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=H, width=W)
    cam = mujoco.MjvCamera()
    cam.distance = 3.0; cam.azimuth = 90; cam.elevation = -10

    # matplotlib figure for left panel
    fig = plt.figure(figsize=(W/100, H/100), dpi=100)
    ax = fig.add_subplot(111, projection="3d")

    out = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W*2, H))
    if not out.isOpened():
        raise RuntimeError("VideoWriter open failed")

    hip_idx = 1  # Hips in SOMA
    for t in range(T):
        # -------- left panel: source BVH stick figure --------
        ax.cla()
        pts = bvh_gpos[t]
        root = pts[hip_idx]
        radius = 1.2
        for j in range(pts.shape[0]):
            pj = parents[j]
            if pj < 0: continue
            ax.plot([pts[pj,0], pts[j,0]],
                    [pts[pj,2], pts[j,2]],
                    [pts[pj,1], pts[j,1]],
                    color="royalblue", linewidth=1.4)
        ax.scatter(pts[:,0], pts[:,2], pts[:,1], s=4, c="crimson")
        ax.set_xlim(root[0]-radius, root[0]+radius)
        ax.set_ylim(root[2]-radius, root[2]+radius)
        ax.set_zlim(0, 2.2)
        ax.view_init(elev=10, azim=-75)
        ax.set_title(f"SOURCE (SOMA human)   frame {t}")
        ax.set_xlabel("X"); ax.set_ylabel("Z(fwd)"); ax.set_zlabel("Y(up)")
        fig.canvas.draw()
        left_img = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        left_img = left_img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        left_img = left_img[:, :, [1,2,3]]
        left_img = cv2.cvtColor(left_img, cv2.COLOR_RGB2BGR)
        if left_img.shape[:2] != (H, W):
            left_img = cv2.resize(left_img, (W, H))

        # -------- right panel: HU_D04 in MuJoCo --------
        data.qpos[:3] = m["root_trans_offset"][t]
        data.qpos[3:7] = m["root_rot"][t]  # converter stores wxyz
        data.qpos[7:] = m["dof"][t]
        mujoco.mj_forward(model, data)
        cam.lookat[:] = data.qpos[:3]
        renderer.update_scene(data, camera=cam)
        right_img = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
        # overlay caption
        cv2.putText(right_img, f"RETARGET (HU_D04) frame {t}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)

        combined = np.hstack([left_img, right_img])
        out.write(combined)
        if t in (0, T//4, T//2, 3*T//4, T-1):
            print(f"  frame {t}/{T}")

    out.release()
    plt.close(fig)

    size = os.path.getsize(OUT) / 1e6
    print(f"rendered → {OUT} ({size:.1f} MB)")
    s3 = boto3.client("s3", aws_access_key_id=AWS_KEY, aws_secret_access_key=AWS_SECRET)
    s3.upload_file(OUT, BUCKET, S3_KEY, ExtraArgs={"ContentType": "video/mp4"})
    url = s3.generate_presigned_url("get_object",
        Params={"Bucket": BUCKET, "Key": S3_KEY}, ExpiresIn=7*24*3600)
    print("\n7-day URL:"); print(url)


if __name__ == "__main__":
    sys.exit(main())
