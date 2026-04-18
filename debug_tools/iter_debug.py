#!/usr/bin/env python3
"""Fast iteration harness for retargeter debugging.

Retargets a truncated walk motion (first 3 seconds), renders side-by-side,
uploads to S3, prints URL. Total wall time ≈ 25-40 seconds.

Usage:
    python iter_debug.py [bvh_path] [suffix]
    # default: uses a confident walk, suffix = 'iter'
"""
import os, sys, time, glob, json
os.environ["MUJOCO_GL"] = "osmesa"
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np, cv2, boto3, joblib, mujoco
import matplotlib.pyplot as plt

import warp as wp
import soma_retargeter.assets.bvh as bvh_utils
import soma_retargeter.assets.csv as csv_utils
from soma_retargeter.animation.animation_buffer import AnimationBuffer
from soma_retargeter.utils.space_conversion_utils import (
    SpaceConverter, get_facing_direction_type_from_str,
)
import soma_retargeter.pipelines.newton_pipeline as newton_pipeline

BVH_DEFAULT = "/workspace/bones-seed/soma_uniform/bvh/210531/walk_forward_confident_impro_003__A001_M.bvh"
MJCF = "/workspace/GR00T-WholeBodyControl/gear_sonic/data/assets/robot_description/mjcf/hu_d04.xml"
TRUNC_SEC = float(os.environ.get("TRUNC_SEC", "10"))   # default 10 sec clips
TARGET_FPS = 30

AWS_KEY = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET = os.environ["AWS_SECRET_ACCESS_KEY"]
BUCKET = "safesentinel-inc"


def truncate_and_decimate(bvh_path, trunc_sec, target_fps):
    skel, anim = bvh_utils.load_bvh(bvh_path)
    src_fps = anim.sample_rate
    stride = max(1, int(round(src_fps / target_fps)))
    max_src_frames = int(trunc_sec * src_fps)
    max_src_frames = min(max_src_frames, anim.num_frames)
    new_lt = anim.local_transforms[:max_src_frames:stride].copy()
    new = AnimationBuffer(
        skeleton=skel, num_frames=new_lt.shape[0],
        sample_rate=src_fps / stride, local_transforms=new_lt,
    )
    return skel, new


def retarget(bvh_path, trunc_sec):
    t0 = time.time()
    skel, anim = truncate_and_decimate(bvh_path, trunc_sec, TARGET_FPS)
    print(f"[{time.time()-t0:.1f}s] loaded+trunc {anim.num_frames} frames @ {anim.sample_rate} fps")

    with wp.ScopedDevice("cuda:0"):
        converter = SpaceConverter(get_facing_direction_type_from_str("Mujoco"))
        tx = converter.transform(wp.transform_identity())
        pipe = newton_pipeline.NewtonPipeline(skel, "soma", "limx_hu_d04")
        print(f"[{time.time()-t0:.1f}s] built pipeline")
        pipe.clear()
        pipe.add_input_motions([anim], [tx], True)
        csv_bufs = pipe.execute()
        print(f"[{time.time()-t0:.1f}s] IK done")
    # save CSV → convert → PKL
    from soma_retargeter.assets.csv import LimXHUD04_31DOF_CSVConfig
    csv_cfg = LimXHUD04_31DOF_CSVConfig()
    csv_dir = "/tmp/iter_csv"; os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, "test.csv")
    csv_utils.save_csv(csv_path, csv_bufs[0], csv_cfg)
    pkl_dir = "/tmp/iter_pkl"; os.makedirs(pkl_dir, exist_ok=True)
    import subprocess
    subprocess.run(["python", "/workspace/GR00T-WholeBodyControl/gear_sonic/data_process/convert_hu_d04_csv_to_motion_lib.py",
                    "--input", csv_dir, "--output", pkl_dir,
                    "--fps", "30", "--fps_source", "30", "--individual", "--num_workers", "1"],
                   capture_output=True, check=True)
    pkl = glob.glob(f"{pkl_dir}/**/*.pkl", recursive=True)[0]
    print(f"[{time.time()-t0:.1f}s] pkl ready: {pkl}")
    return pkl, skel, anim


def render_side_by_side(pkl, bvh_skel, bvh_anim, out_path):
    t0 = time.time()
    W, H = 640, 480
    fps = TARGET_FPS
    # source stick figure frames — apply the same SpaceConverter rotation
    # to the source so both panels share the same world frame (MuJoCo Z-up X-forward).
    # New SpaceConverter: 120° around (1,1,1)/sqrt(3), i.e. quat (0.5,0.5,0.5,0.5).
    from scipy.spatial.transform import Rotation as R_
    R_conv = R_.from_quat([0.5, 0.5, 0.5, 0.5]).as_matrix()

    J = bvh_skel.num_joints
    parents = [int(p) for p in bvh_skel.parent_indices]
    src_gpos = np.zeros((bvh_anim.num_frames, J, 3), dtype=np.float32)
    for i in range(bvh_anim.num_frames):
        raw = bvh_anim.compute_global_transforms(i)[:, :3]
        src_gpos[i] = raw @ R_conv.T

    m = next(iter(joblib.load(pkl).values()))
    T = min(bvh_anim.num_frames, m["dof"].shape[0])

    # robot renderer
    model = mujoco.MjModel.from_xml_path(MJCF)
    data = mujoco.MjData(model)
    r = mujoco.Renderer(model, height=H, width=W)
    # FRONT view: camera in front of robot facing it (robot's "face" visible)
    cam = mujoco.MjvCamera(); cam.distance = 3.0; cam.azimuth = 180; cam.elevation = -5

    fig = plt.figure(figsize=(W/100, H/100), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W*2, H))
    hip_i = bvh_skel.joint_index("Hips")
    for t in range(T):
        ax.cla()
        pts = src_gpos[t]  # already in MuJoCo frame (Z-up) after SpaceConverter rotation
        root = pts[hip_i]; radius = 1.2
        # Plot in MuJoCo frame axes: X=X, Y=Y (fwd/side), Z=up
        for j in range(J):
            pj = parents[j]
            if pj < 0: continue
            ax.plot([pts[pj,0], pts[j,0]], [pts[pj,1], pts[j,1]], [pts[pj,2], pts[j,2]],
                    color="royalblue", linewidth=1.4)
        ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=4, c="crimson")
        ax.set_xlim(root[0]-radius, root[0]+radius)
        ax.set_ylim(root[1]-radius, root[1]+radius); ax.set_zlim(0, 2.2)
        # Match MuJoCo FRONT view (camera at +X looking toward -X)
        ax.view_init(elev=5, azim=0)
        ax.set_xlabel("X"); ax.set_ylabel("Y(left)"); ax.set_zlabel("Z(up)")
        ax.set_title(f"SOURCE (front view) frame {t}")
        fig.canvas.draw()
        li = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        li = li.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, [1,2,3]]
        li = cv2.cvtColor(li, cv2.COLOR_RGB2BGR)
        if li.shape[:2] != (H, W): li = cv2.resize(li, (W, H))

        data.qpos[:3] = m["root_trans_offset"][t]
        data.qpos[3:7] = m["root_rot"][t]
        data.qpos[7:] = m["dof"][t]
        mujoco.mj_forward(model, data)
        cam.lookat[:] = data.qpos[:3]
        r.update_scene(data, camera=cam)
        ri = cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR)
        cv2.putText(ri, f"RETARGET frame {t}", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
        out.write(np.hstack([li, ri]))
    out.release(); plt.close(fig)
    print(f"[{time.time()-t0:.1f}s] rendered to {out_path}")


def upload(path, pkl_path, suffix):
    s3 = boto3.client("s3", aws_access_key_id=AWS_KEY, aws_secret_access_key=AWS_SECRET)
    # named (versioned) + latest (always-current)
    for k in (f"hu_d04_retargeting/iter_{suffix}.mp4",
              "hu_d04_retargeting/iter_latest.mp4"):
        s3.upload_file(path, BUCKET, k, ExtraArgs={"ContentType":"video/mp4"})
    for k in (f"hu_d04_retargeting/iter_{suffix}.pkl",
              "hu_d04_retargeting/iter_latest.pkl"):
        s3.upload_file(pkl_path, BUCKET, k, ExtraArgs={"ContentType":"application/octet-stream"})
    url = s3.generate_presigned_url("get_object",
        Params={"Bucket": BUCKET, "Key": "hu_d04_retargeting/iter_latest.mp4"}, ExpiresIn=7*24*3600)
    return url, f"s3://{BUCKET}/hu_d04_retargeting/iter_{suffix}.mp4"


def main():
    bvh = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else BVH_DEFAULT
    suffix = sys.argv[2] if len(sys.argv) > 2 else f"iter{int(time.time())%10000:04d}"
    t0 = time.time()
    pkl, skel, anim = retarget(bvh, TRUNC_SEC)
    out = f"/tmp/iter_{suffix}.mp4"
    render_side_by_side(pkl, skel, anim, out)
    url, s3path = upload(out, pkl, suffix)
    print()
    print("=" * 80)
    print(f"Total: {time.time()-t0:.1f}s")
    print(f"PKL (for local viewer): {pkl}")
    print(f"URL: {url}")
    print("=" * 80)


if __name__ == "__main__":
    main()
