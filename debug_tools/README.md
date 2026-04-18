# HU_D04 retargeting debug tools

Scripts used during the HU_D04 retargeter debugging session (2026-04-17/18).
All of these expect:
- `soma_env` (Python 3.12 + soma-retargeter) at `/workspace/soma_env/`
- The patched repos applied per `scripts/setup_all.sh`
- AWS credentials in env (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`) when uploading

| script | purpose |
|--------|---------|
| `diagnose_alignment.py` | Render SOMA T-pose vs HU_D04 rest-pose side-by-side (front/side/top). Quick static check that the coordinate frame + offsets are correct. |
| `iter_debug.py` | Fast iteration harness. Truncates a BVH to `TRUNC_SEC=10`s, retargets, renders side-by-side MP4, uploads. ~60-90 s round-trip. |
| `retarget_shard.py` | Per-GPU sharded BVH→CSV retargeter (supports `--target-fps` decimation). |
| `pipeline.sh` | Full BVH → CSV → PKL → filtered pipeline. Uses N sharded retargeters on K GPUs. |
| `side_by_side.py` | Standalone side-by-side MP4 renderer (source BVH stick figure + retargeted robot). |
| `render_bvh.py` | Stick-figure render of just the source BVH (matplotlib, headless). |
| `render_and_upload.py` | Render a PKL with MuJoCo OSMesa and upload to S3. |
| `local_viewer.py` | Interactive MuJoCo viewer for a PKL (needs a display — run on laptop after downloading). |
| `validate_output.py` | Verify the full PKL dataset has correct schema, shapes, no NaNs, expected session coverage. |

## Critical bugs fixed (see main README CHANGELOG)

1. **SpaceConverter (MUJOCO)**: was `90° around +X`, which maps SOMA's lateral `+X` to MuJoCo forward — caused left/right swap and walking-axis mismatch. Corrected to `120° around (1,1,1)` = axis cycle `X→Y→Z→X`.
2. **Joint offsets copy-pasted from G1**: every entry in `soma_to_hu_d04_scaler_config.json:joint_offsets` was byte-for-byte identical to G1's. Recomputed per-body by aligning SOMA T-pose bone quats with HU_D04 rest-pose (arms-folded-forward, elbow -90°) body quats.
3. **ik_map missing intermediate spine**: only `Hips` + `Chest` were mapped, leaving HU_D04's three waist joints (yaw/roll/pitch) under-constrained. Added `Spine1 → waist_yaw_link` and `Spine2 → waist_roll_link`, which eliminates the `waist_pitch` saturation we were seeing (was 90 % pinned at +45° → now 23 %).

## Typical debug loop

```bash
source /workspace/soma_env/bin/activate
export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=...
export CUDA_VISIBLE_DEVICES=0

# 1. Static alignment check
python diagnose_alignment.py       # produces tpose_alignment.png + .json + uploads

# 2. Retarget a short motion and watch side-by-side
TRUNC_SEC=10 python iter_debug.py  # uses default walk; pass BVH path to override

# 3. Full run (all GPUs)
SHARDS=12 GPUS=6 BATCH=10 ./pipeline.sh all
```
