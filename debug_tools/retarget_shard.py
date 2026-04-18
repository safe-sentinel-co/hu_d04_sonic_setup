#!/usr/bin/env python3
"""Sharded BVH -> CSV retargeting for HU_D04.

Runs one shard of the full BVH corpus on one GPU. Launch 6 of these
(one per CUDA_VISIBLE_DEVICES=0..5) to use all GPUs in parallel.

Usage:
    CUDA_VISIBLE_DEVICES=0 python retarget_shard.py \
        --input  /workspace/bones-seed/soma_uniform/bvh \
        --output /workspace/hu_d04_motions/csv \
        --shard-idx 0 --shard-count 6 \
        --batch-size 10
"""
import argparse
import pathlib
import sys
import time
import os

# Silence Warp's startup banner per process
os.environ.setdefault("WARP_SILENT_IMPORT", "1")

import warp as wp
import newton

import soma_retargeter.assets.bvh as bvh_utils
import soma_retargeter.assets.csv as csv_utils
import soma_retargeter.pipelines.utils as pipeline_utils
import soma_retargeter.utils.io_utils as io_utils
from soma_retargeter.animation.animation_buffer import AnimationBuffer
from soma_retargeter.utils.space_conversion_utils import (
    SpaceConverter, get_facing_direction_type_from_str,
)

from tqdm import trange


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="BVH root (rglob *.bvh)")
    p.add_argument("--output", required=True, help="CSV output root")
    p.add_argument("--shard-idx", type=int, required=True)
    p.add_argument("--shard-count", type=int, required=True)
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--retarget-source", default="soma")
    p.add_argument("--retarget-target", default="limx_hu_d04")
    p.add_argument("--facing-direction", default="Mujoco")
    p.add_argument("--device", default="cuda:0",
                   help="Warp device (use cuda:0 after setting CUDA_VISIBLE_DEVICES)")
    p.add_argument("--target-fps", type=float, default=0.0,
                   help="If > 0, decimate BVH animation to this fps before IK")
    p.add_argument("--skip-existing", action="store_true", default=True,
                   help="Skip BVHs that already have a CSV output")
    p.add_argument("--log", default=None, help="Log file path")
    return p.parse_args()


def decimate_animation(anim, target_fps: float):
    """Stride an animation down to target_fps via frame-step decimation."""
    if target_fps <= 0 or target_fps >= anim.sample_rate:
        return anim
    stride = max(1, int(round(anim.sample_rate / target_fps)))
    if stride == 1:
        return anim
    new_lt = anim.local_transforms[::stride].copy()
    return AnimationBuffer(
        skeleton=anim.skeleton,
        num_frames=new_lt.shape[0],
        sample_rate=anim.sample_rate / stride,
        local_transforms=new_lt,
    )


def shard_prefix(args):
    return f"[shard {args.shard_idx}/{args.shard_count}]"


def log(msg, args, fh=None):
    line = f"{shard_prefix(args)} {msg}"
    print(line, flush=True)
    if fh is not None:
        fh.write(line + "\n"); fh.flush()


def main():
    args = parse_args()
    log_fh = open(args.log, "a") if args.log else None
    log(f"device={args.device} pid={os.getpid()} cuda_vis={os.environ.get('CUDA_VISIBLE_DEVICES','?')}", args, log_fh)

    import_path = pathlib.Path(args.input)
    export_path = pathlib.Path(args.output)
    if not import_path.is_dir():
        log(f"ERROR: input dir not found: {import_path}", args, log_fh)
        sys.exit(2)
    export_path.mkdir(parents=True, exist_ok=True)

    # Select CSV config
    if args.retarget_target == "limx_hu_d04":
        from soma_retargeter.assets.csv import LimXHUD04_31DOF_CSVConfig
        csv_config = LimXHUD04_31DOF_CSVConfig()
    else:
        csv_config = csv_utils.UnitreeG129DOF_CSVConfig()

    # Discover BVHs, sort by size (largest first) for even distribution
    all_bvhs = sorted(import_path.rglob("*.bvh"),
                      key=lambda p: p.stat().st_size, reverse=True)
    log(f"found {len(all_bvhs)} BVHs in {import_path}", args, log_fh)

    # Shard by round-robin on the size-sorted list -> similar total work per shard
    my_bvhs = [b for i, b in enumerate(all_bvhs)
               if i % args.shard_count == args.shard_idx]
    log(f"my shard has {len(my_bvhs)} BVHs", args, log_fh)

    # Skip ones already done
    pending = []
    skipped = 0
    for b in my_bvhs:
        rel = b.relative_to(import_path).with_suffix(".csv")
        dst = export_path / rel
        if args.skip_existing and dst.exists() and dst.stat().st_size > 0:
            skipped += 1
            continue
        pending.append(b)
    log(f"skip-existing: pending={len(pending)} skipped={skipped}", args, log_fh)
    if not pending:
        log("nothing to do, exiting", args, log_fh); return

    converter = SpaceConverter(get_facing_direction_type_from_str(args.facing_direction))
    bvh_tx_converter = converter.transform(wp.transform_identity())

    # Build reference skeleton from first BVH
    bvh_importer = bvh_utils.BVHImporter()
    bvh_skeleton, _ = bvh_importer.create_skeleton(pending[0])
    expected_num_joints = bvh_skeleton.num_joints

    # Build retarget pipeline
    import soma_retargeter.pipelines.newton_pipeline as newton_pipeline
    with wp.ScopedDevice(args.device):
        retarget_pipeline = newton_pipeline.NewtonPipeline(
            bvh_skeleton, args.retarget_source, args.retarget_target
        )

        # Batch and process
        batches = [pending[i:i + args.batch_size]
                   for i in range(0, len(pending), args.batch_size)]

        nb_done = 0
        t0 = time.time()
        for bi, batch in enumerate(batches):
            log(f"batch {bi+1}/{len(batches)} ({len(batch)} BVHs)", args, log_fh)
            animations = []
            valid_batch = []
            for file_path in batch:
                try:
                    _, animation = bvh_utils.load_bvh(file_path, bvh_skeleton)
                    if animation.skeleton.num_joints != expected_num_joints:
                        log(f"SKIP (joint mismatch): {file_path}", args, log_fh)
                        continue
                    animation = decimate_animation(animation, args.target_fps)
                    animations.append(animation)
                    valid_batch.append(file_path)
                except Exception as e:
                    log(f"SKIP (load error): {file_path} :: {e}", args, log_fh)
                    continue

            if not animations:
                continue

            try:
                retarget_pipeline.clear()
                retarget_pipeline.add_input_motions(
                    animations, [bvh_tx_converter] * len(animations), True
                )
                csv_buffers = retarget_pipeline.execute()
                assert len(csv_buffers) == len(animations), "length mismatch"
            except Exception as e:
                log(f"batch failed, falling back to one-by-one: {e}", args, log_fh)
                # Fallback: process one at a time to isolate bad files
                for fp, anim in zip(valid_batch, animations):
                    try:
                        retarget_pipeline.clear()
                        retarget_pipeline.add_input_motions(
                            [anim], [bvh_tx_converter], True
                        )
                        cb = retarget_pipeline.execute()
                        if len(cb) == 1:
                            rel = fp.relative_to(import_path).with_suffix(".csv")
                            dst = export_path / rel
                            dst.parent.mkdir(parents=True, exist_ok=True)
                            csv_utils.save_csv(dst, cb[0], csv_config)
                            nb_done += 1
                    except Exception as ie:
                        log(f"FAIL: {fp} :: {ie}", args, log_fh)
                continue

            for i in range(len(csv_buffers)):
                rel = valid_batch[i].relative_to(import_path).with_suffix(".csv")
                dst = export_path / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                csv_utils.save_csv(dst, csv_buffers[i], csv_config)
                nb_done += 1

            elapsed = time.time() - t0
            rate = nb_done / max(elapsed, 1e-6)
            eta_s = (len(pending) - nb_done) / max(rate, 1e-6)
            log(f"progress: {nb_done}/{len(pending)} rate={rate:.2f}/s eta={int(eta_s//60)}m",
                args, log_fh)

        log(f"DONE: {nb_done} motions retargeted in {int((time.time()-t0)//60)}m", args, log_fh)

    if log_fh:
        log_fh.close()


if __name__ == "__main__":
    main()
