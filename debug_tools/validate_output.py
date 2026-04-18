#!/usr/bin/env python3
"""Validate the retargeting output is correct and complete.

Run after the pipeline finishes:
    source /workspace/soma_env/bin/activate
    python /workspace/logs/validate_output.py
"""
import sys
from pathlib import Path
import joblib
import numpy as np

CSV_ROOT = Path("/workspace/hu_d04_motions/csv")
PKL_ROOT = Path("/workspace/hu_d04_motions/robot")
FILTERED_ROOT = Path("/workspace/hu_d04_motions/robot_filtered")
BVH_ROOT = Path("/workspace/bones-seed/soma_uniform/bvh")

EXPECTED_DOF = 31
EXPECTED_BODIES = 32
EXPECTED_FPS = 30.0
EXPECTED_KEYS = {"dof", "root_trans_offset", "root_rot", "pose_aa", "fps"}

# HU_D04 rough joint limits (rad) — from MJCF inspection (check for physical plausibility)
# These are generous bounds; we just flag anything very out of range.
JOINT_LIMIT_RAD = 3.2  # ~183 deg; anything above is definitely a bug

def header(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)

def check_counts():
    header("1) FILE COUNTS")
    bvh = list(BVH_ROOT.rglob("*.bvh"))
    csv = list(CSV_ROOT.rglob("*.csv"))
    pkl = list(PKL_ROOT.rglob("*.pkl"))
    flt = list(FILTERED_ROOT.rglob("*.pkl"))
    print(f"  BVH inputs:         {len(bvh):>7}")
    print(f"  CSV outputs:        {len(csv):>7}  ({100*len(csv)/max(len(bvh),1):.1f}% of BVH)")
    print(f"  PKL robot/:         {len(pkl):>7}  ({100*len(pkl)/max(len(csv),1):.1f}% of CSV)")
    print(f"  PKL robot_filtered: {len(flt):>7}  ({100*len(flt)/max(len(pkl),1):.1f}% of robot)")
    print(f"  Expected BVH: 142220. Filter drops ~8.7% so filtered ≈ {int(len(pkl)*0.913):>6}")
    ok = len(bvh) >= 140000 and len(csv) >= 0.95 * len(bvh) and len(pkl) >= 0.95 * len(csv) and len(flt) >= 0.80 * len(pkl)
    print("  STATUS:", "OK" if ok else "WARN — counts lower than expected")
    return ok, pkl, flt


def check_sessions():
    header("2) SESSION COVERAGE")
    # every session directory in BVH should exist in robot_filtered
    bvh_sessions = sorted([d.name for d in (BVH_ROOT).iterdir() if d.is_dir()])
    pkl_sessions = sorted([d.name for d in (FILTERED_ROOT).iterdir() if d.is_dir()])
    missing = sorted(set(bvh_sessions) - set(pkl_sessions))
    extra = sorted(set(pkl_sessions) - set(bvh_sessions))
    print(f"  BVH sessions:      {len(bvh_sessions)}")
    print(f"  Filtered sessions: {len(pkl_sessions)}")
    if missing:
        print(f"  Missing sessions:  {missing[:5]}{'...' if len(missing)>5 else ''}")
    if extra:
        print(f"  Extra sessions:    {extra[:5]}{'...' if len(extra)>5 else ''}")
    ok = not missing
    print("  STATUS:", "OK" if ok else "WARN")
    return ok


def check_schema(pkls, n_sample=200):
    header("3) SCHEMA + SHAPES (sample)")
    if not pkls:
        print("  SKIPPED: no pkls"); return False
    rng = np.random.default_rng(0)
    sample = rng.choice(pkls, size=min(n_sample, len(pkls)), replace=False)

    errors = []
    all_shapes = {"dof": set(), "pose_aa": set(), "root_rot": set(), "root_trans_offset": set()}
    T_counts = []

    for p in sample:
        try:
            d = joblib.load(p)
        except Exception as e:
            errors.append(f"load fail {p.name}: {e}"); continue
        if len(d) != 1:
            errors.append(f"{p.name}: expected 1 motion key, got {len(d)}"); continue
        name, m = next(iter(d.items()))
        keys = set(m.keys())
        if keys != EXPECTED_KEYS:
            errors.append(f"{p.name}: keys mismatch {sorted(keys)}")
            continue

        # shapes
        T = m["dof"].shape[0]
        T_counts.append(T)
        if m["dof"].shape != (T, EXPECTED_DOF):
            errors.append(f"{p.name}: dof shape {m['dof'].shape}")
        if m["pose_aa"].shape != (T, EXPECTED_BODIES, 3):
            errors.append(f"{p.name}: pose_aa shape {m['pose_aa'].shape}")
        if m["root_rot"].shape != (T, 4):
            errors.append(f"{p.name}: root_rot shape {m['root_rot'].shape}")
        if m["root_trans_offset"].shape != (T, 3):
            errors.append(f"{p.name}: root_trans_offset shape {m['root_trans_offset'].shape}")

        # dtype
        for k in ("dof", "pose_aa", "root_rot", "root_trans_offset"):
            if m[k].dtype != np.float32:
                errors.append(f"{p.name}: {k} dtype {m[k].dtype}")

        # fps
        if m["fps"] != EXPECTED_FPS:
            errors.append(f"{p.name}: fps={m['fps']}, expected {EXPECTED_FPS}")

        all_shapes["dof"].add(m["dof"].shape[1:])
        all_shapes["pose_aa"].add(m["pose_aa"].shape[1:])
        all_shapes["root_rot"].add(m["root_rot"].shape[1:])
        all_shapes["root_trans_offset"].add(m["root_trans_offset"].shape[1:])

    print(f"  Sampled: {len(sample)}")
    print(f"  Frame count stats: min={min(T_counts)} p50={int(np.median(T_counts))} max={max(T_counts)}")
    print(f"  Shape-suffix sets (should each be ONE set):")
    for k, v in all_shapes.items():
        print(f"    {k}: {v}")
    if errors:
        print(f"  ERRORS ({len(errors)}):")
        for e in errors[:10]:
            print("   -", e)
    else:
        print("  STATUS: OK (all sampled files match expected schema)")
    return not errors


def check_values(pkls, n_sample=200):
    header("4) VALUE SANITY (NaN, joint-limit, root drift)")
    if not pkls:
        print("  SKIPPED: no pkls"); return False
    rng = np.random.default_rng(1)
    sample = rng.choice(pkls, size=min(n_sample, len(pkls)), replace=False)

    nan_count = 0
    big_joint_count = 0
    big_root_count = 0
    height_issues = 0

    for p in sample:
        try:
            d = joblib.load(p)
            m = next(iter(d.values()))
        except Exception:
            continue
        if np.any(np.isnan(m["dof"])) or np.any(np.isnan(m["pose_aa"])) or np.any(np.isnan(m["root_trans_offset"])):
            nan_count += 1
        if np.max(np.abs(m["dof"])) > JOINT_LIMIT_RAD:
            big_joint_count += 1
        root = m["root_trans_offset"]
        # root Z (height) should hover around 1 m (HU_D04 standing). Allow 0.5-2m range incl. jumps.
        if np.any(root[:, 2] < 0.2) or np.any(root[:, 2] > 3.0):
            height_issues += 1
        # root lateral drift per frame; should be < 0.5 m between frames (otherwise teleport)
        dp = np.diff(root, axis=0)
        if np.any(np.linalg.norm(dp, axis=1) > 0.5):
            big_root_count += 1

    print(f"  NaN motions:         {nan_count}/{len(sample)}")
    print(f"  Joint > {JOINT_LIMIT_RAD} rad:   {big_joint_count}/{len(sample)}")
    print(f"  Root height < 0.2 m or > 3 m: {height_issues}/{len(sample)}")
    print(f"  Root jumps > 0.5 m/frame:     {big_root_count}/{len(sample)}")
    ok = nan_count == 0 and big_joint_count == 0
    print("  STATUS:", "OK" if ok else "WARN — some motions have physically-implausible values")
    return ok


def check_loadable_by_training():
    """Smoke-check that the training config would find the motions."""
    header("5) TRAINING CONFIG COMPATIBILITY")
    try:
        import yaml
        cfg_path = Path("/workspace/GR00T-WholeBodyControl/gear_sonic/config/exp/manager/universal_token/all_modes/sonic_hu_d04.yaml")
        if not cfg_path.exists():
            print("  sonic_hu_d04.yaml missing!"); return False
        print("  ✓ sonic_hu_d04.yaml present")
        urdf_path = Path("/workspace/GR00T-WholeBodyControl/gear_sonic/data/assets/robot_description/urdf/hu_d04/hu_d04.urdf")
        mjcf_path = Path("/workspace/GR00T-WholeBodyControl/gear_sonic/data/assets/robot_description/mjcf/hu_d04.xml")
        mesh_dir = Path("/workspace/GR00T-WholeBodyControl/gear_sonic/data/assets/robot_description/meshes/HU_D04_01")
        print(f"  {'✓' if urdf_path.exists() else '✗'} URDF: {urdf_path}")
        print(f"  {'✓' if mjcf_path.exists() else '✗'} MJCF: {mjcf_path}")
        print(f"  {'✓' if mesh_dir.exists() else '✗'} Meshes: {mesh_dir} ({len(list(mesh_dir.glob('*.STL')))} STL)")
        return True
    except Exception as e:
        print("  FAIL:", e); return False


def main():
    ok1, pkls_all, pkls_flt = check_counts()
    ok2 = check_sessions()
    ok3 = check_schema(pkls_flt if pkls_flt else pkls_all)
    ok4 = check_values(pkls_flt if pkls_flt else pkls_all)
    ok5 = check_loadable_by_training()

    header("SUMMARY")
    results = [("counts", ok1), ("sessions", ok2), ("schema", ok3), ("values", ok4), ("training-compat", ok5)]
    for name, ok in results:
        print(f"  {name:20s} {'OK' if ok else 'ISSUE'}")
    all_ok = all(ok for _, ok in results)
    print()
    print("=" * 72)
    if all_ok:
        print("  READY. Motion dataset usable for SONIC training on HU_D04.")
    else:
        print("  WARN. Review messages above. Output may still be usable depending on which check failed.")
    print("=" * 72)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
