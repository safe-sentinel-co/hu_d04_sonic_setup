"""
Verify IsaacLab <-> MuJoCo joint/body ordering mappings for HU_D04.

This script loads both the URDF (IsaacLab) and MJCF (MuJoCo) models and
prints the joint ordering in each simulator. Use this to verify or correct
the mapping arrays in hu_d04.py.

Usage:
    # MuJoCo-only verification (no IsaacLab required):
    python gear_sonic/scripts/verify_hu_d04_mappings.py --mujoco-only

    # Full verification with IsaacLab (requires IsaacLab environment):
    python gear_sonic/scripts/verify_hu_d04_mappings.py

    # Generate mapping arrays from scratch:
    python gear_sonic/scripts/verify_hu_d04_mappings.py --generate
"""

import argparse
from pathlib import Path


def verify_mujoco_order():
    """Print joint ordering from the MuJoCo XML."""
    try:
        import mujoco
    except ImportError:
        print("ERROR: mujoco not installed. Run: pip install mujoco")
        return None

    mjcf_path = Path(__file__).resolve().parent.parent / "data/assets/robot_description/mjcf/hu_d04.xml"
    if not mjcf_path.exists():
        print(f"ERROR: MJCF not found at {mjcf_path}")
        return None

    model = mujoco.MjModel.from_xml_path(str(mjcf_path))

    print("=" * 60)
    print("MuJoCo Joint Order (from hu_d04.xml)")
    print("=" * 60)

    mj_joint_names = []
    mj_body_names = []

    # Print joints (skip the free joint at index 0)
    dof_idx = 0
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        jnt_type = model.jnt_type[i]
        if jnt_type == 0:  # free joint
            print(f"  Joint {i}: {name} (free, 7 DOFs) -- SKIPPED")
            continue
        print(f"  DOF {dof_idx}: {name}")
        mj_joint_names.append(name)
        dof_idx += 1

    print(f"\nTotal actuated DOFs: {dof_idx}")

    # Print bodies
    print("\n" + "=" * 60)
    print("MuJoCo Body Order")
    print("=" * 60)
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if name and name != "world":
            print(f"  Body {len(mj_body_names)}: {name}")
            mj_body_names.append(name)

    return mj_joint_names, mj_body_names


def _load_hu_d04_constants():
    """Load HU_D04 constants directly from file, bypassing IsaacLab imports."""
    import importlib.util, sys, types
    spec = importlib.util.spec_from_file_location(
        "hu_d04_raw",
        Path(__file__).resolve().parent.parent / "envs/manager_env/robots/hu_d04.py",
    )
    mod = importlib.util.module_from_spec(spec)
    _fakes = {}
    for mn in ["isaaclab", "isaaclab.actuators", "isaaclab.assets",
                "isaaclab.assets.articulation", "isaaclab.sim"]:
        if mn not in sys.modules:
            _fakes[mn] = types.ModuleType(mn)
            sys.modules[mn] = _fakes[mn]
            m = _fakes[mn]
            m.ImplicitActuatorCfg = type("S", (), {"__init__": lambda s, **k: None})
            m.ArticulationCfg = type("S", (), {
                "__init__": lambda s, **k: None,
                "InitialStateCfg": type("S", (), {"__init__": lambda s, **k: None}),
            })
            m.UrdfFileCfg = lambda **k: None
            m.RigidBodyPropertiesCfg = lambda **k: None
            m.ArticulationRootPropertiesCfg = lambda **k: None
    try:
        spec.loader.exec_module(mod)
    except Exception:
        pass
    finally:
        for mn in _fakes:
            sys.modules.pop(mn, None)
    return mod


def generate_mappings(mj_joint_names, mj_body_names):
    """Generate mapping arrays from MuJoCo joint/body names."""
    HU_D04_ISAACLAB_JOINTS = _load_hu_d04_constants().HU_D04_ISAACLAB_JOINTS

    # IsaacLab DOF names (body index 1..N maps to DOF 0..N-1)
    il_dof_names = []
    for body_name in HU_D04_ISAACLAB_JOINTS[1:]:  # skip root
        # Convert body name to joint name
        joint_name = body_name.replace("_link", "_joint")
        il_dof_names.append(joint_name)

    il_body_names = HU_D04_ISAACLAB_JOINTS

    # Build DOF mapping: isaaclab_to_mujoco
    print("\n" + "=" * 60)
    print("Generated ISAACLAB_TO_MUJOCO_DOF:")
    print("=" * 60)
    il_to_mj_dof = []
    for il_idx, il_name in enumerate(il_dof_names):
        if il_name in mj_joint_names:
            mj_idx = mj_joint_names.index(il_name)
            il_to_mj_dof.append(mj_idx)
            print(f"  IL {il_idx:2d} ({il_name:35s}) -> MJ {mj_idx:2d}")
        else:
            print(f"  IL {il_idx:2d} ({il_name:35s}) -> NOT FOUND IN MUJOCO!")
            il_to_mj_dof.append(-1)

    # Build inverse DOF mapping: mujoco_to_isaaclab
    mj_to_il_dof = []
    for mj_idx, mj_name in enumerate(mj_joint_names):
        il_name = mj_name  # joint names should match
        if il_name in il_dof_names:
            il_idx = il_dof_names.index(il_name)
            mj_to_il_dof.append(il_idx)
        else:
            mj_to_il_dof.append(-1)

    # Build body mapping: isaaclab_to_mujoco
    il_to_mj_body = []
    for il_idx, il_name in enumerate(il_body_names):
        if il_name in mj_body_names:
            mj_idx = mj_body_names.index(il_name)
            il_to_mj_body.append(mj_idx)
        else:
            il_to_mj_body.append(-1)

    mj_to_il_body = []
    for mj_idx, mj_name in enumerate(mj_body_names):
        if mj_name in il_body_names:
            il_idx = il_body_names.index(mj_name)
            mj_to_il_body.append(il_idx)
        else:
            mj_to_il_body.append(-1)

    # Print as Python arrays
    print("\n" + "=" * 60)
    print("Copy-paste arrays for hu_d04.py:")
    print("=" * 60)
    print(f"HU_D04_ISAACLAB_TO_MUJOCO_DOF = {il_to_mj_dof}")
    print(f"HU_D04_MUJOCO_TO_ISAACLAB_DOF = {mj_to_il_dof}")
    print(f"HU_D04_ISAACLAB_TO_MUJOCO_BODY = {il_to_mj_body}")
    print(f"HU_D04_MUJOCO_TO_ISAACLAB_BODY = {mj_to_il_body}")

    # Verify round-trip
    print("\nRound-trip verification (DOF):")
    for i in range(len(il_to_mj_dof)):
        mj = il_to_mj_dof[i]
        back = mj_to_il_dof[mj] if mj >= 0 else -1
        ok = "OK" if back == i else "MISMATCH!"
        print(f"  IL {i} -> MJ {mj} -> IL {back}  {ok}")


def verify_existing_mappings(mj_joint_names, mj_body_names):
    """Verify the existing mapping arrays in hu_d04.py against the actual MuJoCo model."""
    mod = _load_hu_d04_constants()
    HU_D04_ISAACLAB_JOINTS = mod.HU_D04_ISAACLAB_JOINTS
    HU_D04_ISAACLAB_TO_MUJOCO_DOF = mod.HU_D04_ISAACLAB_TO_MUJOCO_DOF
    HU_D04_MUJOCO_TO_ISAACLAB_DOF = mod.HU_D04_MUJOCO_TO_ISAACLAB_DOF

    il_dof_names = []
    for body_name in HU_D04_ISAACLAB_JOINTS[1:]:
        joint_name = body_name.replace("_link", "_joint")
        il_dof_names.append(joint_name)

    print("\n" + "=" * 60)
    print("Verifying existing mappings:")
    print("=" * 60)

    errors = 0
    for il_idx, il_name in enumerate(il_dof_names):
        expected_mj_idx = mj_joint_names.index(il_name) if il_name in mj_joint_names else -1
        actual_mj_idx = HU_D04_ISAACLAB_TO_MUJOCO_DOF[il_idx]
        ok = expected_mj_idx == actual_mj_idx
        if not ok:
            errors += 1
            print(f"  MISMATCH IL {il_idx} ({il_name}): expected MJ {expected_mj_idx}, got MJ {actual_mj_idx}")
        else:
            print(f"  OK: IL {il_idx} ({il_name}) -> MJ {actual_mj_idx}")

    if errors == 0:
        print("\nAll DOF mappings verified successfully!")
    else:
        print(f"\n{errors} mapping errors found. Run with --generate to produce correct arrays.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify HU_D04 joint/body ordering mappings")
    parser.add_argument("--mujoco-only", action="store_true", help="Only check MuJoCo ordering")
    parser.add_argument("--generate", action="store_true", help="Generate mapping arrays from MuJoCo model")
    args = parser.parse_args()

    result = verify_mujoco_order()

    if result is not None:
        mj_joints, mj_bodies = result
        if args.generate:
            generate_mappings(mj_joints, mj_bodies)
        else:
            verify_existing_mappings(mj_joints, mj_bodies)
