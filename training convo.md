# HU_D04 SONIC Training - Full Conversation Log

## Overview
This document captures the complete conversation and learnings from integrating the LimX HU_D04 humanoid (31 DOF) into the GEAR-SONIC whole-body control framework, including all troubleshooting, decisions, and discoveries made along the way.

---

## Phase 1: Research and Planning

### Initial Goal
Train GEAR-SONIC on a new embodiment (LimX HU_D04) to enable:
- 3-point PICO VR controller teleoperation
- Simulation control of the humanoid

### Repos Involved
- **GR00T-WholeBodyControl**: https://github.com/NVlabs/GR00T-WholeBodyControl.git (SONIC framework)
- **humanoid-description**: https://github.com/limxdynamics/humanoid-description (HU_D04 URDF/meshes)
- **soma-retargeter**: https://github.com/NVIDIA/soma-retargeter (motion retargeting)

### Key Documentation References
- New embodiments guide: https://nvlabs.github.io/GR00T-WholeBodyControl/user_guide/new_embodiments.html
- Training guide: https://nvlabs.github.io/GR00T-WholeBodyControl/user_guide/training.html
- Installation: https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/installation_training.html
- Paper: https://nvlabs.github.io/GEAR-SONIC/static/pdf/sonic_paper.pdf

### HU_D04 Robot Specifications (from URDF analysis)
- **31 DOF**: 6L leg + 6R leg + 3 waist + 7L arm + 7R arm + 2 head
- **Root body**: `base_link` (not `pelvis` like G1)
- **Torso equivalent**: `waist_pitch_link` (G1 uses `torso_link`)
- **Wrist endpoints**: `left_wrist_roll_link`, `right_wrist_roll_link` (G1 uses `left_wrist_yaw_link`)
- **Standing height**: ~1.0m (from MJCF)
- **Oblique hip axes**: left_hip_pitch axis is `(0, 0.90631, -0.42262)`, NOT a cardinal axis

### Motor Specs (from SRDF)
| Group | Rotor Mass (kg) | Gear Ratio | Armature | Effort (Nm) |
|-------|----------------|------------|----------|-------------|
| Hip/Knee | 0.000226 | 25 | 0.14125 | 140 |
| Ankle/Waist | 0.0001424 | 36 | 0.1845504 | 42 |
| Shoulder/Elbow | 0.000141873 | 25 | 0.0886706 | 42 |
| Wrist/Head | 0.000019308 | 28.17 | 0.01532178 | 19 |

### Existing Robot Implementations (H2 used as reference)
The H2 robot (31 DOF) was the primary reference for adding HU_D04. Key pattern:
- Robot config in `gear_sonic/envs/manager_env/robots/<robot>.py`
- Registered in `modular_tracking_env_cfg.py` `robot_mapping` dict
- Order converter in `gear_sonic/trl/utils/order_converter.py`
- Experiment config in `gear_sonic/config/exp/manager/universal_token/all_modes/`

---

## Phase 2: File Creation (HU_D04 Integration)

### Files Created in GR00T-WholeBodyControl

#### 1. URDF (`gear_sonic/data/assets/robot_description/urdf/hu_d04/hu_d04.urdf`)
- Copied from `humanoid-description/HU_D04_description/urdf/HU_D04_01.urdf`
- Fixed mesh paths: `package://HU_D04_description/meshes/` → `package://robot_description/meshes/`
- 93 STL mesh files copied to `gear_sonic/data/assets/robot_description/meshes/HU_D04_01/`

#### 2. Simplified MJCF (`gear_sonic/data/assets/robot_description/mjcf/hu_d04.xml`)
- Created from scratch based on URDF geometry
- **Stripped parallel linkage mechanisms**: The original HU_D04 MJCF had achilles rod joints, waist A/B joints, and ball joints for closed-loop kinematic chains. These don't exist in the URDF and aren't part of the 31 controlled DOFs.
- Only contains the 31 revolute joints + 1 free joint
- `meshdir` set to `../meshes/` (relative to MJCF file location)
- Includes simplified collision geometry (capsules, boxes, cylinders) alongside visual meshes

#### 3. Robot Config (`gear_sonic/envs/manager_env/robots/hu_d04.py`)
- **Actuator PD gains**: Derived from SRDF motor specs using the same formula as H2/G1:
  - `armature = rotor_mass * gear_ratio^2`
  - `KP (stiffness) = armature * (10Hz * 2pi)^2`
  - `KD (damping) = 2 * 2.0 * armature * (10Hz * 2pi)`
- **Joint/body ordering**: IsaacLab BFS traversal (32 bodies including root)
- **IsaacLab↔MuJoCo mappings**: 31-element arrays mapping BFS to DFS ordering
- **Actuator groups**: legs, feet, waist, head, arms, wrists (6 groups vs H2's 7)
- **Action scale**: `0.25 * effort_limit / stiffness` per joint (same formula as H2)
- **Initial standing pose**: Hip pitch -0.312, knee 0.669, ankle pitch -0.363

#### 4. Experiment Config (`sonic_hu_d04.yaml`)
- Based on `sonic_h2.yaml` template
- Body name mappings for HU_D04:
  - `anchor_body: "base_link"` (was `pelvis`)
  - `vr_3point_body`: `waist_pitch_link`, `left_wrist_roll_link`, `right_wrist_roll_link`
  - `reward_point_body`: same 3 bodies
  - `body_names`: 14 tracked bodies matching HU_D04's kinematic chain
- Removed termination overrides (caused `TypeError` - these are handled by base config defaults)
- MJCF asset: `hu_d04.xml`

#### 5. Order Converter (`HuD04Converter` in `order_converter.py`)
- VR 3-point body names: `waist_pitch_link`, `left_wrist_roll_link`, `right_wrist_roll_link`
- Foot body names: `left_ankle_roll_link`, `right_ankle_roll_link`

#### 6. PKL Converter (`convert_hu_d04_csv_to_motion_lib.py`)
- Separate from the G1 converter because:
  - 31 DOFs vs 29
  - 32 bodies vs 30
  - Oblique hip_pitch axes (non-cardinal vectors)
- DOF_AXIS array has normalized axis vectors extracted from the MJCF
- Handles CSV→PKL conversion with downsampling (120fps→30fps)

#### 7. Verification Scripts
- `verify_hu_d04_mappings.py`: Loads MJCF in MuJoCo, prints actual joint ordering, verifies mapping arrays
- `verify_hu_d04_isaaclab.py`: Parses URDF XML, does BFS traversal, verifies body ordering

### Files Modified in GR00T-WholeBodyControl
- `robots/__init__.py`: Added `hu_d04` import
- `modular_tracking_env_cfg.py`: Added `hu_d04` to `robot_mapping` dict, imported `hu_d04` module

---

## Phase 3: SOMA Retargeter Modifications

### Discovery: SOMA Retargeter is G1-Only
The SOMA retargeter was hardcoded for Unitree G1 (29 DOF) output. Adding HU_D04 required modifying 6 source files across the repo.

### Changes Made

#### New Config Files (`configs/limx_hu_d04/`)
1. **`soma_to_hu_d04_retargeter_config.json`**: IK mapping with HU_D04 body names
   - `pelvis` → `base_link`
   - `torso_link` → `waist_pitch_link`
   - `left_wrist_yaw_link` → `left_wrist_roll_link`
   - Same IK weights as G1 config
2. **`soma_to_hu_d04_scaler_config.json`**: Human-to-robot scaling
   - Slightly smaller scale factors than G1 (HU_D04 is ~1.0m vs G1 ~1.04m)
   - Same joint hierarchy and offset structure
3. **`hu_d04_feet_stabilizer_config.json`**: Feet IK stabilization
   - Uses `base_link` instead of `pelvis` for root effector

#### Source Code Changes
1. **`pipelines/utils.py`**: Added `LIMX_HU_D04` enum, string mappings, config loading branch
2. **`pipelines/newton_pipeline.py`**: 
   - Added MJCF loading branch for HU_D04 (loads from local file, not Newton asset download)
   - **Refactored**: Moved common init code (scaler, IK model, mapping, stabilizer) OUT of the G1 `if` block so it runs for all robot types
3. **`pipelines/feet_stabilizer.py`**: Same refactoring pattern - MJCF loading is robot-specific, rest is shared
4. **`assets/csv.py`**: Added `LimXHUD04_31DOF_CSVConfig` with 31 DOF column header in MuJoCo DFS order
5. **`app/bvh_to_csv_converter.py`**: Added `limx_hu_d04` to target options, CSV config routing in batch export

### SOMA Retargeter Requires Python 3.12
- Pinned dependencies: `numpy==2.4.3`, `scipy==1.17.1`, `warp-lang==1.12.0`
- Needs separate venv (`soma_env`) from training env (`sonic_env`)
- Uses Newton physics library for IK solving (GPU-accelerated via Warp)

---

## Phase 4: Joint Mapping Verification

### How IsaacLab Ordering Works
IsaacLab traverses the URDF kinematic tree in **Breadth-First Search (BFS)** order. The root body is index 0, then all its children at depth 1, then depth 2, etc.

### How MuJoCo Ordering Works
MuJoCo traverses the MJCF tree in **Depth-First Search (DFS)** order, following the XML element order.

### Verification Results
Both sides verified independently:

**MuJoCo side** (loaded actual MJCF into MuJoCo):
```
All DOF mappings verified successfully!
```

**IsaacLab side** (BFS traversal of URDF XML):
```
All 32 body orderings match!
```

### Note on Verification Script
The verify script originally tried to `import hu_d04` which triggered `__init__.py` → `g1.py` → `isaaclab.actuators` (needs full Isaac Sim). Fixed by using `importlib.util.spec_from_file_location` to load the module directly, with stub mock classes for IsaacLab types.

---

## Phase 5: Environment Setup

### Machine Specs
- **GPUs**: 4x NVIDIA RTX PRO 6000 Blackwell Server Edition (98 GB each)
- **OS**: Ubuntu 22.04
- **CUDA**: 12.9
- **Python**: 3.10.12
- **Driver**: 580.126.09 (COMPUTE-ONLY - no Vulkan/rendering)

### Installation Steps and Issues

#### 1. pip and venv
```bash
sudo apt install python3-pip python3-venv
python3 -m venv ~/sonic_env
```

#### 2. setuptools Issue
- `setuptools` 82.x removed `pkg_resources` module
- Isaac Lab's `flatdict` dependency uses `pkg_resources` during build
- **Fix**: Pin `setuptools==75.8.2`

#### 3. Isaac Lab Installation
- `pip install isaaclab` fails with flatdict build error in isolated build env
- **Fix**: `PIP_NO_BUILD_ISOLATION=false pip install isaaclab`
- Isaac Lab 2.1.0 installed (latest on pip, despite docs saying 2.3+ required)

#### 4. Isaac Lab EULA
- First import requires interactive EULA acceptance
- **Fix**: `OMNI_KIT_ACCEPT_EULA=YES` environment variable

#### 5. PyTorch + Blackwell GPU Compatibility (MAJOR ISSUE)
- Blackwell GPUs have compute capability sm_120
- PyTorch cu124: supports up to sm_90 → **FAILS**
- PyTorch cu126: supports up to sm_90 → **FAILS**
- PyTorch cu128: includes sm_120 → **WORKS**
- Isaac Lab pins torch==2.5.1 but we needed 2.11.0+cu128 (pip warning, not fatal)

#### 6. Missing Python Packages
Installed incrementally as import errors surfaced:
- `lxml` (for URDF parsing in motion_lib)
- `open3d` (for mesh processing)
- `mujoco` (for joint mapping verification)

#### 7. Isaac Lab Missing Function
- SONIC uses `quat_apply_inverse` from `isaaclab.utils.math`
- Isaac Lab 2.1 only has `quat_apply` and `quat_apply_yaw`
- **Fix**: Patched math.py to add the function (conjugate quaternion rotation)

#### 8. LD_LIBRARY_PATH for Omniverse Extensions
- Isaac Sim pip install stores shared libraries in `~/.local/share/ov/data/exts/v2/`
- `libhdx.so` and other PhysX libraries are there but not on the default library path
- **Fix**: 
```bash
OV_LIB=$(find ~/.local/share/ov/data/exts/v2 -name "*.so" -exec dirname {} \; | sort -u | tr '\n' ':')
export LD_LIBRARY_PATH="${OV_LIB}${LD_LIBRARY_PATH}"
```

---

## Phase 6: Motion Data Retargeting (End-to-End Verified)

### Pipeline
```
BVH (SOMA skeleton, 120fps) 
  → SOMA Retargeter → CSV (HU_D04 31 DOF, 120fps, degrees/cm)
  → convert_hu_d04_csv_to_motion_lib.py → PKL (MuJoCo order, 30fps, radians/m)
  → filter_and_copy_bones_data.py → PKL (filtered)
```

### Test Run Results
- 10 sample BVH files from SOMA retargeter repo
- All 10 retargeted successfully (7.96s per motion)
- PKL output verified: `dof: (272, 31)`, `pose_aa: (272, 32, 3)`, `fps: 30`
- All 10 passed motion filtering

### Bones-SEED Data Access
- Dataset is gated: https://huggingface.co/datasets/bones-studio/seed
- Requires manual approval from repo authors
- Access request submitted but still pending
- Full dataset: 142K motions, ~114 GB as `soma_uniform.tar.gz`

### Important: Use the HU_D04-Specific Converter
The default `convert_soma_csv_to_motion_lib.py` is hardcoded for G1 (29 DOF, cardinal axes only). Using it on HU_D04 CSVs produces WRONG shapes (`dof: (T, 29)` instead of `(T, 31)`). Always use `convert_hu_d04_csv_to_motion_lib.py`.

---

## Phase 7: Smoke Test Training (BLOCKED)

### What Worked
- Isaac Sim booted in headless mode
- Config loaded correctly (after removing invalid termination overrides)
- Scene creation started
- Terrain generated ("Generating terrains randomly took 1.19 seconds")
- URDF path found
- Physics step-size: 0.005, Environment step-size: 0.02

### What Blocked It

#### CRITICAL: Vulkan Required Even in Headless Mode
Isaac Sim requires Vulkan rendering even when running headless. This machine has a **compute-only** NVIDIA driver:
- CUDA works perfectly
- No `libGLX_nvidia.so`, no `libnvidia-glcore.so`, no NVIDIA Vulkan ICD
- `vulkaninfo` only shows `llvmpipe` (CPU software renderer)

Error chain:
```
VkResult: ERROR_INCOMPATIBLE_DRIVER
vkCreateInstance failed. Vulkan 1.1 is not supported
Failed to create any GPU devices
```

Without Vulkan, the process spins at 100%+ CPU doing nothing productive for 10+ minutes.

#### Docker Attempt Also Failed
- Pulled `nvcr.io/nvidia/isaac-sim:4.5.0` (~15 GB)
- Crashed with segfault in `libomni.kit.renderer.plugin.so`
- Likely because the Isaac Sim 4.5.0 container doesn't support Blackwell GPUs

### Resolution
Need a machine with:
1. Full NVIDIA driver (not compute-only) that includes Vulkan ICD
2. OR the full .run driver installer with `--no-kernel-module` flag

---

## Key Learnings and Gotchas

### 1. Isaac Sim REQUIRES Vulkan
There is no way to run Isaac Sim/Isaac Lab training without Vulkan, even headless. This is the single most important infrastructure requirement. Always verify with `vulkaninfo` before attempting training.

### 2. Blackwell GPUs Need cu128+ PyTorch
sm_120 support was only added in CUDA 12.8 PyTorch wheels. This applies to RTX 5000 series, RTX PRO 6000, and other Blackwell GPUs.

### 3. Two Separate Python Environments
- `sonic_env` (Python 3.10): Training, Isaac Lab, PyTorch
- `soma_env` (Python 3.12): SOMA retargeter, Newton physics
These cannot be merged due to conflicting Python versions and dependency pins.

### 4. HU_D04 Has Non-Cardinal Joint Axes
The hip_pitch joints rotate around `(0, 0.906, -0.423)` not a simple `(0, 1, 0)`. The G1 converter assumes cardinal axes and silently produces wrong axis-angle data. Always use the HU_D04-specific converter.

### 5. Simplified MJCF is Essential
The original HU_D04 MJCF has parallel linkage mechanisms (achilles rods, waist A/B) with ball joints and equality constraints. These must be stripped for SONIC training - the motion library expects only the 31 controlled revolute joints.

### 6. Body Name Mapping is Critical
G1's `pelvis` → HU_D04's `base_link`
G1's `torso_link` → HU_D04's `waist_pitch_link`
G1's `*_wrist_yaw_link` → HU_D04's `*_wrist_roll_link`
Getting these wrong means rewards, terminations, and VR tracking all target non-existent bodies.

### 7. SOMA Retargeter Requires Significant Modification
Despite the documentation suggesting it supports custom robots, the SOMA retargeter is deeply hardcoded for G1. Adding a new robot requires changes to 6 files across the codebase, plus 3 new JSON config files.

### 8. Isaac Lab 2.1 vs SONIC Requirements
SONIC docs say "Isaac Lab 2.3+" but pip only has 2.1. Missing `quat_apply_inverse` must be patched manually. There may be other incompatibilities lurking.

### 9. Python Output Buffering
When redirecting Python output to a log file, use `PYTHONUNBUFFERED=1 python -u` to see output in real time. Without this, Isaac Sim's C-level buffering can make it look like the process is stuck when it's actually working.

### 10. setuptools Version Matters
setuptools 82.x broke `pkg_resources` which `flatdict` (Isaac Lab dependency) needs. Pin to 75.8.2.

---

## File Inventory

### New Files Created (17 total)
```
GR00T-WholeBodyControl/
├── gear_sonic/config/exp/manager/universal_token/all_modes/sonic_hu_d04.yaml
├── gear_sonic/data/assets/robot_description/urdf/hu_d04/hu_d04.urdf
├── gear_sonic/data/assets/robot_description/mjcf/hu_d04.xml
├── gear_sonic/data/assets/robot_description/meshes/HU_D04_01/ (93 STL files)
├── gear_sonic/data_process/convert_hu_d04_csv_to_motion_lib.py
├── gear_sonic/envs/manager_env/robots/hu_d04.py
├── gear_sonic/scripts/verify_hu_d04_mappings.py
└── gear_sonic/scripts/verify_hu_d04_isaaclab.py

soma-retargeter/
├── soma_retargeter/configs/limx_hu_d04/soma_to_hu_d04_retargeter_config.json
├── soma_retargeter/configs/limx_hu_d04/soma_to_hu_d04_scaler_config.json
├── soma_retargeter/configs/limx_hu_d04/hu_d04_feet_stabilizer_config.json
├── assets/hu_d04_batch_config.json
└── assets/robots/limx_hu_d04/mjcf/hu_d04.xml
```

### Files Modified (8 total)
```
GR00T-WholeBodyControl/
├── gear_sonic/envs/manager_env/robots/__init__.py (+1 line)
├── gear_sonic/envs/manager_env/modular_tracking_env_cfg.py (+7 lines)
└── gear_sonic/trl/utils/order_converter.py (+27 lines)

soma-retargeter/
├── app/bvh_to_csv_converter.py (+12 lines)
├── soma_retargeter/assets/csv.py (+40 lines)
├── soma_retargeter/pipelines/feet_stabilizer.py (refactored)
├── soma_retargeter/pipelines/newton_pipeline.py (refactored)
└── soma_retargeter/pipelines/utils.py (+15 lines)

Isaac Lab (runtime patch)/
└── isaaclab/utils/math.py (+18 lines: quat_apply_inverse)
```

---

## Next Steps (when Vulkan-capable machine available)

1. **Run smoke test**: `num_envs=1, headless=True, 5 iterations`
2. **Get Bones-SEED access**: Pending approval at HuggingFace
3. **Full retargeting**: 142K BVH → CSV → PKL pipeline
4. **Full training**: 4096 envs, 100K+ iterations, 4-8 GPUs
5. **ONNX export**: For deployment
6. **PICO teleoperation**: Adapt `pico_manager_thread_server.py` for HU_D04
7. **KP/KD tuning**: If robot oscillates or falls during training
