# SONIC Training for LimX HU_D04 Humanoid

Training [GEAR-SONIC](https://github.com/NVlabs/GR00T-WholeBodyControl) whole-body control on the [LimX HU_D04](https://github.com/limxdynamics/humanoid-description) humanoid robot (31 DOF) with 3-point PICO VR controller teleoperation.

## Hardware Requirements

### GPU
- NVIDIA GPU with CUDA 12.x (L40, A100, H100, or RTX PRO 6000 Blackwell)
- Training: 4-8 GPUs recommended (64+ for full convergence speed)
- Memory: 24+ GB per GPU

### CRITICAL: Vulkan Rendering Support
**Isaac Sim requires Vulkan even in headless mode.** Your NVIDIA driver must include rendering libraries (Vulkan ICD, GLX). Compute-only drivers will NOT work.

Verify with:
```bash
vulkaninfo --summary 2>&1 | grep deviceName
# Must show your NVIDIA GPU, not "llvmpipe"
```

If it shows llvmpipe or fails, your driver is compute-only. Fix:
```bash
# Option A: Install NVIDIA GL/Vulkan package
sudo apt install libnvidia-gl-<DRIVER_VERSION>

# Option B: Full driver installer (keeps existing kernel module)
# Download .run from nvidia.com for your driver version
sudo sh NVIDIA-Linux-x86_64-<VER>.run --no-kernel-module

# Option C: Use a cloud VM image with "full" NVIDIA driver (not "compute-only")
```

### Blackwell GPUs (RTX PRO 6000, etc.)
Blackwell (sm_120) requires PyTorch built with CUDA 12.8+. The setup script auto-detects this and uses `cu128` wheels. Earlier CUDA wheels will fail with:
```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

### Software
- Ubuntu 22.04
- Python 3.10 (Isaac Lab) + Python 3.12 (SOMA retargeter)
- ~200 GB disk space (Isaac Sim, motion data, checkpoints)

## Quick Start

```bash
git clone https://github.com/<YOUR_USER>/hu_d04_sonic_setup.git
cd hu_d04_sonic_setup
chmod +x scripts/setup_all.sh
./scripts/setup_all.sh
```

The setup script will:
1. Install system packages (Vulkan, OpenGL, git-lfs)
2. Clone GR00T-WholeBodyControl, soma-retargeter, humanoid-description
3. Apply all HU_D04 patches to both repos
4. Copy URDF, MJCF, and mesh files
5. Create `sonic_env` (Python 3.10 venv for training)
6. Create `soma_env` (Python 3.12 venv for motion retargeting)
7. Download SONIC checkpoints + SMPL data (~30 GB)
8. Verify joint mappings

## What This Repo Contains

### Patches for GR00T-WholeBodyControl
| File | Description |
|------|-------------|
| `gear_sonic/envs/manager_env/robots/hu_d04.py` | Full robot config: 31 DOF, actuator PD gains from SRDF motor specs, IsaacLab/MuJoCo joint mappings |
| `gear_sonic/data/assets/.../urdf/hu_d04/hu_d04.urdf` | URDF with mesh paths fixed for gear_sonic layout |
| `gear_sonic/data/assets/.../mjcf/hu_d04.xml` | Simplified MJCF (no parallel linkages) for motion library |
| `gear_sonic/config/.../sonic_hu_d04.yaml` | Experiment config with HU_D04 body names for rewards, VR tracking, terminations |
| `gear_sonic/data_process/convert_hu_d04_csv_to_motion_lib.py` | CSV-to-PKL converter handling HU_D04's 31 DOFs and oblique hip axes |
| `gear_sonic/scripts/verify_hu_d04_mappings.py` | Verify MuJoCo-side joint ordering |
| `gear_sonic/scripts/verify_hu_d04_isaaclab.py` | Verify IsaacLab-side BFS body ordering |
| `gear_sonic/trl/utils/order_converter.py` | Added `HuD04Converter` class |
| `gear_sonic/envs/manager_env/modular_tracking_env_cfg.py` | Registered `hu_d04` in robot_mapping |
| `gear_sonic/envs/manager_env/robots/__init__.py` | Added hu_d04 import |

### Patches for SOMA Retargeter
| File | Description |
|------|-------------|
| `soma_retargeter/configs/limx_hu_d04/` | 3 JSON configs: IK retargeter, human-robot scaler, feet stabilizer |
| `soma_retargeter/pipelines/utils.py` | Added `LIMX_HU_D04` target type enum |
| `soma_retargeter/pipelines/newton_pipeline.py` | Refactored to load HU_D04 MJCF, shared init code |
| `soma_retargeter/pipelines/feet_stabilizer.py` | Added HU_D04 robot type branch |
| `soma_retargeter/assets/csv.py` | Added `LimXHUD04_31DOF_CSVConfig` (31 DOF CSV format) |
| `app/bvh_to_csv_converter.py` | Added `limx_hu_d04` target option + CSV config routing |
| `assets/hu_d04_batch_config.json` | Batch retargeting config for headless processing |

## HU_D04 Robot Specifications

| Property | Value |
|----------|-------|
| Total DOF | 31 (6L leg + 6R leg + 3 waist + 7L arm + 7R arm + 2 head) |
| Root body | `base_link` |
| Torso (VR tracking) | `waist_pitch_link` |
| Wrist endpoints (VR tracking) | `left_wrist_roll_link`, `right_wrist_roll_link` |
| Foot contacts | `left_ankle_roll_link`, `right_ankle_roll_link` |
| Standing height | ~1.0 m |

### Motor Groups and PD Gains
Derived from SRDF rotor specs: `armature = rotor_mass * gear_ratio^2`, `KP = armature * omega^2`, `KD = 2 * zeta * armature * omega` (10 Hz natural freq, damping ratio 2.0).

| Group | Joints | Effort (Nm) | Armature |
|-------|--------|-------------|----------|
| Hip/Knee | hip_pitch/roll/yaw, knee | 140 | 0.14125 |
| Ankle | ankle_pitch/roll | 42 | 0.1845 |
| Waist | waist_yaw/roll/pitch | 42 | 0.1845 |
| Shoulder/Elbow | shoulder_pitch/roll/yaw, elbow | 42 | 0.0887 |
| Wrist | wrist_yaw/pitch/roll | 19 | 0.0153 |
| Head | head_yaw/pitch | 19 | 0.0153 |

## Step-by-Step Manual Setup

If you prefer to run steps individually instead of using `setup_all.sh`:

### Step 1: Training Environment

```bash
# Create venv
python3 -m venv ~/sonic_env
source ~/sonic_env/bin/activate
pip install --upgrade pip setuptools==75.8.2

# PyTorch (use cu128 for Blackwell, cu124 for older GPUs)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Isaac Lab (PIP_NO_BUILD_ISOLATION needed for flatdict build issue)
PIP_NO_BUILD_ISOLATION=false pip install isaaclab

# Fix dependencies
pip install numpy==1.26.4 pillow==11.0.0 lxml open3d mujoco

# Install gear_sonic
cd ~/GR00T-WholeBodyControl
pip install -e "gear_sonic/[training]"

# Download checkpoints
pip install huggingface_hub
python download_from_hf.py --training
```

### Step 2: Patch Isaac Lab
Isaac Lab 2.1 (latest pip) is missing `quat_apply_inverse`. Apply the patch:
```bash
python scripts/patch_isaaclab_math.py \
    $(find ~/sonic_env -path "*/isaaclab/utils/math.py" | head -1)
```

### Step 3: Verify Joint Mappings
```bash
cd ~/GR00T-WholeBodyControl
python gear_sonic/scripts/verify_hu_d04_mappings.py
# Should print: "All DOF mappings verified successfully!"

python gear_sonic/scripts/verify_hu_d04_isaaclab.py
# Should print: "All 32 body orderings match!"
```

### Step 4: SOMA Retargeter Environment (for motion data)
```bash
# Requires Python 3.12
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.12 python3.12-venv python3.12-dev

python3.12 -m venv ~/soma_env
source ~/soma_env/bin/activate
cd ~/soma-retargeter
git lfs pull
pip install -e .
pip install huggingface_hub
```

### Step 5: Motion Data Retargeting

#### 5a: Download Bones-SEED BVH data (requires access approval)
Go to https://huggingface.co/datasets/bones-studio/seed, accept the license, then:
```bash
source ~/soma_env/bin/activate
hf auth login --token <YOUR_TOKEN>
hf download bones-studio/seed --repo-type dataset \
    --include "soma_uniform.tar.gz" --local-dir ~/bones-seed
cd ~/bones-seed && tar -xzf soma_uniform.tar.gz
```

#### 5b: Retarget BVH to HU_D04
Edit `~/soma-retargeter/assets/hu_d04_batch_config.json` to point `import_folder` to the extracted BVH directory, then:
```bash
source ~/soma_env/bin/activate
cd ~/soma-retargeter
python app/bvh_to_csv_converter.py \
    --config assets/hu_d04_batch_config.json --viewer null
```

#### 5c: Convert CSV to PKL and filter
```bash
source ~/sonic_env/bin/activate
cd ~/GR00T-WholeBodyControl

python gear_sonic/data_process/convert_hu_d04_csv_to_motion_lib.py \
    --input ~/soma-retargeter/assets/motions/hu_d04_csv/ \
    --output data/hu_d04_motions/robot \
    --fps 30 --fps_source 120 --individual --num_workers 16

python gear_sonic/data_process/filter_and_copy_bones_data.py \
    --source data/hu_d04_motions/robot \
    --dest data/hu_d04_motions/robot_filtered --workers 16
```

### Step 6: Training

```bash
source ~/sonic_env/bin/activate
cd ~/GR00T-WholeBodyControl

# Set LD_LIBRARY_PATH for Omniverse extensions
OV_LIB=$(find ~/.local/share/ov/data/exts/v2 -name "*.so" -exec dirname {} \; 2>/dev/null | sort -u | tr '\n' ':')
export LD_LIBRARY_PATH="${OV_LIB}${LD_LIBRARY_PATH}"

# Smoke test (1 env, 5 iterations)
OMNI_KIT_ACCEPT_EULA=YES WANDB_MODE=disabled python gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/all_modes/sonic_hu_d04 \
    num_envs=1 headless=True \
    ++algo.config.num_learning_iterations=5 \
    ++manager_env.commands.motion.motion_lib_cfg.motion_file=data/hu_d04_motions/robot_filtered \
    use_wandb=false

# Full training (4 GPUs)
OMNI_KIT_ACCEPT_EULA=YES accelerate launch --num_processes=4 \
    gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/all_modes/sonic_hu_d04 \
    num_envs=4096 headless=True \
    ++manager_env.commands.motion.motion_lib_cfg.motion_file=data/hu_d04_motions/robot_filtered

# Full training (8 GPUs, multi-node)
OMNI_KIT_ACCEPT_EULA=YES accelerate launch \
    --multi_gpu --num_machines=1 --num_processes=8 \
    gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/all_modes/sonic_hu_d04 \
    num_envs=4096 headless=True \
    ++manager_env.commands.motion.motion_lib_cfg.motion_file=data/hu_d04_motions/robot_filtered
```

### Step 7: ONNX Export + PICO Teleoperation

```bash
# Export trained model to ONNX
python gear_sonic/eval_agent_trl.py \
    +checkpoint=<path_to_checkpoint.pt> \
    +headless=True ++num_envs=1 +export_onnx_only=true

# Run PICO VR teleoperation server
python gear_sonic/scripts/pico_manager_thread_server.py --manager --vis_vr3pt
```

Note: The PICO server has G1-specific hand IK code. For HU_D04, the 3-point VR body tracking (torso + wrists) works via the teleop encoder without robot-specific changes. Hand IK would need adaptation for HU_D04's wrist chain (yaw->pitch->roll vs G1's roll->pitch->yaw).

## Training Expectations

| Hardware | Time to 100K iterations |
|----------|------------------------|
| 4x A100 | ~5-7 days |
| 8x A100 | ~3-5 days |
| 8x H100 | ~2-3 days |
| 4x RTX PRO 6000 | ~4-6 days (estimated) |

### Key Metrics to Monitor
| Metric | Target | Description |
|--------|--------|-------------|
| `rewards/total` | > 3.0 | Total accumulated reward |
| `rewards/tracking_vr_5point_local` | > 0.80 | VR tracking quality |
| `success_rate` (eval) | > 0.97 | Motions without early termination |
| `mpjpe_l` (eval) | < 30 mm | Local per-joint position error |

## Known Issues and Fixes

### 1. `setuptools` 82.x breaks `flatdict` build
Isaac Lab depends on `flatdict` which uses `pkg_resources`, removed in setuptools 82. Fix: pin `setuptools==75.8.2`.

### 2. Isaac Lab 2.1 missing `quat_apply_inverse`
SONIC uses `quat_apply_inverse` from isaaclab.utils.math, but Isaac Lab 2.1 (latest pip) doesn't have it. The setup script patches it automatically.

### 3. Compute-only NVIDIA drivers
Cloud GPU instances often have compute-only drivers (CUDA works, Vulkan doesn't). Isaac Sim requires Vulkan even in headless mode. Install the full driver with rendering libraries.

### 4. Blackwell (sm_120) PyTorch compatibility
PyTorch cu124 and cu126 do NOT have sm_120 kernels. You MUST use cu128+. The setup script auto-detects this.

### 5. HU_D04 oblique hip axes
HU_D04's hip_pitch joints have non-cardinal axes: `(0, 0.90631, -0.42262)` and `(0, 0.90631, 0.42262)`. The custom PKL converter (`convert_hu_d04_csv_to_motion_lib.py`) handles this -- do NOT use the default G1 converter.

### 6. `LD_LIBRARY_PATH` for Omniverse extensions
Isaac Sim's pip install stores shared libraries in `~/.local/share/ov/data/exts/v2/`. You must add these to `LD_LIBRARY_PATH` before running training, otherwise you get `libhdx.so: cannot open shared object file`.

### 7. Isaac Sim requires `isaacsim[all,extscache]` from NVIDIA PyPI
A bare `pip install isaacsim` installs the bootstrap package without the `SimulationApp` extension. You must install the full package with extensions:
```bash
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
```
Without this, training fails with `TypeError: 'NoneType' object is not callable` when creating `SimulationApp`.

### 8. `torch_humanoid_batch.py` assumes cardinal joint axes
The SONIC motion library parser (`torch_humanoid_batch.py`) uses `int()` to parse MJCF joint axis values, which fails for HU_D04's oblique hip axes (e.g. `0.90631`). The setup patch changes this to `float()`. It also filters XML comments when parsing `<actuator>` children.

### 9. `base_com` event references `torso_link` (G1-specific)
The base training config hardcodes `torso_link` in the `base_com` randomization event. HU_D04 uses `waist_pitch_link` instead. The HU_D04 experiment config overrides this.

### 10. `undesired_contacts` reward: `exclude_body_names` not supported
The `undesired_contacts` reward function only accepts `sensor_cfg` and `threshold` parameters. To exclude specific bodies from contact penalties, use a negative-lookahead regex in `body_names` instead of a separate `exclude_body_names` parameter.

### 11. Missing pip dependencies: `tensordict`, `vector-quantize-pytorch`
These are required by the SONIC actor-critic modules but not listed in `gear_sonic`'s training extras. The setup script installs them explicitly.

## Architecture Notes

### Two Python Environments Required
- **`sonic_env`** (Python 3.10/3.11): Isaac Lab, Isaac Sim, PyTorch, gear_sonic training -- used for training and PKL conversion
- **`soma_env`** (Python 3.12): SOMA retargeter with Newton physics, warp -- used for BVH-to-CSV retargeting only

These can't be combined due to conflicting Python version and dependency requirements.

### Motion Data Pipeline
```
BVH (SOMA skeleton, 120fps)
  |-- SOMA Retargeter (soma_env) --> CSV (HU_D04 31 DOF, 120fps, degrees/cm)
  |-- convert_hu_d04_csv_to_motion_lib.py (sonic_env) --> PKL (MuJoCo order, 30fps, radians/m)
  |-- filter_and_copy_bones_data.py --> PKL (filtered)
  --> Training input
```

### VR 3-Point Tracking Bodies
For PICO controller teleoperation:
- Torso: `waist_pitch_link` (equivalent to G1's `torso_link`)
- Left wrist: `left_wrist_roll_link` (end of left arm chain)
- Right wrist: `right_wrist_roll_link` (end of right arm chain)
