# Training SONIC on HU_D04 from the retargeted dataset

This guide picks up *after* the retargeting pipeline (`scripts/setup_all.sh` + `debug_tools/pipeline.sh`) has already produced the motion corpus. It covers downloading the dataset and launching SONIC whole-body RL training on a Vulkan-capable GPU box.

---

## 1. Dataset layout

The retargeting pipeline produces three tarballs, available on S3 at `s3://safesentinel-inc/hu_d04_motions/`:

| file | size | contents | use |
|------|------|----------|-----|
| `robot_filtered.tar.gz` | ~7 GB | **102,506 PKLs** (filter-approved), MuJoCo-order 30 fps | **Training input — use this** |
| `robot.tar.gz` | ~8 GB | 112,030 PKLs (all retargeted motions, pre-filter) | Use if you want to re-filter with different keywords |
| `csv.tar.gz` | ~6 GB | 112,030 CSVs (pre-PKL intermediate) | Only needed if re-running the CSV→PKL conversion |

Each PKL has:
```python
{
    "<motion_name>": {
        "dof":               (T, 31)    float32,  # joint angles in radians, MuJoCo DFS order
        "root_trans_offset": (T, 3)     float32,  # meters
        "root_rot":          (T, 4)     float32,  # quaternion (w, x, y, z)
        "pose_aa":           (T, 32, 3) float32,  # axis-angle per body (base_link + 31 links)
        "fps":               30.0,
    }
}
```

---

## 2. Download on the training box

Replace `AWS_…_KEY` with your own or use `aws configure`. These credentials just need read access to `s3://safesentinel-inc/hu_d04_motions/`.

```bash
# one-time setup
pip install awscli
export AWS_ACCESS_KEY_ID=<YOUR_ID>
export AWS_SECRET_ACCESS_KEY=<YOUR_SECRET>
export AWS_DEFAULT_REGION=us-east-1

# download + unpack to where SONIC expects
cd ~/GR00T-WholeBodyControl
mkdir -p data/hu_d04_motions
cd data/hu_d04_motions

aws s3 cp s3://safesentinel-inc/hu_d04_motions/robot_filtered.tar.gz .
tar xzf robot_filtered.tar.gz
rm robot_filtered.tar.gz

# sanity
find robot_filtered -name '*.pkl' | wc -l      # 102506
ls robot_filtered/ | wc -l                     # 124 session dirs
```

---

## 3. Training-box prerequisites (Vulkan-capable)

SONIC uses Isaac Sim 5.1 which **requires a functional Vulkan userspace driver** even in headless mode. Compute-only drivers (common on cloud "PyTorch" templates) WILL fail. Verify first:

```bash
sudo apt install -y vulkan-tools
vulkaninfo --summary 2>&1 | grep deviceName
# Must show "NVIDIA GeForce RTX ..." — NOT just "llvmpipe"
```

Known-good templates:
- RunPod → search community templates for **"Isaac Sim"** or **"Omniverse"**
- Lambda Labs → all GPU instances
- AWS G5/G6/p4d with Deep-Learning AMI including **DCV/GRID**

If vulkaninfo only shows `llvmpipe`, **stop — the driver is compute-only**. Isaac Sim will crash without error on `vkCreateInstance` failure.

---

## 4. Environment setup

On the training box:

```bash
git clone https://github.com/safe-sentinel-co/hu_d04_sonic_setup.git
cd hu_d04_sonic_setup
chmod +x scripts/setup_all.sh
./scripts/setup_all.sh
```

This will:
1. Install system packages (Vulkan, OpenGL, git-lfs)
2. Clone GR00T-WholeBodyControl, soma-retargeter, humanoid-description
3. Apply all HU_D04 patches (the fixed SpaceConverter + joint_offsets)
4. Create `~/sonic_env` (Python 3.10/3.11 venv) with Isaac Lab + Isaac Sim + gear_sonic
5. Create `~/soma_env` (Python 3.12) — not needed for training but needed if you re-retarget
6. Download SONIC pretrained checkpoints + SMPL data (~30 GB)
7. Patch Isaac Lab's `math.py` to add `quat_apply_inverse`

---

## 5. Smoke test (1 env, 5 iterations)

Verify Isaac Sim boots and the HU_D04 config loads before committing to a multi-day run:

```bash
source ~/sonic_env/bin/activate
cd ~/GR00T-WholeBodyControl

# Omniverse needs this to find shared libs
OV_LIB=$(find ~/.local/share/ov/data/exts/v2 -name "*.so" -exec dirname {} \; 2>/dev/null | sort -u | tr '\n' ':')
export LD_LIBRARY_PATH="${OV_LIB}${LD_LIBRARY_PATH}"

OMNI_KIT_ACCEPT_EULA=YES WANDB_MODE=disabled python gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/all_modes/sonic_hu_d04 \
    num_envs=1 headless=True \
    ++algo.config.num_learning_iterations=5 \
    ++manager_env.commands.motion.motion_lib_cfg.motion_file=data/hu_d04_motions/robot_filtered \
    use_wandb=false
```

You should see:
- Isaac Sim boots (no Vulkan error)
- `sonic_hu_d04.yaml` experiment config loaded
- Terrain generated
- 5 policy-update iterations logged
- Clean exit (or Ctrl-C after you see it working)

If smoke passes, you're ready for the real run.

---

## 6. Full training

```bash
# 4 GPUs, single node
OMNI_KIT_ACCEPT_EULA=YES accelerate launch --num_processes=4 \
    gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/all_modes/sonic_hu_d04 \
    num_envs=4096 headless=True \
    ++manager_env.commands.motion.motion_lib_cfg.motion_file=data/hu_d04_motions/robot_filtered
```

```bash
# 8 GPUs, single node
OMNI_KIT_ACCEPT_EULA=YES accelerate launch \
    --multi_gpu --num_machines=1 --num_processes=8 \
    gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/all_modes/sonic_hu_d04 \
    num_envs=4096 headless=True \
    ++manager_env.commands.motion.motion_lib_cfg.motion_file=data/hu_d04_motions/robot_filtered
```

### Notes on `num_envs`

| GPU VRAM | max safe `num_envs` |
|----------|---------------------|
| 24 GB (RTX 4090) | ~2048 |
| 48 GB (L40, A6000) | ~4096 |
| 80 GB (A100-80, H100) | ~8192 |

### Target metrics (SONIC paper targets)

| metric | target | meaning |
|--------|--------|---------|
| `rewards/total` | > 3.0 | total accumulated reward |
| `rewards/tracking_vr_5point_local` | > 0.80 | VR-tracking quality |
| `success_rate` (eval) | > 0.97 | motions without early termination |
| `mpjpe_l` (eval) | < 30 mm | per-joint position error |

### Expected wall-time to 100K iterations (for reference)

| hardware | time |
|----------|------|
| 4x A100-80 | 5-7 days |
| 8x A100-80 | 3-5 days |
| 8x H100    | 2-3 days |
| 4x RTX PRO 6000 Blackwell | 4-6 days (estimate) |

---

## 7. Export to ONNX (after training)

```bash
source ~/sonic_env/bin/activate
cd ~/GR00T-WholeBodyControl
python gear_sonic/eval_agent_trl.py \
    +checkpoint=<path_to_best_checkpoint.pt> \
    +headless=True ++num_envs=1 +export_onnx_only=true
```

---

## 8. PICO 3-point teleoperation

HU_D04 VR-tracking bodies are:

| VR point | HU_D04 body |
|----------|-------------|
| Torso    | `waist_pitch_link` |
| Left wrist  | `left_wrist_roll_link` |
| Right wrist | `right_wrist_roll_link` |

Run the teleop server:

```bash
python gear_sonic/scripts/pico_manager_thread_server.py --manager --vis_vr3pt
```

Note: the existing PICO hand-IK code is G1-specific (assumes a `roll→pitch→yaw` wrist chain). HU_D04 uses `yaw→pitch→roll` and a different rotation composition — adapting requires porting the IK logic to HU_D04's wrist order.

---

## Troubleshooting

### Isaac Sim hangs/crashes on startup
- Run `vulkaninfo --summary`. If you see `ERROR_INCOMPATIBLE_DRIVER` or `llvmpipe`, the driver is compute-only. Switch to a Vulkan-capable template.
- On some pods, set: `export OMNI_KIT_ACCEPT_EULA=YES` before launching.

### `libhdx.so: cannot open shared object file`
- Omniverse stores some plugin shared libs in `~/.local/share/ov/data/exts/v2/`. Add them to `LD_LIBRARY_PATH`:
  ```bash
  OV_LIB=$(find ~/.local/share/ov/data/exts/v2 -name "*.so" -exec dirname {} \; | sort -u | tr '\n' ':')
  export LD_LIBRARY_PATH="${OV_LIB}${LD_LIBRARY_PATH}"
  ```

### OOM during training
- Reduce `num_envs` by 2x until it fits. `num_envs=1024` still trains.

### Re-filter the motion set
- Unpack `robot.tar.gz` (all retargeted motions, pre-filter) to `data/hu_d04_motions/robot/`
- Run `gear_sonic/data_process/filter_and_copy_bones_data.py` with your preferred keyword list:
  ```bash
  python gear_sonic/data_process/filter_and_copy_bones_data.py \
      --source data/hu_d04_motions/robot \
      --dest data/hu_d04_motions/robot_filtered_custom \
      --add-keywords my_extra_keyword another_one \
      --workers 16
  ```

---

## Changelog — retargeter bug fixes (2026-04-18)

Three bugs were masking each other in the out-of-the-box HU_D04 retargeter patches. All fixed in commit `474f965`:

1. **SpaceConverter (Mujoco)** was `90° around +X` — converted Y-up → Z-up but not the handedness. Fixed to `120° around (1,1,1)` (axis-cycle X→Y→Z→X).
2. **`joint_offsets.q`** were byte-for-byte copies of G1's values. Recomputed against HU_D04 rest pose (elbow −90° to match SOMA's hand-forward reference).
3. **`ik_map` missing intermediate spine** — waist_pitch saturated at +45° 90 % of the time. Fixed by bumping `Chest.r_weight` and `Hips.r_weight` so the three waist joints redistribute rotation correctly.

The dataset uploaded to S3 is retargeted with all fixes applied.
