# Training SONIC on HU_D04 — from the pre-retargeted dataset

Use this guide when you already have the retargeted PKL dataset on S3
and only want to train. No retargeting, no BVH downloads — skip straight
to RL.

---

## Dataset on S3

`s3://safesentinel-inc/hu_d04_motions/`

| file | size | contents | use |
|------|------|----------|-----|
| **`robot_filtered.tar.gz`** | **6.7 GB** | **102,506 PKLs** (training-ready) | **→ Training input** |
| `robot.tar.gz` | 7.7 GB | 112,030 PKLs (pre-filter) | Only if re-filtering with different keywords |
| `csv.tar.gz` | 5.3 GB | 112,030 intermediate CSVs | Only if re-running CSV→PKL |

Each PKL contains:
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

## Pod requirements

**Must be Vulkan-capable** — Isaac Sim requires a functional NVIDIA GL/Vulkan userspace, not just CUDA. Compute-only drivers fail.

Recommended templates:
- **RunPod** community: `Isaac Sim 4.5.0` or `NVIDIA Isaac Lab`
- **Lambda Labs**: any GPU instance (all ship full driver)
- **AWS**: `p4d.24xlarge` / `p5.48xlarge` + Deep Learning AMI (Ubuntu 22.04) with GRID/DCV
- **GCP**: A2 / A3 + Deep Learning VM with "NVIDIA Driver (Full)"

**Avoid**: stock `runpod/pytorch:*`, CUDA-only base images, anything that fails `vulkaninfo`.

GPU sizing:

| GPU | num_envs safe | 100K iter ETA |
|-----|---------------|---------------|
| 8× H100 80GB | 8192 | 2–3 days |
| 8× A100 80GB | 8192 | 3–5 days |
| 4× A100 80GB | 8192 | 5–7 days |
| 4× L40 / A6000 48GB | 4096 | ~6–8 days |
| 6× RTX 4090 24GB | 2048 | 8–10 days |

---

## One-shot quick start

```bash
curl -sL https://raw.githubusercontent.com/safe-sentinel-co/hu_d04_sonic_setup/main/remote_training_setup.sh -o setup.sh
chmod +x setup.sh

export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export N_GPUS=auto
export NUM_ENVS=4096      # 2048 for 24GB; 8192 for 80GB

bash setup.sh smoke       # 5-iter sanity check, ~5 min
bash setup.sh train       # full training, nohup-detached
```

The script:
1. `apt install` vulkan-tools, git, python3-pip
2. `vulkaninfo --summary` — fails fast if not Vulkan-capable
3. Clones `hu_d04_sonic_setup`, runs `scripts/setup_all.sh` (installs Isaac Lab + gear_sonic + pretrained checkpoints, ~30 GB)
4. Pulls `robot_filtered.tar.gz` from S3 and unpacks to `~/GR00T-WholeBodyControl/data/hu_d04_motions/robot_filtered/`
5. `smoke` → 5 iterations; `train` → full run in `nohup` with `~/train.log` + `~/train.pid`

**Monitor**:
```bash
tail -f ~/train.log
watch -n 5 nvidia-smi
```

---

## Manual step-by-step (if you don't want to run the script)

### 1. Verify Vulkan
```bash
sudo apt-get update -qq
sudo apt-get install -y vulkan-tools git git-lfs curl unzip python3-pip
vulkaninfo --summary 2>&1 | grep deviceName
# ✅ NVIDIA GeForce ...    ❌ llvmpipe or ERROR → wrong pod
nvidia-smi
```

### 2. Clone + setup_all
```bash
git clone https://github.com/safe-sentinel-co/hu_d04_sonic_setup.git ~/hu_d04_sonic_setup
cd ~/hu_d04_sonic_setup
chmod +x scripts/setup_all.sh
./scripts/setup_all.sh 2>&1 | tee ~/setup.log
```

### 3. Download dataset from S3
```bash
pip install awscli
export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... AWS_DEFAULT_REGION=us-east-1

cd ~/GR00T-WholeBodyControl
mkdir -p data/hu_d04_motions && cd data/hu_d04_motions
aws s3 cp s3://safesentinel-inc/hu_d04_motions/robot_filtered.tar.gz .
tar xzf robot_filtered.tar.gz && rm robot_filtered.tar.gz

find robot_filtered -name '*.pkl' | wc -l   # 102506
```

### 4. Activate env + fix LD_LIBRARY_PATH
```bash
source ~/sonic_env/bin/activate
OV_LIB=$(find ~/.local/share/ov/data/exts/v2 -name "*.so" -exec dirname {} \; | sort -u | tr '\n' ':')
export LD_LIBRARY_PATH="${OV_LIB}${LD_LIBRARY_PATH}"
```

### 5. Smoke test (1 env, 5 iters — ~5 min)
```bash
cd ~/GR00T-WholeBodyControl
OMNI_KIT_ACCEPT_EULA=YES WANDB_MODE=disabled python gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/all_modes/sonic_hu_d04 \
    num_envs=1 headless=True \
    ++algo.config.num_learning_iterations=5 \
    ++manager_env.commands.motion.motion_lib_cfg.motion_file=data/hu_d04_motions/robot_filtered \
    use_wandb=false
```

### 6. Full training
```bash
# 8 GPUs (best)
OMNI_KIT_ACCEPT_EULA=YES accelerate launch --multi_gpu --num_machines=1 --num_processes=8 \
    gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/all_modes/sonic_hu_d04 \
    num_envs=4096 headless=True \
    ++manager_env.commands.motion.motion_lib_cfg.motion_file=data/hu_d04_motions/robot_filtered

# 4 GPUs
OMNI_KIT_ACCEPT_EULA=YES accelerate launch --num_processes=4 \
    gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/all_modes/sonic_hu_d04 \
    num_envs=4096 headless=True \
    ++manager_env.commands.motion.motion_lib_cfg.motion_file=data/hu_d04_motions/robot_filtered
```

### 7. Detach (survive SSH disconnect)
Wrap step 6 in `nohup` and redirect:
```bash
nohup bash -c "<command from step 6>" > ~/train.log 2>&1 &
echo $! > ~/train.pid
tail -f ~/train.log
```

---

## Target metrics (SONIC paper)

| metric | target | meaning |
|--------|--------|---------|
| `rewards/total` | > 3.0 | total accumulated reward |
| `rewards/tracking_vr_5point_local` | > 0.80 | VR-tracking quality |
| `success_rate` (eval) | > 0.97 | motions without early termination |
| `mpjpe_l` (eval) | < 30 mm | per-joint position error |

---

## Export ONNX after training

```bash
python gear_sonic/eval_agent_trl.py \
    +checkpoint=<path_to_best_checkpoint.pt> \
    +headless=True ++num_envs=1 +export_onnx_only=true
```

---

## Troubleshooting

| symptom | fix |
|---------|-----|
| `vulkaninfo` → `llvmpipe` / `ERROR_INCOMPATIBLE_DRIVER` | Wrong pod image; use Isaac Sim / Isaac Lab template |
| `libhdx.so: cannot open shared object file` | `LD_LIBRARY_PATH` for Omniverse exts (see step 4) |
| `CUDA error: out of memory` | Halve `num_envs` (8192 → 4096 → 2048 → 1024) |
| `quat_apply_inverse` not found in isaaclab | `setup_all.sh` already patches this — re-run it |
| Blackwell GPU + cu124 wheels fail | `setup_all.sh` auto-detects and uses cu128 |

## Re-filter the dataset with different keywords

Default filter drops motions matching `sitting`, `stair`, `handstand`, `cartwheel`, etc. (see `filter_and_copy_bones_data.py`). To change:

```bash
aws s3 cp s3://safesentinel-inc/hu_d04_motions/robot.tar.gz .
tar xzf robot.tar.gz
python ~/GR00T-WholeBodyControl/gear_sonic/data_process/filter_and_copy_bones_data.py \
    --source robot \
    --dest data/hu_d04_motions/robot_filtered_custom \
    --add-keywords my_keyword another_one \
    --workers 16
# then point --motion_file at robot_filtered_custom in the training command
```
