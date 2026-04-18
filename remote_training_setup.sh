#!/bin/bash
# Full end-to-end training setup on a Vulkan-capable remote pod.
# Run on the TRAINING BOX (not the retargeting box). Assumes Ubuntu 22.04,
# CUDA 12.x, NVIDIA GPU driver with full GL/Vulkan (not compute-only).
#
# Usage:
#   export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=...
#   export N_GPUS=4                          # or 8, or auto
#   export NUM_ENVS=4096                     # 2048 on 4090 24GB; 4096 on A100/L40; 8192 on H100
#   bash remote_training_setup.sh smoke      # smoke test (5 iters, ~5 min)
#   bash remote_training_setup.sh train      # full training (multi-day)
#
# Modes:
#   verify   - only run the Vulkan + nvidia-smi sanity check
#   install  - install everything, download dataset, STOP before training
#   smoke    - install then run 5-iter smoke test
#   train    - install (if needed) then launch full training in background (nohup)
set -euo pipefail
MODE="${1:-install}"

N_GPUS="${N_GPUS:-auto}"
NUM_ENVS="${NUM_ENVS:-4096}"
DATASET_BUCKET="${DATASET_BUCKET:-safesentinel-inc}"
DATASET_KEY="${DATASET_KEY:-hu_d04_motions/robot_filtered.tar.gz}"
REPO_DIR="${HOME}/hu_d04_sonic_setup"

if [ "$N_GPUS" = "auto" ]; then
    N_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
    [ "$N_GPUS" -lt 1 ] && N_GPUS=1
fi


# ---------- 0. Verify Vulkan-capable driver ----------
verify_env() {
    echo "[0] Verifying pod is Vulkan-capable..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq vulkan-tools git git-lfs curl unzip python3-pip
    if ! vulkaninfo --summary 2>&1 | grep -q "NVIDIA"; then
        echo
        echo "ERROR: Vulkan driver not functional on this pod."
        echo "       Isaac Sim will fail to start without it."
        echo "       Use a pod image with full NVIDIA driver"
        echo "       (Isaac Sim / Isaac Lab template on RunPod, Lambda, etc.)"
        vulkaninfo --summary 2>&1 | head -3
        exit 1
    fi
    echo "   Vulkan OK — $(vulkaninfo --summary 2>&1 | grep deviceName | head -1)"
    echo "[0] GPU check: ${N_GPUS} GPU(s) visible"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
}


# ---------- 1. Clone + install ----------
install_all() {
    verify_env
    echo
    echo "[1] Cloning setup repo..."
    if [ ! -d "$REPO_DIR" ]; then
        git clone https://github.com/safe-sentinel-co/hu_d04_sonic_setup.git "$REPO_DIR"
    else
        git -C "$REPO_DIR" pull --ff-only
    fi
    cd "$REPO_DIR"
    chmod +x scripts/setup_all.sh

    echo
    echo "[2] Running scripts/setup_all.sh (installs Isaac Lab, gear_sonic,"
    echo "    downloads SONIC checkpoints + SMPL data, ~30 GB)..."
    ./scripts/setup_all.sh 2>&1 | tee ~/setup.log

    echo
    echo "[3] Downloading retargeted dataset from s3://${DATASET_BUCKET}/${DATASET_KEY}..."
    pip install -q awscli
    : "${AWS_ACCESS_KEY_ID:?set AWS_ACCESS_KEY_ID before running}"
    : "${AWS_SECRET_ACCESS_KEY:?set AWS_SECRET_ACCESS_KEY before running}"
    mkdir -p "${HOME}/GR00T-WholeBodyControl/data/hu_d04_motions"
    cd "${HOME}/GR00T-WholeBodyControl/data/hu_d04_motions"
    aws s3 cp "s3://${DATASET_BUCKET}/${DATASET_KEY}" .
    tar xzf "$(basename "$DATASET_KEY")"
    rm -f "$(basename "$DATASET_KEY")"
    echo "   PKL count: $(find robot_filtered -name '*.pkl' | wc -l)"
    echo "   Session count: $(ls robot_filtered | wc -l)"
}


# ---------- 2. Build training command ----------
build_train_cmd() {
    local extra="$1"
    cat <<EOF
source ~/sonic_env/bin/activate
OV_LIB=\$(find ~/.local/share/ov/data/exts/v2 -name "*.so" -exec dirname {} \\; 2>/dev/null | sort -u | tr '\n' ':')
export LD_LIBRARY_PATH="\${OV_LIB}\${LD_LIBRARY_PATH}"
cd ~/GR00T-WholeBodyControl
OMNI_KIT_ACCEPT_EULA=YES accelerate launch --num_processes=${N_GPUS} \\
    gear_sonic/train_agent_trl.py \\
    +exp=manager/universal_token/all_modes/sonic_hu_d04 \\
    num_envs=${NUM_ENVS} headless=True \\
    ${extra} \\
    ++manager_env.commands.motion.motion_lib_cfg.motion_file=data/hu_d04_motions/robot_filtered
EOF
}


# ---------- 3. Smoke test ----------
run_smoke() {
    echo
    echo "[4] SMOKE TEST — 1 env, 5 iterations..."
    source "${HOME}/sonic_env/bin/activate"
    OV_LIB=$(find "${HOME}/.local/share/ov/data/exts/v2" -name "*.so" -exec dirname {} \; 2>/dev/null | sort -u | tr '\n' ':')
    export LD_LIBRARY_PATH="${OV_LIB}${LD_LIBRARY_PATH}"
    cd "${HOME}/GR00T-WholeBodyControl"
    OMNI_KIT_ACCEPT_EULA=YES WANDB_MODE=disabled python gear_sonic/train_agent_trl.py \
        +exp=manager/universal_token/all_modes/sonic_hu_d04 \
        num_envs=1 headless=True \
        ++algo.config.num_learning_iterations=5 \
        ++manager_env.commands.motion.motion_lib_cfg.motion_file=data/hu_d04_motions/robot_filtered \
        use_wandb=false
    echo
    echo "Smoke test completed. Inspect output for Isaac Sim boot,"
    echo "config load, terrain gen, and 5 iteration logs."
}


# ---------- 4. Full training ----------
run_train() {
    echo
    echo "[5] LAUNCHING FULL TRAINING"
    echo "    ${N_GPUS} GPU(s), num_envs=${NUM_ENVS}"
    echo "    Log: ~/train.log  (tail -f to watch)"
    echo "    PID: ~/train.pid"
    local cmd
    cmd=$(build_train_cmd "")
    nohup bash -c "$cmd" > "${HOME}/train.log" 2>&1 &
    echo $! > "${HOME}/train.pid"
    echo "   launched pid=$(cat ${HOME}/train.pid)"
    sleep 10
    echo
    echo "--- first 30 lines of training log ---"
    head -30 "${HOME}/train.log" 2>/dev/null || echo "(log not yet flushed)"
    echo
    echo "Detached with nohup — survives SSH disconnect."
    echo "Monitor: tail -f ~/train.log   |   watch -n5 nvidia-smi"
}


case "$MODE" in
    verify)  verify_env ;;
    install) install_all ;;
    smoke)   install_all; run_smoke ;;
    train)   install_all; run_train ;;
    train_only) run_train ;;  # skip install, just launch (when already installed)
    *) echo "Usage: $0 {verify|install|smoke|train|train_only}"; exit 1 ;;
esac
