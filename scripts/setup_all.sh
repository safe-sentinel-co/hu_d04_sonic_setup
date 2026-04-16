#!/bin/bash
set -e

# =============================================================================
# SONIC HU_D04 Training Setup Script
# Sets up everything needed to train GEAR-SONIC on the LimX HU_D04 humanoid.
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
WORK_DIR="$HOME"

echo "============================================"
echo "  SONIC HU_D04 Training Setup"
echo "============================================"

# ---- 0. Prerequisites check ----
echo ""
echo "[Step 0] Checking prerequisites..."

if ! nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found. NVIDIA GPU driver required."
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "  GPU: $GPU_NAME"

# Check for Vulkan support (CRITICAL)
if ! ldconfig -p 2>/dev/null | grep -q "libGLX_nvidia"; then
    echo ""
    echo "  WARNING: NVIDIA rendering libraries (Vulkan/GLX) NOT found!"
    echo "  Isaac Sim requires Vulkan even in headless mode."
    echo "  Your driver appears to be compute-only."
    echo ""
    echo "  FIX: Install the full NVIDIA driver (not compute-only):"
    echo "    Option A: sudo apt install libnvidia-gl-<VERSION>"
    echo "    Option B: Download the .run installer from nvidia.com and run:"
    echo "              sudo sh NVIDIA-Linux-x86_64-<VER>.run --no-kernel-module"
    echo "    Option C: Use a VM image with full NVIDIA driver pre-installed"
    echo ""
    echo "  Continuing setup, but training will FAIL without Vulkan."
    echo ""
fi

python3 --version || { echo "ERROR: Python3 required"; exit 1; }
PYTHON_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "  Python: $PYTHON_VER"

# ---- 1. System packages ----
echo ""
echo "[Step 1] Installing system packages..."
sudo apt update -qq
sudo apt install -y -qq python3-pip python3-venv python3.12 python3.12-venv python3.12-dev \
    git-lfs libgl1-mesa-glx libglib2.0-0 libegl1 libsm6 libxrender1 libxext6 libice6 \
    libvulkan1 libglu1-mesa libxt6 2>/dev/null
echo "  Done."

# ---- 2. Clone repos ----
echo ""
echo "[Step 2] Cloning repositories..."

if [ ! -d "$WORK_DIR/GR00T-WholeBodyControl" ]; then
    git clone https://github.com/NVlabs/GR00T-WholeBodyControl.git "$WORK_DIR/GR00T-WholeBodyControl"
else
    echo "  GR00T-WholeBodyControl already exists, skipping clone."
fi

if [ ! -d "$WORK_DIR/soma-retargeter" ]; then
    git clone https://github.com/NVIDIA/soma-retargeter.git "$WORK_DIR/soma-retargeter"
    cd "$WORK_DIR/soma-retargeter" && git lfs pull && cd -
else
    echo "  soma-retargeter already exists, skipping clone."
fi

if [ ! -d "$WORK_DIR/humanoid-description" ]; then
    git clone https://github.com/limxdynamics/humanoid-description.git "$WORK_DIR/humanoid-description"
else
    echo "  humanoid-description already exists, skipping clone."
fi
echo "  Done."

# ---- 3. Apply HU_D04 patches to GR00T ----
echo ""
echo "[Step 3] Applying HU_D04 integration to GR00T-WholeBodyControl..."
cd "$WORK_DIR/GR00T-WholeBodyControl"

# Copy new files
cp -r "$REPO_DIR/patches/gear_sonic/new_files/"* . 2>/dev/null || true

# Copy mesh files from humanoid-description
mkdir -p gear_sonic/data/assets/robot_description/meshes
cp -r "$WORK_DIR/humanoid-description/HU_D04_description/meshes/HU_D04_01" \
    gear_sonic/data/assets/robot_description/meshes/HU_D04_01

# Apply code modifications
git apply "$REPO_DIR/patches/gear_sonic/modifications.patch" 2>/dev/null || {
    echo "  Patch already applied or conflicts detected. Checking manually..."
    grep -q "hu_d04" gear_sonic/envs/manager_env/robots/__init__.py && echo "  Already patched." || {
        echo "  ERROR: Patch failed. Apply manually."
        exit 1
    }
}
echo "  Done."

# ---- 4. Apply HU_D04 patches to SOMA retargeter ----
echo ""
echo "[Step 4] Applying HU_D04 support to SOMA retargeter..."
cd "$WORK_DIR/soma-retargeter"

# Copy new config files
cp -r "$REPO_DIR/patches/soma_retargeter/new_files/"* . 2>/dev/null || true

# Copy MJCF and link meshes for SOMA
mkdir -p assets/robots/limx_hu_d04/mjcf
mkdir -p assets/robots/limx_hu_d04/meshes
cp "$WORK_DIR/GR00T-WholeBodyControl/gear_sonic/data/assets/robot_description/mjcf/hu_d04.xml" \
    assets/robots/limx_hu_d04/mjcf/
ln -sf "$WORK_DIR/GR00T-WholeBodyControl/gear_sonic/data/assets/robot_description/meshes/HU_D04_01" \
    assets/robots/limx_hu_d04/meshes/HU_D04_01

# Apply code modifications
git apply "$REPO_DIR/patches/soma_retargeter/modifications.patch" 2>/dev/null || {
    echo "  Patch already applied or conflicts detected."
    grep -q "LIMX_HU_D04" soma_retargeter/pipelines/utils.py && echo "  Already patched." || {
        echo "  ERROR: Patch failed. Apply manually."
        exit 1
    }
}
echo "  Done."

# ---- 5. Create training Python environment ----
echo ""
echo "[Step 5] Setting up training Python environment (sonic_env)..."

if [ ! -d "$WORK_DIR/sonic_env" ]; then
    python3 -m venv "$WORK_DIR/sonic_env"
fi
source "$WORK_DIR/sonic_env/bin/activate"
pip install --upgrade pip setuptools==75.8.2 2>/dev/null

# Detect GPU architecture for PyTorch CUDA version
GPU_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
echo "  GPU compute capability: $GPU_CC"

if [ "$GPU_CC" -ge "120" ] 2>/dev/null; then
    echo "  Blackwell GPU detected -- using PyTorch cu128"
    TORCH_INDEX="https://download.pytorch.org/whl/cu128"
elif [ "$GPU_CC" -ge "90" ] 2>/dev/null; then
    echo "  Hopper/Ada GPU detected -- using PyTorch cu124"
    TORCH_INDEX="https://download.pytorch.org/whl/cu124"
else
    echo "  Using PyTorch cu124"
    TORCH_INDEX="https://download.pytorch.org/whl/cu124"
fi

pip install torch torchvision --index-url "$TORCH_INDEX" 2>/dev/null
PIP_NO_BUILD_ISOLATION=false pip install isaaclab 2>/dev/null

# Fix numpy for Isaac Lab compatibility
pip install numpy==1.26.4 pillow==11.0.0 2>/dev/null

# Install gear_sonic and remaining deps
pip install lxml open3d mujoco 2>/dev/null
pip install -e "$WORK_DIR/GR00T-WholeBodyControl/gear_sonic/[training]" 2>/dev/null

# Patch Isaac Lab: add missing quat_apply_inverse
MATH_PY=$(find "$WORK_DIR/sonic_env" -path "*/isaaclab/utils/math.py" | head -1)
if [ -n "$MATH_PY" ] && ! grep -q "quat_apply_inverse" "$MATH_PY"; then
    echo "  Patching Isaac Lab: adding quat_apply_inverse..."
    python3 "$REPO_DIR/scripts/patch_isaaclab_math.py" "$MATH_PY"
fi

echo "  Done."

# ---- 6. Create SOMA retargeter Python environment ----
echo ""
echo "[Step 6] Setting up SOMA retargeter environment (soma_env)..."

if ! python3.12 --version &>/dev/null; then
    echo "  ERROR: Python 3.12 required for SOMA retargeter."
    echo "  Install with: sudo add-apt-repository ppa:deadsnakes/ppa && sudo apt install python3.12 python3.12-venv"
    exit 1
fi

if [ ! -d "$WORK_DIR/soma_env" ]; then
    python3.12 -m venv "$WORK_DIR/soma_env"
fi
source "$WORK_DIR/soma_env/bin/activate"
pip install --upgrade pip setuptools 2>/dev/null
pip install -e "$WORK_DIR/soma-retargeter" 2>/dev/null
pip install huggingface_hub 2>/dev/null
echo "  Done."

# ---- 7. Download checkpoints and SMPL data ----
echo ""
echo "[Step 7] Downloading model checkpoints and SMPL data..."
source "$WORK_DIR/sonic_env/bin/activate"
cd "$WORK_DIR/GR00T-WholeBodyControl"
if [ ! -f "sonic_release/last.pt" ]; then
    pip install huggingface_hub 2>/dev/null
    python download_from_hf.py --training
else
    echo "  Checkpoints already downloaded."
fi
echo "  Done."

# ---- 8. Verify joint mappings ----
echo ""
echo "[Step 8] Verifying HU_D04 joint mappings..."
source "$WORK_DIR/sonic_env/bin/activate"
cd "$WORK_DIR/GR00T-WholeBodyControl"
python gear_sonic/scripts/verify_hu_d04_mappings.py --mujoco-only 2>&1 | tail -5
python gear_sonic/scripts/verify_hu_d04_mappings.py 2>&1 | tail -3
echo "  Done."

echo ""
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo ""
echo "  1. RETARGET MOTION DATA (need Bones-SEED BVH access):"
echo "     source ~/soma_env/bin/activate"
echo "     cd ~/soma-retargeter"
echo "     python app/bvh_to_csv_converter.py --config assets/hu_d04_batch_config.json --viewer null"
echo ""
echo "  2. CONVERT TO PKL:"
echo "     source ~/sonic_env/bin/activate"
echo "     cd ~/GR00T-WholeBodyControl"
echo "     python gear_sonic/data_process/convert_hu_d04_csv_to_motion_lib.py \\"
echo "         --input ~/soma-retargeter/assets/motions/hu_d04_csv/ \\"
echo "         --output data/hu_d04_motions/robot --fps 30 --fps_source 120 --individual --num_workers 16"
echo "     python gear_sonic/data_process/filter_and_copy_bones_data.py \\"
echo "         --source data/hu_d04_motions/robot --dest data/hu_d04_motions/robot_filtered --workers 16"
echo ""
echo "  3. SMOKE TEST (requires Vulkan-capable GPU driver):"
echo "     source ~/sonic_env/bin/activate"
echo "     cd ~/GR00T-WholeBodyControl"
echo '     OV_LIB=$(find ~/.local/share/ov/data/exts/v2 -name "*.so" -exec dirname {} \; 2>/dev/null | sort -u | tr "\n" ":")'
echo '     export LD_LIBRARY_PATH="${OV_LIB}${LD_LIBRARY_PATH}"'
echo "     OMNI_KIT_ACCEPT_EULA=YES WANDB_MODE=disabled python gear_sonic/train_agent_trl.py \\"
echo "         +exp=manager/universal_token/all_modes/sonic_hu_d04 \\"
echo "         num_envs=1 headless=True \\"
echo "         ++algo.config.num_learning_iterations=5 \\"
echo "         ++manager_env.commands.motion.motion_lib_cfg.motion_file=data/hu_d04_motions/robot_filtered \\"
echo "         use_wandb=false"
echo ""
echo "  4. FULL TRAINING (4-8 GPUs):"
echo "     accelerate launch --num_processes=4 gear_sonic/train_agent_trl.py \\"
echo "         +exp=manager/universal_token/all_modes/sonic_hu_d04 \\"
echo "         num_envs=4096 headless=True \\"
echo "         ++manager_env.commands.motion.motion_lib_cfg.motion_file=data/hu_d04_motions/robot_filtered"
