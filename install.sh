#!/bin/bash
# Tunix + Qwen3-1.7B GSM8K 依赖安装脚本
# 使用 Miniconda 创建隔离环境

set -e

echo "========================================="
echo "Installing Tunix Dependencies for TPU"
echo "========================================="

# 定义环境名称
ENV_NAME="tunix-env"
MINICONDA_DIR="$HOME/miniconda3"

# 安装 Miniconda（如果未安装）
if [ ! -d "$MINICONDA_DIR" ]; then
    echo "Installing Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p $MINICONDA_DIR
    rm /tmp/miniconda.sh
fi

# 初始化 conda
source "$MINICONDA_DIR/etc/profile.d/conda.sh"

# 删除旧环境（如果存在）并创建新环境
echo "Creating fresh conda environment: $ENV_NAME"
conda env remove -n $ENV_NAME -y 2>/dev/null || true
conda create -n $ENV_NAME python=3.11 -y

# 激活环境
conda activate $ENV_NAME

# 验证 Python 版本
echo "Python version: $(python --version)"

# 检查 JAX 是否预装（TPU VM 通常预装）
echo "Checking for system JAX..."
if python -c "import jax; print(f'Found system JAX {jax.__version__}')" 2>/dev/null; then
    echo "✅ System JAX detected"
    # TPU VM 上的 JAX 通常在系统 Python 中，需要安装到 conda 环境
fi

# 安装 JAX for TPU
echo "Installing JAX for TPU..."
pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# 安装核心依赖
echo "Installing Tunix from PyPI..."
pip install "google-tunix[prod]"

# 验证 Tunix 安装
echo "Verifying Tunix installation..."
python -c "import tunix; print('✅ Tunix installed successfully')" || {
    echo "❌ ERROR: Tunix installation failed"
    exit 1
}

# 安装 HuggingFace CLI（用于下载模型）
echo "Installing HuggingFace CLI..."
pip install huggingface_hub[cli]

# 安装 SGLang-JAX（用于高效 rollout）
echo "Installing SGLang-JAX..."
if [ ! -d "/tmp/sglang-jax" ]; then
    git clone https://github.com/sgl-project/sglang-jax.git /tmp/sglang-jax
fi
cd /tmp/sglang-jax/python
pip install -e .
cd -

# 检查 JAX TPU 支持
echo "Checking JAX TPU support..."
python -c "import jax; print(f'JAX version: {jax.__version__}'); print(f'JAX devices: {jax.devices()}')" || {
    echo "⚠️  WARNING: JAX TPU not detected. Make sure you're on a TPU machine."
}

# 验证 SGLang-JAX 安装
echo "Verifying SGLang-JAX installation..."
python -c "import sglang_jax; print('✅ SGLang-JAX installed successfully')" || {
    echo "⚠️  WARNING: SGLang-JAX installation may have issues"
}

echo "========================================="
echo "Installation Complete!"
echo "========================================="
echo ""
echo "Environment: $ENV_NAME"
echo "To activate: conda activate $ENV_NAME"
echo ""
echo "Next steps:"
echo "1. Run test script: python plugin/test_basic.py"
echo "2. Run training: python scripts/train_qwen3_gsm8k.sh"
echo ""
