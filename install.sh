#!/bin/bash
# Tunix + Qwen3-1.7B GSM8K 依赖安装脚本

set -e

echo "========================================="
echo "Installing Tunix Dependencies for TPU"
echo "========================================="

# 检查 Python 版本
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
echo "Python version: $python_version"

if ! [[ "$python_version" =~ ^3\.(11|12|13)$ ]]; then
    echo "ERROR: Python 3.11+ required, found $python_version"
    exit 1
fi

# 安装核心依赖
echo "Installing Tunix from PyPI..."
pip install "google-tunix[prod]" --upgrade

# 验证安装
echo "Verifying installation..."
python3 -c "import tunix; print(f'Tunix installed successfully')" || {
    echo "ERROR: Tunix installation failed"
    exit 1
}

# 安装 HuggingFace CLI（用于下载模型）
echo "Installing HuggingFace CLI..."
pip install huggingface_hub[cli] --upgrade

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
python3 -c "import jax; print(f'JAX devices: {jax.devices()}')" || {
    echo "WARNING: JAX TPU not detected. Make sure you're on a TPU machine."
}

# 验证 SGLang-JAX 安装
echo "Verifying SGLang-JAX installation..."
python3 -c "import sglang_jax; print('SGLang-JAX installed successfully')" || {
    echo "WARNING: SGLang-JAX installation may have issues"
}

echo "========================================="
echo "Installation Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Run training script: bash scripts/train_qwen3_gsm8k.sh"
