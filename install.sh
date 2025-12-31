#!/bin/bash
# Tunix + Qwen3-1.7B GSM8K 依赖安装脚本
# 使用系统 Python 3.12 + venv

set -e

echo "========================================="
echo "Installing Tunix Dependencies for TPU"
echo "========================================="

# 0. 安装 Python 3.12（如果未安装）
if ! command -v python3.12 &> /dev/null; then
    echo "Installing Python 3.12..."
    sudo add-apt-repository ppa:deadsnakes/ppa -y
    sudo apt-get update
    sudo apt-get install -y python3.12 python3.12-venv python3.12-dev
fi

# 1. 创建 Python 3.12 虚拟环境
echo "Creating Python 3.12 virtual environment..."
rm -rf ~/tunix-venv
python3.12 -m venv ~/tunix-venv
source ~/tunix-venv/bin/activate

# 验证 Python 版本
python_version=$(python --version)
echo "✅ Using $python_version"

# 2. 升级 pip
echo "Upgrading pip..."
pip install --upgrade pip

# 3. 安装 JAX for TPU
echo "Installing JAX for TPU..."
pip install -U jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# 验证 JAX
python -c "import jax; print(f'JAX version: {jax.__version__}'); print(f'JAX devices: {jax.devices()}')"
echo "✅ JAX installed"

# 4. 安装 Flax 和其他 JAX 依赖
echo "Installing Flax, Optax, Chex..."
pip install -U flax optax chex

# 5. 安装 Tunix
echo "Installing Tunix..."
pip install google-tunix[prod]

# 验证 Tunix
python -c "import tunix; print('✅ Tunix installed successfully')" || {
    echo "❌ ERROR: Tunix installation failed"
    exit 1
}

# 6. 安装 HuggingFace CLI
echo "Installing HuggingFace CLI..."
pip install huggingface_hub[cli]

# 7. 安装 SGLang-JAX
echo "Installing SGLang-JAX..."
if [ ! -d "/tmp/sglang-jax" ]; then
    git clone https://github.com/sgl-project/sglang-jax.git /tmp/sglang-jax
fi
cd /tmp/sglang-jax/python
pip install -e .
cd -

# 验证 SGLang-JAX
python -c "import sglang_jax; print('✅ SGLang-JAX installed successfully')" || {
    echo "⚠️  WARNING: SGLang-JAX installation may have issues"
}

echo "========================================="
echo "Installation Complete!"
echo "========================================="
echo ""
echo "Virtual environment: ~/tunix-venv"
echo "To use in new sessions, run:"
echo "  source ~/tunix-venv/bin/activate"
echo ""
echo "Next steps:"
echo "1. Run test script: python plugin/test_basic.py"
echo "2. Run training: python scripts/train_qwen3_gsm8k.sh"
echo ""
