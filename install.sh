#!/bin/bash
# Tunix + Qwen3-1.7B GSM8K 依赖安装脚本
# 参考: https://github.com/sgl-project/sglang-jax/blob/main/scripts/setup_tpu.sh

set -e

echo "========================================="
echo "Installing Tunix Dependencies for TPU"
echo "========================================="

# 1. 安装 Miniconda3（删除旧环境）
echo "Installing Miniconda3..."
rm -rf ~/miniconda3
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -u
rm Miniconda3-latest-Linux-x86_64.sh

# 初始化 conda
~/miniconda3/bin/conda init bash
eval "$(~/miniconda3/bin/conda shell.bash hook)"

# 配置 conda（禁用自动激活 base，接受 TOS）
conda config --set auto_activate_base false
conda config --set channel_priority flexible

# 创建 Python 3.11 环境（Tunix 和 SGLang-JAX 需要）
echo "Creating Python 3.11 environment..."
conda create -n base-python311 python=3.11 -y -c conda-forge
conda activate base-python311

echo "✅ Miniconda3 installed with Python 3.11"

# 2. 安装 JAX for TPU
echo "Installing JAX for TPU..."
pip install -U jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# 验证 JAX
python -c "import jax; print(f'JAX version: {jax.__version__}'); print(f'JAX devices: {jax.devices()}')"
echo "✅ JAX installed"

# 3. 安装 Flax 和其他 JAX 依赖
echo "Installing Flax, Optax, Chex..."
pip install -U flax optax chex

# 4. 安装 Tunix
echo "Installing Tunix..."
pip install google-tunix[prod]

# 验证 Tunix
python -c "import tunix; print('✅ Tunix installed successfully')" || {
    echo "❌ ERROR: Tunix installation failed"
    exit 1
}

# 5. 安装 HuggingFace CLI
echo "Installing HuggingFace CLI..."
pip install huggingface_hub[cli]

# 6. 安装 SGLang-JAX
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

# 7. 安装 Pillow-SIMD（可选，性能优化）
echo "Installing Pillow-SIMD for performance..."
conda install -c conda-forge -y libjpeg-turbo
pip uninstall -y pillow || true
CC="cc -mavx2" pip install -U --force-reinstall pillow-simd

echo "========================================="
echo "Installation Complete!"
echo "========================================="
echo ""
echo "To use in new sessions, run:"
echo "  eval \"\$(~/miniconda3/bin/conda shell.bash hook)\""
echo "  conda activate base-python311"
echo ""
echo "Next steps:"
echo "1. Run test script: python plugin/test_basic.py"
echo "2. Run training: python scripts/train_qwen3_gsm8k.sh"
echo ""
