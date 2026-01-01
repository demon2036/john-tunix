#!/bin/bash
# Automated TPU Setup Script for Qwen3 GRPO Training
# This script sets up a fresh TPU VM with all dependencies and configurations

set -e

echo "=== TPU Setup Started ==="

# Configuration - get from environment or prompt
if [ -z "$HF_TOKEN" ]; then
  echo "Warning: HF_TOKEN not set. You'll need to set it before running training."
  echo "  export HF_TOKEN=your_huggingface_token"
fi

if [ -z "$WANDB_API_KEY" ]; then
  echo "Warning: WANDB_API_KEY not set. You'll need to set it before running training."
  echo "  export WANDB_API_KEY=your_wandb_key"
fi

# Add environment variables to .bashrc for persistence (if they exist)
if [ -n "$HF_TOKEN" ] || [ -n "$WANDB_API_KEY" ]; then
  echo "Setting up environment variables..."
  cat >> ~/.bashrc << EOF
export HF_TOKEN=${HF_TOKEN}
export WANDB_API_KEY=${WANDB_API_KEY}
export PATH=\$HOME/.local/bin:\$PATH
EOF
  source ~/.bashrc
fi

# Clone repositories
echo "Cloning repositories..."
cd ~

if [ ! -d "john-tunix" ]; then
  git clone https://github.com/demon2036/john-tunix.git
else
  cd john-tunix && git pull && cd ~
fi

if [ ! -d "tunix" ]; then
  git clone https://github.com/google/tunix.git
else
  cd tunix && git pull && cd ~
fi

# Upgrade pip and fix packaging
echo "Upgrading pip and setuptools..."
python3 -m pip install --upgrade pip setuptools wheel 'packaging<22' --quiet

# Install all dependencies
echo "Installing Python dependencies (this may take a few minutes)..."
python3 -m pip install \
  flax \
  'jax[tpu]' \
  optax \
  wandb \
  tensorflow-datasets \
  'transformers<=4.57.1' \
  orbax-checkpoint \
  google-metrax \
  grain \
  qwix \
  omegaconf \
  python-dotenv \
  datasets \
  fsspec \
  huggingface_hub \
  jaxtyping \
  kagglehub \
  numba \
  pylatexenc \
  sentencepiece \
  sympy \
  hf_transfer \
  clu \
  etils \
  --quiet

echo "=== TPU Setup Complete ==="
echo ""
echo "Environment variables set:"
echo "  HF_TOKEN: ${HF_TOKEN:0:20}..."
echo "  WANDB_API_KEY: ${WANDB_API_KEY:0:20}..."
echo ""
echo "To start training, run:"
echo "  cd ~/john-tunix && bash scripts/train_qwen3_gsm8k.sh"
echo ""
