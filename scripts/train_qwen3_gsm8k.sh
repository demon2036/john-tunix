#!/bin/bash
# Qwen3-1.7B GRPO Training on GSM8K

set -ex

# Check HF_TOKEN
if [ -z "$HF_TOKEN" ]; then
  echo "ERROR: HF_TOKEN environment variable not set"
  echo "Please set it with: export HF_TOKEN=your_token_here"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TUNIX_DIR="$(dirname "$PROJECT_ROOT")/tunix"

cd "$TUNIX_DIR"

python3 -m tunix.cli.grpo_main \
  tunix/cli/base_config.yaml \
  override_config_file="${PROJECT_ROOT}/configs/qwen3_1.7b_gsm8k.yaml" \
  reference_model_config.model_download_path="/tmp/models/qwen3" \
  rl_training_config.max_steps=${max_steps:-100} \
  reward_functions="['${PROJECT_ROOT}/plugin/reward_fn/gsm8k_qwen3.py']"
