#!/bin/bash
# Qwen3-1.7B GSM8K GRPO 训练脚本

set -x  # Enable xtrace for debugging

# 配置参数
batch_size=${batch_size:-1}
num_batches=${num_batches:-3738}
num_train_epochs=${num_train_epochs:-1}
warmup_ratio=${warmup_ratio:-0.1}
train_fraction=${train_fraction:-1.0}
max_steps=${max_steps:-1}  # 默认1步用于测试

echo "========================================="
echo "Qwen3-1.7B GSM8K GRPO Training"
echo "========================================="
echo "Parameters:"
echo "  Batch Size: $batch_size"
echo "  Num Batches: $num_batches"
echo "  Num Epochs: $num_train_epochs"
echo "  Max Steps: $max_steps"
echo "  Warmup Ratio: $warmup_ratio"
echo "  Train Fraction: $train_fraction"
echo "========================================="

# 计算 warmup steps
warmup_steps=$(awk "BEGIN {printf \"%.0f\", $warmup_ratio * $max_steps}")
echo "Warmup steps: $warmup_steps"

# 检查 HF Token
if [ -z "$HF_TOKEN" ]; then
    echo "WARNING: HF_TOKEN not set. Model download may fail."
    echo "Set it with: export HF_TOKEN=your_token"
fi

# 运行训练（需要在 tunix 目录下运行）
cd /home/john/test/tunix || { echo "ERROR: tunix directory not found"; exit 1; }

python3 -m tunix.cli.grpo_main \
  base_config.yaml \
  reference_model_config.model_name="qwen3-1.7b" \
  reference_model_config.model_id="Qwen/Qwen3-1.7B" \
  reference_model_config.model_source="huggingface" \
  reference_model_config.model_download_path="/tmp/models/qwen3-1.7b" \
  reference_model_config.intermediate_ckpt_dir="/tmp/intermediate_ckpt/qwen3_1.7b" \
  reference_model_config.mesh.shape="(2,4)" \
  reference_model_config.mesh.axis_names="('fsdp','tp')" \
  reference_model_config.rng_seed=42 \
  actor_model_config.mesh.shape="(2,4)" \
  actor_model_config.mesh.axis_names="('fsdp','tp')" \
  rollout_model_config.mesh.shape="(2,4)" \
  rollout_model_config.mesh.axis_names="('fsdp','tp')" \
  tokenizer_config.tokenizer_path="Qwen/Qwen3-1.7B" \
  tokenizer_config.tokenizer_type="huggingface" \
  tokenizer_config.add_bos=false \
  tokenizer_config.add_eos=true \
  dataset_name="gsm8k" \
  batch_size=$batch_size \
  num_batches=$num_batches \
  num_test_batches=100 \
  num_train_epochs=$num_train_epochs \
  rl_training_config.actor_optimizer_config.opt_type="adamw" \
  rl_training_config.actor_optimizer_config.peak_value=3e-6 \
  rl_training_config.actor_optimizer_config.schedule_type="warmup_cosine_decay_schedule" \
  rl_training_config.actor_optimizer_config.init_value=0.0 \
  rl_training_config.actor_optimizer_config.end_value=0.0 \
  rl_training_config.actor_optimizer_config.warmup_ratio=$warmup_ratio \
  rl_training_config.actor_optimizer_config.warmup_steps=$warmup_steps \
  rl_training_config.actor_optimizer_config.decay_steps=$max_steps \
  rl_training_config.actor_optimizer_config.b1=0.9 \
  rl_training_config.actor_optimizer_config.b2=0.99 \
  rl_training_config.actor_optimizer_config.weight_decay=0.1 \
  rl_training_config.actor_optimizer_config.max_grad_norm=0.1 \
  rl_training_config.eval_every_n_steps=1 \
  rl_training_config.max_steps=$max_steps \
  rl_training_config.metrics_logging_options.log_dir="/tmp/tensorboard/grpo_qwen3_1.7b" \
  rl_training_config.metrics_logging_options.flush_every_n_steps=1 \
  rl_training_config.checkpointing_options.save_interval_steps=1 \
  rl_training_config.checkpointing_options.max_to_keep=2 \
  rl_training_config.profiler_options={} \
  rollout_config.total_generation_steps=768 \
  rollout_config.max_prompt_length=256 \
  rollout_config.temperature=0.9 \
  rollout_config.top_p=1.0 \
  rollout_config.top_k=50 \
  rollout_engine="sglang" \
  offload_to_cpu=false \
  grpo_config.num_generations=2 \
  grpo_config.num_iterations=1 \
  grpo_config.beta=0.08 \
  grpo_config.epsilon=0.2 \
  reward_functions="['tunix/cli/reward_fn/gsm8k.py']"

echo "========================================="
echo "Training Complete!"
echo "========================================="
