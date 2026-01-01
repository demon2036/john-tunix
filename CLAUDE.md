# CLAUDE.md - Qwen3-1.7B GSM8K Training Guide

这是 Claude Code 的项目指南，包含 TPU 训练的关键配置、常见问题和解决方案。

## 重要发现

### TPU 镜像要求 (关键!)

| 镜像 | 能否运行 | 说明 |
|------|---------|------|
| `v2-alpha-tpuv6e` | ✅ 可以 | 正确的 v6e TPU 镜像 |
| `tpu-ubuntu2204-base` | ❌ 不行 | 基础镜像，JAX 无法检测 TPU |

**检查当前 TPU 镜像:**
```bash
gcloud compute tpus tpu-vm describe john-tpu-v6e-16 \
  --zone=europe-west4-a \
  --format='yaml(runtimeVersion)'
```

**如果镜像错误，必须删除重建:**
```bash
# 删除
gcloud compute tpus tpu-vm delete john-tpu-v6e-16 \
  --zone=europe-west4-a --quiet

# 重建 (使用正确镜像)
gcloud compute tpus tpu-vm create john-tpu-v6e-16 \
  --zone=europe-west4-a \
  --accelerator-type=v6e-16 \
  --version=v2-alpha-tpuv6e \
  --preemptible  # 可选，更容易获得资源
```

### 内存配置 (防止 OOM)

Qwen3-1.7B 词表很大 (151,936 tokens)，需要特殊配置：

| 配置 | 安全值 | 原始值 | 说明 |
|------|--------|--------|------|
| `batch_size` | **1** | 16 | 必须减小 |
| `num_generations` | **4** | 8 | 必须减小 |
| `mesh` | (4,2) | (4,2) | 4 FSDP + 2 TP |

**OOM 错误示例:**
```
RESOURCE_EXHAUSTED: Attempting to reserve 30.31G. Only 29.35G free.
```

**解决方案 - 运行时覆盖:**
```bash
cd ~/tunix && python3 -m tunix.cli.grpo_main \
  tunix/cli/base_config.yaml \
  override_config_file="/home/john/john-tunix/configs/qwen3_1.7b_gsm8k.yaml" \
  batch_size=1 \
  grpo_config.num_generations=4 \
  rl_training_config.max_steps=10
```

## TPU 管理命令

### 查看 TPU 列表
```bash
gcloud compute tpus tpu-vm list --zone=europe-west4-a
```

### 查看 TPU 详情
```bash
gcloud compute tpus tpu-vm describe john-tpu-v6e-16 \
  --zone=europe-west4-a \
  --format='yaml(runtimeVersion,state,acceleratorConfig)'
```

### SSH 连接
```bash
# 单节点
gcloud compute tpus tpu-vm ssh john-tpu-v6e-16 \
  --zone=europe-west4-a \
  --project=civil-rarity-482610-s5

# 所有节点 (多主机)
gcloud compute tpus tpu-vm ssh john-tpu-v6e-16 \
  --zone=europe-west4-a \
  --project=civil-rarity-482610-s5 \
  --worker=all --command="hostname"
```

### 创建 TPU (正确方式)
```bash
gcloud compute tpus tpu-vm create john-tpu-v6e-16 \
  --zone=europe-west4-a \
  --project=civil-rarity-482610-s5 \
  --accelerator-type=v6e-16 \
  --version=v2-alpha-tpuv6e \
  --preemptible
```

### 删除 TPU
```bash
gcloud compute tpus tpu-vm delete john-tpu-v6e-16 \
  --zone=europe-west4-a --quiet
```

## 部署流程

### 方式 1: 使用 setup_tpu.sh (推荐)

```bash
./setup_tpu.sh --hf-token=your_hf_token_here
```

参数:
- `--hf-token=xxx` - HuggingFace token (必需)
- `--wandb-key=xxx` - Weights & Biases key (可选)
- `--skip-install` - 跳过依赖安装

### 方式 2: 手动部署

```bash
# 1. SSH 到 TPU (所有节点)
gcloud compute tpus tpu-vm ssh john-tpu-v6e-16 --zone=europe-west4-a --worker=all --command='
cd ~ && git clone https://github.com/demon2036/john-tunix.git
git clone https://github.com/google/tunix.git
cd john-tunix && bash install.sh
'

# 2. 验证 JAX (在所有节点)
gcloud compute tpus tpu-vm ssh john-tpu-v6e-16 --zone=europe-west4-a --worker=all --command='
source ~/tunix-venv/bin/activate
python -c "import jax; print(jax.devices())"
'

# 3. 运行训练 (在所有节点同时执行)
gcloud compute tpus tpu-vm ssh john-tpu-v6e-16 --zone=europe-west4-a --worker=all --command='
source ~/tunix-venv/bin/activate
export HF_TOKEN="your_token"
cd ~/john-tunix && bash scripts/train_qwen3_gsm8k.sh
'
```

## 验证 JAX/TPU

正确输出 (每个节点):
```
Backend: tpu
Device count: 16  # 多主机总共 16 芯片
Devices: [TpuDevice(id=0, ...), TpuDevice(id=1, ...), ...]
```

错误输出 (镜像问题):
```
RuntimeError: Unable to initialize backend 'tpu': INTERNAL: Failed to get global TPU topology.
```

## 训练性能

10 步验证训练结果 (batch_size=1, num_generations=4):

| Step | Loss | 速度 |
|------|------|------|
| 1 | 0.000057 | 123s (编译) |
| 2-10 | ~0.00004 | ~37-40s |

总时间: 约 7 分钟 (10 步)

## 文件结构

```
john-tunix/
├── configs/
│   └── qwen3_1.7b_gsm8k.yaml   # GRPO 训练配置
├── plugin/
│   └── reward_fn/
│       └── gsm8k_qwen3.py      # GSM8K 奖励函数
├── scripts/
│   ├── train_qwen3_gsm8k.sh    # 主训练脚本
│   ├── test_grpo_qwen3.sh      # 3 步测试
│   └── test_wandb.sh           # Wandb 测试
├── install.sh                   # 依赖安装
├── setup_tpu.sh                 # TPU 部署脚本
├── CLAUDE.md                    # 本文件
└── README.md                    # 项目说明
```

## 配置文件说明

### configs/qwen3_1.7b_gsm8k.yaml

关键参数:
```yaml
# Mesh 配置 - 4 FSDP + 2 TP (用于大词表)
reference_model_config:
  mesh:
    shape: "(4,2)"
    axis_names: "('fsdp','tp')"

# 批处理 - 需要减小防止 OOM
batch_size: 16  # 实际运行时覆盖为 1

# GRPO 配置
grpo_config:
  num_generations: 8  # 实际运行时覆盖为 4
  beta: 0.08
  epsilon: 0.2

# Rollout 配置
rollout_config:
  total_generation_steps: 2048
  max_prompt_length: 256
  temperature: 0.9
```

## 常见问题

### Q: 多主机 TPU 错误 "non-addressable devices"

A: 这是多主机 JAX 的常见问题。运行修复脚本：
```bash
# 在所有节点上运行
gcloud compute tpus tpu-vm ssh john-tpu-v6e-16 --zone=europe-west4-a \
  --worker=all --command='
  source ~/tunix-venv/bin/activate
  cd ~/john-tunix
  python3 scripts/fix_multihost.py
  python3 scripts/fix_sharding.py
'
```

修复的内容：
- `fix_multihost.py` - 修复 sampler.py 使用 `process_allgather(x, tiled=True)`
- `fix_sharding.py` - 修复 sharding_utils.py 将全局数组转换为本地数据

### Q: JAX 报 "Failed to get global TPU topology"

A: TPU 镜像错误。必须使用 `v2-alpha-tpuv6e`，不能用 `tpu-ubuntu2204-base`。

解决: 删除并重建 TPU (见上方命令)。

### Q: OOM (RESOURCE_EXHAUSTED)

A: 减小 batch_size 和 num_generations:
```bash
batch_size=1 grpo_config.num_generations=4
```

### Q: install.sh 在 JAX 验证时卡住

A: install.sh 中的 `jax.devices()` 在 SSH 会话中可能失败。已修复为只检查版本。

### Q: 如何查看训练日志

A:
```bash
# 实时查看
gcloud compute tpus tpu-vm ssh john-tpu-v6e-16 --zone=europe-west4-a \
  --command='tail -f ~/train.log'

# 查看所有节点
gcloud compute tpus tpu-vm ssh john-tpu-v6e-16 --zone=europe-west4-a \
  --worker=all --command='tail -20 ~/train.log'
```

### Q: 如何停止训练

A:
```bash
# 找进程 (所有节点)
gcloud compute tpus tpu-vm ssh john-tpu-v6e-16 --zone=europe-west4-a \
  --worker=all --command='ps aux | grep grpo_main'

# 杀进程 (所有节点)
gcloud compute tpus tpu-vm ssh john-tpu-v6e-16 --zone=europe-west4-a \
  --worker=all --command='pkill -f grpo_main'
```

## 凭证信息

| 凭证 | 用途 |
|------|------|
| HF_TOKEN | 下载 Qwen3-1.7B 模型 (设置为环境变量) |
| WANDB_API_KEY | 训练日志记录 (项目: ultrathink) |

**注意**: 凭证应通过环境变量或命令行参数传递，不要保存在代码中。

## TPU 信息

| 属性 | 值 |
|------|-----|
| 名称 | john-tpu-v6e-16 |
| 备用 | node-1 |
| 区域 | europe-west4-a |
| 类型 | v6e-16 (16 芯片, 4x4 拓扑, 多主机) |
| 镜像 | v2-alpha-tpuv6e |
| 项目 | civil-rarity-482610-s5 |
| 内存 | 每芯片 ~31 GiB HBM |
| 多主机 | ✅ 需要 --worker=all |

## 技术细节

### Qwen3-1.7B 配置
- 层数: 28
- 隐藏维度: 2048
- 词汇表: 151,936 (很大，需要 TP)

### GSM8K 数据集
- 训练集: 7,473 题
- 测试集: 1,319 题

### GRPO 训练参数
- 优化器: AdamW
- 学习率: 3e-6
- 调度: warmup_cosine_decay
- Warmup: 10%
- 梯度裁剪: 0.1

## 参考

- [Tunix GitHub](https://github.com/google/tunix)
- [Qwen3 HuggingFace](https://huggingface.co/Qwen/Qwen3-1.7B)
- [GSM8K Dataset](https://github.com/openai/grade-school-math)
