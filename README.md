# John's Tunix - Qwen3-1.7B GSM8K Training

使用 Google Tunix 在 TPU 上训练 Qwen3-1.7B 模型完成 GSM8K 数学推理任务。

## 项目概述

- **模型**: Qwen3-1.7B ([Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B))
- **数据集**: GSM8K (小学数学应用题)
- **训练方法**: GRPO (Group Relative Policy Optimization) - 全参数训练
- **Rollout 引擎**: SGLang-JAX (高效推理)
- **硬件**: Google Cloud TPU v6e-8 (2x4 topology)
- **训练步数**: 1步 (测试配置)

## 快速开始

### 前置要求

1. Python 3.11+
2. Google Cloud 账号和项目 (civil-rarity-482610-s5)
3. TPU v6e-8 实例 (node-1, europe-west4-a)
4. GitHub 账号

### 安装步骤

#### 方式 1: 本地运行（如果有 TPU 访问）

```bash
# 1. 克隆本仓库
git clone https://github.com/john/john-tunix.git
cd john-tunix

# 2. 克隆 tunix (需要基础配置文件)
cd ..
git clone https://github.com/google/tunix.git

# 3. 安装依赖
cd john-tunix
bash install.sh

# 4. 运行训练
bash scripts/train_qwen3_gsm8k.sh
```

#### 方式 2: 通过 gcloud 在远程 TPU 运行（推荐）

```bash
# 1. 克隆本仓库
git clone https://github.com/john/john-tunix.git
cd john-tunix

# 2. 设置环境变量
export PROJECT_ID=civil-rarity-482610-s5

# 3. 运行 TPU 训练
bash setup_tpu.sh
```

脚本将自动：
- ✅ 验证 TPU 状态
- ✅ 克隆仓库到 TPU VM
- ✅ 安装依赖
- ✅ 启动训练

### 监控训练

#### 连接到 TPU

```bash
gcloud compute tpus tpu-vm ssh node-1 \
  --zone=europe-west4-a \
  --project=civil-rarity-482610-s5
```

#### 查看训练日志

```bash
# 在 TPU VM 上
tail -f /tmp/tensorboard/grpo_qwen3_1.7b/*.log
```

#### 检查 Checkpoint

```bash
# 在 TPU VM 上
ls -la /tmp/ckpts/qwen3_gsm8k/
```

#### 使用 TensorBoard

```bash
# 在 TPU 上启动 TensorBoard
tensorboard --logdir=/tmp/tensorboard/grpo_qwen3_1.7b/ --port=6006

# 本地端口转发
gcloud compute tpus tpu-vm ssh node-1 \
  --zone=europe-west4-a \
  --project=civil-rarity-482610-s5 \
  --ssh-flag="-L 6006:localhost:6006"

# 打开浏览器访问 http://localhost:6006
```

## 配置说明

### 关键参数

- `max_steps=1`: 测试运行，只训练1步验证流程
- `batch_size=1`: 每步处理1个样本
- `num_generations=2`: GRPO 每个 prompt 生成2个响应
- `rollout_engine="sglang"`: 使用 SGLang-JAX 进行高效推理
- `mesh.shape="(2,4)"`: TPU v6e-8 的拓扑配置
- **全参数训练**: 不使用 LoRA，直接训练所有参数

### 完整训练配置

如需完整训练（非测试），修改 `scripts/train_qwen3_gsm8k.sh`:

```bash
max_steps=3738  # 完整 epoch (7473 samples / batch_size=2)
batch_size=2
num_generations=4
```

或在运行时传递参数：

```bash
max_steps=3738 batch_size=2 bash scripts/train_qwen3_gsm8k.sh
```

## 项目结构

```
john-tunix/
├── install.sh                    # 依赖安装脚本
├── setup_tpu.sh                  # TPU 连接和运行脚本
├── scripts/
│   └── train_qwen3_gsm8k.sh     # 训练启动脚本
└── README.md                     # 本文件
```

## TPU 信息

- **名称**: node-1
- **区域**: europe-west4-a
- **类型**: v6e-8 (8个 TPU 芯片, 2x4 拓扑)
- **内存**: 每个芯片 16GB HBM
- **项目**: civil-rarity-482610-s5

## 技术细节

### 模型架构

Qwen3-1.7B 配置:
- 层数: 28
- 隐藏维度: 2048
- 中间维度: 6144
- 注意力头: 16
- KV 头: 8
- 词汇表: 151936

### GSM8K 数据集

- 训练集: 7473 道题
- 测试集: 1319 道题
- 格式: 数学应用题 + 答案

### GRPO 算法

Group Relative Policy Optimization:
- 对同一 prompt 生成多个响应
- 使用 group-wise advantage 计算奖励
- KL 散度惩罚防止过度偏离参考模型

### 训练参数

- **优化器**: AdamW
- **学习率**: 3e-6
- **调度器**: warmup_cosine_decay
- **Warmup 比例**: 0.1
- **梯度裁剪**: 0.1
- **权重衰减**: 0.1

## 常见问题

### Q: 如何增加训练步数？

A: 修改 `scripts/train_qwen3_gsm8k.sh` 中的 `max_steps` 变量，或在运行时传递：

```bash
max_steps=1000 bash scripts/train_qwen3_gsm8k.sh
```

### Q: 内存不足怎么办？

A:
1. 减小 `batch_size` 到 1
2. 减小 `num_generations` 到 2
3. 考虑使用 LoRA（需要修改训练脚本）

### Q: 训练失败，如何调试？

A:
1. 连接到 TPU: `gcloud compute tpus tpu-vm ssh node-1 --zone=europe-west4-a`
2. 检查 Python 进程: `ps aux | grep python`
3. 查看日志: `tail -f /tmp/tensorboard/grpo_qwen3_1.7b/*.log`
4. 验证 JAX 设备: `python3 -c "import jax; print(jax.devices())"`

### Q: 如何停止训练？

A:
1. 连接到 TPU
2. 查找进程: `ps aux | grep grpo_main`
3. 终止进程: `kill -9 <PID>`

## 性能优化建议

### 1. 增加批处理大小

```bash
batch_size=4 bash scripts/train_qwen3_gsm8k.sh
```

### 2. 使用梯度累积

修改脚本中的:
```bash
rl_training_config.gradient_accumulation_steps=4
```

### 3. 调整生成数量

```bash
# 减少生成数量以加快速度
grpo_config.num_generations=2

# 增加生成数量以提高质量
grpo_config.num_generations=8
```

## 参考资源

- [Tunix GitHub](https://github.com/google/tunix)
- [Tunix 文档](https://tunix.readthedocs.io/)
- [Qwen3 HuggingFace](https://huggingface.co/Qwen/Qwen3-1.7B)
- [GSM8K Dataset](https://github.com/openai/grade-school-math)
- [GRPO Paper](https://arxiv.org/abs/2402.03300)

## 许可证

本项目遵循 Apache 2.0 许可证。

## 致谢

- Google Tunix 团队
- Alibaba Qwen 团队
- OpenAI GSM8K 数据集
- Google Cloud TPU 团队
