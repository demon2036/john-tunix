#!/bin/bash
# TPU 连接和运行脚本

set -e

TPU_NAME="node-1"
ZONE="europe-west4-a"
TPU_TYPE="v6e-8"
PROJECT_ID=${PROJECT_ID:-"civil-rarity-482610-s5"}

echo "========================================="
echo "TPU Setup and Training Launch"
echo "========================================="
echo "TPU Name: $TPU_NAME"
echo "Zone: $ZONE"
echo "TPU Type: $TPU_TYPE"
echo "Project ID: $PROJECT_ID"
echo "========================================="

# 检查是否设置了 PROJECT_ID
if [ -z "$PROJECT_ID" ]; then
    echo "ERROR: PROJECT_ID not set"
    echo "Please set it: export PROJECT_ID=your-gcp-project-id"
    exit 1
fi

# 检查 TPU 状态
echo "Checking TPU status..."
gcloud compute tpus tpu-vm describe $TPU_NAME \
  --zone=$ZONE \
  --project=$PROJECT_ID || {
    echo "ERROR: TPU not found or not accessible"
    exit 1
}

# 创建 TPU 启动脚本
cat > /tmp/tpu_train.sh << 'EOF'
#!/bin/bash
set -e

echo "========================================="
echo "TPU Training Environment Setup"
echo "========================================="

# 克隆仓库
cd ~
if [ ! -d "john-tunix" ]; then
    echo "Cloning john-tunix repository..."
    git clone https://github.com/john/john-tunix.git || {
        echo "ERROR: Failed to clone repository. Make sure it exists on GitHub."
        exit 1
    }
fi
cd john-tunix

# 克隆 tunix（如果需要）
if [ ! -d "../tunix" ]; then
    echo "Cloning tunix repository..."
    cd ~
    git clone https://github.com/google/tunix.git
    cd tunix
else
    echo "Updating tunix repository..."
    cd ../tunix
    git pull
fi

# 安装依赖
echo "Installing dependencies..."
cd ~/john-tunix
bash install.sh

# 运行训练
echo "Starting training..."
bash scripts/train_qwen3_gsm8k.sh

echo "========================================="
echo "Training completed!"
echo "========================================="
EOF

# 上传并执行脚本
echo "Uploading training script to TPU..."
gcloud compute tpus tpu-vm scp /tmp/tpu_train.sh $TPU_NAME:~/tpu_train.sh \
  --zone=$ZONE \
  --project=$PROJECT_ID

echo "Starting training on TPU..."
echo "NOTE: This will run in the foreground. Press Ctrl+C to detach (training will continue)."
echo ""

gcloud compute tpus tpu-vm ssh $TPU_NAME \
  --zone=$ZONE \
  --project=$PROJECT_ID \
  --command="bash ~/tpu_train.sh"

echo "========================================="
echo "Training launched on TPU!"
echo "========================================="
echo ""
echo "To monitor training:"
echo "  gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --project=$PROJECT_ID"
echo ""
echo "To view logs:"
echo "  tail -f /tmp/tensorboard/grpo_qwen3_1.7b/*.log"
