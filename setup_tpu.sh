#!/bin/bash
# TPU Setup and Training via GitHub
# Usage: ./setup_tpu.sh --hf-token=<your_token> [--wandb-key=<key>] [--skip-install]

set -e

# ============== 配置 ==============
TPU_NAME="john-tpu-v6e-8"
ZONE="europe-west4-a"
PROJECT_ID="${PROJECT_ID:-civil-rarity-482610-s5}"
REPO_URL="https://github.com/demon2036/john-tunix.git"
TUNIX_URL="https://github.com/google/tunix.git"

# ============== 参数解析 ==============
HF_TOKEN=""
WANDB_KEY=""
SKIP_INSTALL=false

for arg in "$@"; do
  case $arg in
    --hf-token=*) HF_TOKEN="${arg#*=}" ;;
    --wandb-key=*) WANDB_KEY="${arg#*=}" ;;
    --skip-install) SKIP_INSTALL=true ;;
    --help|-h)
      echo "Usage: ./setup_tpu.sh --hf-token=<token> [options]"
      echo ""
      echo "Options:"
      echo "  --hf-token=TOKEN    HuggingFace token (required)"
      echo "  --wandb-key=KEY     Weights & Biases API key (optional)"
      echo "  --skip-install      Skip dependency installation"
      echo "  --help, -h          Show this help message"
      exit 0
      ;;
    *) echo "Unknown arg: $arg"; exit 1 ;;
  esac
done

# 检查必需参数
if [ -z "$HF_TOKEN" ]; then
  echo "ERROR: --hf-token is required"
  echo "Usage: ./setup_tpu.sh --hf-token=<token>"
  exit 1
fi

# ============== 远程脚本 ==============
REMOTE_SCRIPT=$(cat <<'REMOTE_EOF'
#!/bin/bash
set -e

HF_TOKEN="__HF_TOKEN__"
WANDB_KEY="__WANDB_KEY__"
SKIP_INSTALL="__SKIP_INSTALL__"

echo "========================================="
echo "TPU Training Environment Setup"
echo "========================================="

cd ~

# 1. Clone or pull john-tunix
if [ -d "john-tunix" ]; then
  echo "[1/4] Updating john-tunix..."
  cd john-tunix && git pull && cd ~
else
  echo "[1/4] Cloning john-tunix..."
  git clone https://github.com/demon2036/john-tunix.git
fi

# 2. Clone or pull tunix
if [ -d "tunix" ]; then
  echo "[2/4] Updating tunix..."
  cd tunix && git pull && cd ~
else
  echo "[2/4] Cloning tunix..."
  git clone https://github.com/google/tunix.git
fi

# 3. 智能检测安装
if [ "$SKIP_INSTALL" = "true" ]; then
  echo "[3/4] Skipping install (--skip-install flag)..."
elif [ -d "tunix-venv" ] && [ -f "tunix-venv/bin/activate" ]; then
  echo "[3/4] Virtual environment exists, skipping install..."
else
  echo "[3/4] Running install.sh (this may take a few minutes)..."
  cd ~/john-tunix && bash install.sh
fi

# 4. 激活 venv 并运行训练
echo "[4/4] Starting training..."
source ~/tunix-venv/bin/activate
export HF_TOKEN="$HF_TOKEN"
[ -n "$WANDB_KEY" ] && export WANDB_API_KEY="$WANDB_KEY"

cd ~/john-tunix
bash scripts/train_qwen3_gsm8k.sh

echo "========================================="
echo "Training completed!"
echo "========================================="
REMOTE_EOF
)

# 替换占位符
REMOTE_SCRIPT="${REMOTE_SCRIPT//__HF_TOKEN__/$HF_TOKEN}"
REMOTE_SCRIPT="${REMOTE_SCRIPT//__WANDB_KEY__/$WANDB_KEY}"
REMOTE_SCRIPT="${REMOTE_SCRIPT//__SKIP_INSTALL__/$SKIP_INSTALL}"

# ============== 执行 ==============
echo "========================================="
echo "TPU Training via GitHub"
echo "========================================="
echo "TPU: $TPU_NAME ($ZONE)"
echo "Project: $PROJECT_ID"
echo "Repo: $REPO_URL"
echo "========================================="
echo ""

gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
  --zone="$ZONE" \
  --project="$PROJECT_ID" \
  --command="$REMOTE_SCRIPT"
