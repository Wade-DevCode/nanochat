#!/bin/bash

# 本脚本用于训练你自己的 GPT-2 级别 LLM（预训练 + 微调）
# 它设计为在空白 8XH100 GPU 节点上运行，完整流程约需 3 小时。

# 1) 启动示例（最简单）：
# bash runs/speedrun.sh
# 2) 在 screen 会话中启动（因为运行约需 3 小时）：
# screen -L -Logfile runs/speedrun.log -S speedrun bash runs/speedrun.sh
# 3) 启动并记录 wandb 日志，但请先参考下方配置 wandb：
# WANDB_RUN=speedrun screen -L -Logfile runs/speedrun.log -S speedrun bash runs/speedrun.sh

# 默认中间产物目录位于 ~/.cache/nanochat
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# 使用 uv 设置 Python venv

# 安装 uv（如果尚未安装）
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# 创建本地 .venv 虚拟环境（如果不存在）
[ -d ".venv" ] || uv venv
# 安装仓库依赖
uv sync --extra gpu
# 激活 venv，让 `python` 使用项目虚拟环境而不是系统 Python
source .venv/bin/activate

# -----------------------------------------------------------------------------
# wandb 设置
# 如果希望用 wandb 记录日志（很好用，推荐）。
# 1) 请确保先登录 wandb，例如运行：
#    `wandb login`
# 2) 运行脚本时设置 WANDB_RUN 环境变量，例如：
#    `WANDB_RUN=d26 bash speedrun.sh`
if [ -z "$WANDB_RUN" ]; then
    # 默认使用 "dummy"：这是特殊情况，会跳过 wandb 日志
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# 运行过程中会把 Markdown 报告写入 base dir 下的 report/ 目录。
# 该命令会清空目录，并写入包含系统信息和开始时间戳的头部章节。
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer

# 下载预训练数据集的前约 2B 字符
# 每个 data shard 约 250M 字符
# 因此此时下载 2e9 / 250e6 = 8 个 data shard
# 每个 shard 约 100MB 文本（压缩后），所以磁盘上约 800MB 数据
# 数据准备方式详见 dev/repackage_data_reference.py
python -m nanochat.dataset -n 8
# tokenizer 训练时，立刻在后台继续下载更多 shard
# GPT-2 能力级预训练大约需要 150 个 shard，额外加 20 个作为余量。
# 整个数据集最多可用 shard 数为 6542。
python -m nanochat.dataset -n 170 &
DATASET_DOWNLOAD_PID=$!
# 在约 2B 字符数据上训练 vocab size 为 2**15 = 32768 的 tokenizer
python -m scripts.tok_train
# 评估 tokenizer（报告压缩率等）
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Base model（预训练）
echo "等待数据集下载完成..."
wait $DATASET_DOWNLOAD_PID

# d24 模型（略微欠训练以超过 GPT-2 => 将 data:params ratio 从计算最优 10.5（默认）降低到 8）
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=24 --target-param-data-ratio=8 --device-batch-size=16 --fp8 --run=$WANDB_RUN
# 评估模型：CORE 指标、train/val 上的 BPB，以及生成采样
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval -- --device-batch-size=16

# -----------------------------------------------------------------------------
# SFT（教模型对话特殊 token、工具使用、多选题）

# 下载 2.3MB 合成身份对话，为 nanochat 注入人格
# 数据准备方式见 dev/gen_synthetic_data.py，也可了解如何轻松调整它
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# 运行 SFT 并评估模型
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- --device-batch-size=16 --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft

# 通过 CLI 和模型聊天！去掉 -p 可进入交互式聊天
# python -m scripts.chat_cli -p "Why is the sky blue?"

# 更好的是，通过漂亮的 ChatGPT 风格 WebUI 和你的模型聊天
# python -m scripts.chat_web

# -----------------------------------------------------------------------------
# 汇总所有章节，生成完整报告
# 输出为 report.md，并会复制到当前目录以便查看
python -m nanochat.report generate
