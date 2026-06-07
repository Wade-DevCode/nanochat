#!/bin/bash

# 展示一个在 CPU（或 MacBook 的 MPS）上跑通部分代码路径的示例。
# 本脚本最后一次更新/调优时间为 2026 年 1 月 17 日。

# 运行方式：
# bash runs/runcpu.sh

# 注意：训练 LLM 需要 GPU 算力和经费。只靠 MacBook 走不了太远。
# 请把这个 run 当成教学/有趣演示，而不是预期能得到好效果的流程。
# 也可以手动逐条运行本脚本，把命令复制粘贴到终端中执行。

# 各种环境设置
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra cpu
source .venv/bin/activate
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# 在约 2B 字符上训练 tokenizer（在作者的 MacBook Pro M3 Max 上约 34 秒）
python -m nanochat.dataset -n 8
python -m scripts.tok_train --max-chars=2000000000
python -m scripts.tok_eval

# 训练一个小型 4 层模型
# 作者把这个 run 调到在 MacBook Pro M3 Max 上约 30 分钟完成。
# 要获得更好结果，可以尝试增加 num_iterations，或向你喜欢的 LLM 寻找其他思路。
python -m scripts.base_train \
    --depth=6 \
    --head-dim=64 \
    --window-pattern=L \
    --max-seq-len=512 \
    --device-batch-size=32 \
    --total-batch-size=16384 \
    --eval-every=100 \
    --eval-tokens=524288 \
    --core-metric-every=-1 \
    --sample-every=100 \
    --num-iterations=5000 \
    --run=$WANDB_RUN
python -m scripts.base_eval --device-batch-size=1 --split-tokens=16384 --max-per-task=16

# SFT（在作者的 MacBook Pro M3 Max 上约 10 分钟）
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
python -m scripts.chat_sft \
    --max-seq-len=512 \
    --device-batch-size=32 \
    --total-batch-size=16384 \
    --eval-every=200 \
    --eval-tokens=524288 \
    --num-iterations=1500 \
    --run=$WANDB_RUN

# 通过 CLI 和模型聊天
# 模型应该能回答法国首都是 Paris。
# 它甚至可能知道天空是蓝色的。
# 有时在提问前先说 Hi，模型会表现得更好。
# python -m scripts.chat_cli -p "What is the capital of France?"

# 通过漂亮的 ChatGPT 风格 WebUI 和模型聊天
# python -m scripts.chat_web
