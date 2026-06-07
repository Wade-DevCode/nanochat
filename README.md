# nanochat 中文版

![nanochat logo](dev/nanochat.png)
![scaling laws](dev/scaling_laws_jan26.png)

nanochat 是一个用于训练 LLM 的极简实验框架。它设计为在单个 GPU 节点上运行，代码尽量精简、易改，并覆盖 LLM 的主要阶段：分词器训练、预训练、微调、评测、推理，以及聊天 Web UI。举例来说，你可以只花 48 美元左右（约 2 小时 8XH100 GPU 节点时间）训练出一个具备 GPT-2 能力级别的 LLM，而 GPT-2 在 2019 年的训练成本约为 43,000 美元；训练完成后，还可以在类似 ChatGPT 的网页界面里和它聊天。如果使用 spot 实例，总成本可能接近 15 美元。更一般地说，nanochat 默认配置好了一个“计算最优”模型迷你系列，只需要调一个复杂度旋钮：`--depth`，也就是 GPT Transformer 模型的层数（GPT-2 能力大约对应 depth 26）。其他超参数（Transformer 宽度、head 数、学习率调整、训练长度、权重衰减等）都会按较优方式自动计算。

关于仓库问题，推荐使用 Devin/Cognition 的 [DeepWiki](https://deepwiki.com/karpathy/nanochat) 提问，或使用 [Discussions](https://github.com/karpathy/nanochat/discussions)，也可以到 Discord 的 [#nanochat](https://discord.com/channels/1020383067459821711/1427295580895314031) 频道交流。

## Time-to-GPT-2 排行榜

目前开发重点主要在预训练阶段，因为它消耗的计算量最多。受 modded-nanogpt 仓库启发，为了激励进展和社区协作，nanochat 维护了一个“GPT-2 speedrun”排行榜：用 DCLM CORE 分数衡量，把 nanochat 模型训练到 GPT-2 级别能力所需的真实墙钟时间。[runs/speedrun.sh](runs/speedrun.sh) 脚本始终代表当前参考训练流程：训练 GPT-2 级别模型并和它对话。当前排行榜如下：

| # | 时间 | val_bpb | CORE | 描述 | 日期 | Commit | 贡献者 |
|---|-------------|---------|------|-------------|------|--------|--------------|
| 0 | 168 小时 | - | 0.2565 | 原始 OpenAI GPT-2 checkpoint | 2019 | - | OpenAI |
| 1 | 3.04 | 0.74833 | 0.2585 | d24 baseline，略微过训练 | Jan 29 2026 | 348fbb3 | @karpathy |
| 2 | 2.91 | 0.74504 | 0.2578 | d26 略微欠训练 **+fp8** | Feb 2 2026 | a67eba3 | @karpathy |
| 3 | 2.76 | 0.74645 | 0.2602 | 将总 batch size 提升到 1M tokens | Feb 5 2026 | 2c062aa | @karpathy |
| 4 | 2.02 | 0.71854 | 0.2571 | 数据集切换到 NVIDIA ClimbMix | Mar 4 2026 | 324e69c | @ddudek @karpathy |
| 5 | 1.80 | 0.71808 | 0.2690 | autoresearch [round 1](https://x.com/karpathy/status/2031135152349524125) | Mar 9 2026 | 6ed7d1d | @karpathy |
| 6 | 1.65 | 0.71800 | 0.2626 | autoresearch round 2 | Mar 14 2026 | a825e63 | @karpathy |

我们最关心的主要指标是“time to GPT-2”：在一个 8XH100 GPU 节点上，超过 GPT-2 (1.6B) CORE 指标所需的墙钟时间。GPT-2 的 CORE 分数是 0.256525。2019 年训练 GPT-2 约花费 43,000 美元；由于过去 7 年整个技术栈的进步，现在可以在远低于 100 美元的成本内快得多地完成同等能力训练（例如当前约 3 美元/GPU/小时，8XH100 节点约 24 美元/小时，2 小时约 48 美元）。

如何解读和贡献排行榜，请参见 [dev/LEADERBOARD.md](dev/LEADERBOARD.md)。

## 快速开始

### 安装

nanochat 使用 [uv](https://docs.astral.sh/uv/) 管理依赖。安装：

```bash
uv sync --extra gpu    # CUDA (A100/H100 等) 使用
uv sync --extra cpu    # 或仅 CPU / MPS 使用
source .venv/bin/activate
```

开发环境（增加 pytest、matplotlib、ipykernel、transformers 等）：

```bash
uv sync --extra gpu --group dev
```

### 复现 GPT-2 并和它聊天

最有趣的玩法是训练你自己的 GPT-2，并和它聊天。完整流水线都在单个文件 [runs/speedrun.sh](runs/speedrun.sh) 中，该脚本设计为在 8XH100 GPU 节点上运行。从你喜欢的云服务商启动一个新的 8XH100 GPU 机器（例如作者使用并喜欢 [Lambda](https://lambda.ai/service/gpu-cloud)），然后启动训练脚本：

```bash
bash runs/speedrun.sh
```

建议在 screen 会话中运行，因为整个过程大约需要 3 小时。完成后，你可以通过类似 ChatGPT 的 Web UI 和模型聊天。再次确认本地 uv 虚拟环境已经激活（运行 `source .venv/bin/activate`），然后启动服务：

```bash
python -m scripts.chat_web
```

然后访问命令行显示的 URL。注意要用正确地址访问，例如在 Lambda 上使用当前节点的公网 IP 加端口，如 [http://209.20.xxx.xxx:8000/](http://209.20.xxx.xxx:8000/) 等。接着就像平常和 ChatGPT 聊天一样和你的 LLM 对话：让它写故事或诗，问它“你知道我是谁吗”观察幻觉，问它为什么天空是蓝色的，或者为什么是绿色的。这个 speedrun 是 4e19 FLOPs 能力级别的模型，所以有点像在和幼儿园小朋友聊天 :)。

---

<img width="2672" height="1520" alt="image" src="https://github.com/user-attachments/assets/ed39ddf8-2370-437a-bedc-0f39781e76b5" />

---

补充说明：

- 代码在 Ampere 8XA100 GPU 节点上也可以正常运行，只是会慢一些。
- 去掉 `torchrun` 后，所有代码也可以在单 GPU 上正常运行，并产生几乎相同的结果（代码会自动切换到梯度累积），但需要等待 8 倍时间。
- 如果你的 GPU 显存少于 80GB，需要调整部分超参数，否则会 OOM / 显存不足。查找脚本里的 `--device-batch-size`，逐步减小直到能跑起来。例如从 32（默认）减到 16、8、4、2，甚至 1。再小的话，你就需要更了解自己在做什么，并更有创造性地调整。
- 大部分代码都是比较常规的 PyTorch，所以理论上应能在任何支持 PyTorch 的设备上运行，例如 xpu、mps 等；但作者没有亲自覆盖所有路径，因此可能会有尖角。

## 研究

如果你是研究者，并希望帮助改进 nanochat，值得关注的两个脚本是 [runs/scaling_laws.sh](runs/scaling_laws.sh) 和 [runs/miniseries.sh](runs/miniseries.sh)。相关说明见 [Jan 7 miniseries v1](https://github.com/karpathy/nanochat/discussions/420)。快速实验时（约 5 分钟预训练），作者喜欢训练 12 层模型（GPT-1 规模），例如：

```bash
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=12 \
    --run="d12" \
    --model-tag="d12" \
    --core-metric-every=999999 \
    --sample-every=-1 \
    --save-every=-1 \
```

这会使用 wandb（运行名为 `"d12"`），只在最后一步运行 CORE 指标，不采样也不保存中间 checkpoint。作者通常会改一处代码，重新跑一次 d12（或 d16 等），看它是否有帮助，以此迭代。判断一次 run 是否有帮助时，作者会关注 wandb 图表：

1. `val_bpb`（用 bits per byte 表示、与词表大小无关的验证损失）随 `step`、`total_training_time` 和 `total_training_flops` 的变化。
2. `core_metric`（DCLM CORE 分数）。
3. VRAM 使用率、`train/mfu`（Model FLOPS utilization）、`train/tok_per_sec`（训练吞吐）。

示例见 [这里](https://github.com/karpathy/nanochat/pull/498#issuecomment-3850720044)。

需要注意的是，nanochat 的编写和配置围绕一个复杂度旋钮：Transformer 的 depth。这个整数会自动决定其他所有超参数（Transformer 宽度、head 数、学习率调整、训练长度、权重衰减等），让训练出的模型接近计算最优。理念是用户不需要思考或设置这些细节，只需用 `--depth` 请求更小或更大的模型，其他都会“直接可用”。扫过不同 depth，就能得到 nanochat 的计算最优模型迷你系列。目前最受关注的 GPT-2 能力模型，在当前代码下大约落在 d24-d26 区间。但任何候选改动都必须足够有原则，能适用于所有 depth 设置。

## 在 CPU / MPS 上运行

[runs/runcpu.sh](runs/runcpu.sh) 展示了在 CPU 或 Apple Silicon 上运行的极简示例。它会大幅缩小要训练的 LLM，使其能在几十分钟训练时间内跑完。用这种方式不会得到很强的结果。

## 精度 / dtype

nanochat 不使用 `torch.amp.autocast`。相反，精度通过一个全局 `COMPUTE_DTYPE` 显式管理（定义在 `nanochat/common.py`）。默认值会根据硬件自动检测：

| 硬件 | 默认 dtype | 原因 |
|----------|--------------|-----|
| CUDA SM 80+ (A100, H100, ...) | `bfloat16` | 原生 bf16 tensor cores |
| CUDA SM < 80 (V100, T4, ...) | `float32` | 无 bf16；可通过 `NANOCHAT_DTYPE=float16` 使用 fp16（使用 GradScaler） |
| CPU / MPS | `float32` | 无低精度 tensor cores |

可以用 `NANOCHAT_DTYPE` 环境变量覆盖默认值：

```bash
NANOCHAT_DTYPE=float32 python -m scripts.chat_cli -p "hello"   # 强制 fp32
NANOCHAT_DTYPE=bfloat16 torchrun --nproc_per_node=8 -m scripts.base_train  # 强制 bf16
```

工作方式：模型权重以 fp32 存储（保证优化器精度），但自定义 `Linear` 层会在 forward 时把权重转换到 `COMPUTE_DTYPE`。Embedding 直接以 `COMPUTE_DTYPE` 存储以节省内存。这带来了与 autocast 相同的混合精度收益，但我们能完全显式地控制哪些操作以何种精度运行。

注意：`float16` 训练会在 `base_train.py` 中自动启用 `GradScaler`，以防梯度下溢。SFT 也支持这一点，但 RL 目前不支持。fp16 推理在各处都可以正常工作。

## 指南

作者发布过一些可能有帮助的指南，按时间从新到旧排列：

- [Feb 1 2026: Beating GPT-2 for <<$100: the nanochat journey](https://github.com/karpathy/nanochat/discussions/481)
- [Jan 7 miniseries v1](https://github.com/karpathy/nanochat/discussions/420) 记录了第一批 nanochat 模型迷你系列。
- 为 nanochat 添加新能力，见 [Guide: counting r in strawberry (and how to add abilities generally)](https://github.com/karpathy/nanochat/discussions/164)。
- 自定义你的 nanochat，见 Discussions 中的 [Guide: infusing identity to your nanochat](https://github.com/karpathy/nanochat/discussions/139)，它说明了如何通过合成数据生成以及将数据混入 SFT 阶段，来调整 nanochat 的人格。
- [Oct 13 2025: original nanochat post](https://github.com/karpathy/nanochat/discussions/1) 是介绍 nanochat 的原始帖子，不过现在包含一些过时信息，模型也比当前 master 老很多、效果更差。

## 文件结构

```text
.
├── LICENSE
├── README.md
├── dev
│   ├── gen_synthetic_data.py       # 身份设定用合成数据示例
│   ├── generate_logo.html
│   ├── nanochat.png
│   └── repackage_data_reference.py # 预训练数据 shard 生成
├── nanochat
│   ├── __init__.py                 # 空文件
│   ├── checkpoint_manager.py       # 保存/加载模型 checkpoint
│   ├── common.py                   # 杂项小工具和易用性辅助
│   ├── core_eval.py                # 评估 base model 的 CORE 分数（DCLM 论文）
│   ├── dataloader.py               # 分词用分布式数据加载器
│   ├── dataset.py                  # 预训练数据下载/读取工具
│   ├── engine.py                   # 使用 KV Cache 的高效模型推理
│   ├── execution.py                # 允许 LLM 把 Python 代码作为工具执行
│   ├── gpt.py                      # GPT nn.Module Transformer
│   ├── logo.svg
│   ├── loss_eval.py                # 评估 bits per byte（而非普通 loss）
│   ├── optim.py                    # AdamW + Muon 优化器，支持单 GPU 和分布式
│   ├── report.py                   # 编写 nanochat Report 的工具
│   ├── tokenizer.py                # GPT-4 风格 BPE Tokenizer 包装器
│   └── ui.html                     # nanochat 前端 HTML/CSS/JS
├── pyproject.toml
├── runs
│   ├── miniseries.sh               # 迷你系列训练脚本
│   ├── runcpu.sh                   # 在 CPU/MPS 上运行的小示例
│   ├── scaling_laws.sh             # Scaling laws 实验
│   └── speedrun.sh                 # 训练约 100 美元成本的 nanochat d20
├── scripts
│   ├── base_eval.py                # Base model：CORE 分数、bits per byte、采样
│   ├── base_train.py               # Base model：训练
│   ├── chat_cli.py                 # Chat model：通过 CLI 对话
│   ├── chat_eval.py                # Chat model：评测任务
│   ├── chat_rl.py                  # Chat model：强化学习
│   ├── chat_sft.py                 # Chat model：训练 SFT
│   ├── chat_web.py                 # Chat model：通过 WebUI 对话
│   ├── tok_eval.py                 # Tokenizer：评估压缩率
│   └── tok_train.py                # Tokenizer：训练
├── tasks
│   ├── arc.py                      # 多选科学题
│   ├── common.py                   # TaskMixture | TaskSequence
│   ├── customjson.py               # 从任意 jsonl 对话构造 Task
│   ├── gsm8k.py                    # 小学数学 8K 题集
│   ├── humaneval.py                # 名称有点误导；简单 Python 编码任务
│   ├── mmlu.py                     # 多主题多选题
│   ├── smoltalk.py                 # 来自 HF 的 SmolTalk 聚合数据集
│   └── spellingbee.py              # 教模型拼写/数字母的任务
├── tests
│   └── test_engine.py
└── uv.lock
```

## 贡献

nanochat 的目标是推动微型模型的最新水平，让大家能以低于 1000 美元的预算端到端地使用和研究它们。可访问性不仅关乎总成本，也关乎认知复杂度：nanochat 不是一个配置项无穷无尽的 LLM “框架”；代码库中没有巨大的配置对象、模型工厂，或 if-then-else 怪物。它是一个单一、内聚、极简、可读、可改、极易 fork 的“强 baseline”代码库，设计为从头跑到尾，并产出一个你可以聊天的 ChatGPT 模型。目前作者个人最感兴趣的部分是缩短到达 GPT-2 能力的延迟（即 CORE 分数超过 0.256525）。当前需要约 3 小时，但通过改进预训练阶段，还可以进一步缩短。

当前 AI 政策：披露。提交 PR 时，请声明哪些部分有实质性 LLM 贡献，且这些部分不是你亲自编写或你没有完全理解。

## 致谢

- 名称 nanochat 源自作者早期项目 [nanoGPT](https://github.com/karpathy/nanoGPT)，后者只覆盖预训练。
- nanochat 也受到 [modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt) 启发；该项目用清晰指标和排行榜把 nanoGPT 游戏化，nanochat 借鉴了很多思想以及部分预训练实现。
- 感谢 [HuggingFace](https://huggingface.co/) 提供 fineweb 和 smoltalk。
- 感谢 [Lambda](https://lambda.ai/service/gpu-cloud) 提供本项目开发所用计算资源。
- 感谢首席 LLM whisperer Alec Radford 提供建议/指导。
- 感谢 repo czar Sofie [@svlandeg](https://github.com/svlandeg) 帮助管理 nanochat 的 issues、pull requests 和 discussions。

## 引用

如果 nanochat 对你的研究有帮助，可按如下方式引用：

```bibtex
@misc{nanochat,
  author = {Andrej Karpathy},
  title = {nanochat: The best ChatGPT that \$100 can buy},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/karpathy/nanochat}
}
```

## 许可证

MIT
