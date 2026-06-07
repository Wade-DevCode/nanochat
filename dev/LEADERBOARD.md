# 排行榜

本文档说明如何参与 nanochat 的 “Time-to-GPT-2” 排行榜。

我们最关心的主要指标是 “time to GPT-2”：在一个 8XH100 GPU 节点上，超过 GPT-2 (1.6B) CORE 指标所需的墙钟时间。最初在 2019 年，OpenAI 使用 32 个 TPU v3 芯片训练 GPT-2，耗时 168 小时（7 天）；按当时 8 美元/小时/TPUv3 计算，总成本约 43K 美元。GPT-2 达到 0.256525 CORE 分数；CORE 是 DCLM 论文提出的集合指标，汇总了 ARC/MMLU 等 22 个评测。

## 如何参与

[runs/speedrun.sh](../runs/speedrun.sh) 脚本始终实现当前排行榜上的最佳参考流程。

实践中，作者会对 `base_train` 命令做一些小调优。例如，在所有环境配置完成并训练好 tokenizer 后，作者喜欢这样运行：

```bash
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=26 \
    --run="d26-feb2-fp8-ratio8.25" \
    --model-tag="d26_feb2_fp8_ratio8.25" \
    --device-batch-size=16 \
    --sample-every=-1 \
    --save-every=-1 \
    --core-metric-max-per-task=-1 \
    --core-metric-every=999999 \
    --target-param-data-ratio=8.25 \
    --fp8
```

说明：

- `depth` 控制 Transformer 的规模。
- `run` 是 wandb 运行名。
- `model-tag` 是 checkpoint 在磁盘上的位置。
- `device-batch-size`：理想情况下希望设为 32，因为在序列长度 2048（默认）和 8 张 GPU 下，`32 X 2048 X 8 = 524,288`，这是在该规模附近验证过较好的目标总 batch size。不过对更大的模型（例如 d26），32 太大，会 OOM，因此降低 2 倍到 16。`base_train.py` 脚本会自动补偿，计算出需要使用 2 倍梯度累积来达到目标总 batch size。因此它会 forward+backward 两次，然后执行一次 optimizer step。简而言之，理想值是 32；如果放不下，就降低到 16、8 等，并保持 2 的幂，这样梯度累积的数学会更整齐。
- `sample-every = -1` 关闭周期性采样。
- `core-metric-max-per-task=-1` 表示运行完整 CORE eval。
- `core-metric-every=999999` 是一种稍微取巧的方式，让 CORE eval 只在整个 run 的最后发生一次。
- `target-param-data-ratio=8.25` 控制训练长度。脚本通过取非 embedding 模型参数数量并乘以这个数字来决定训练 token 数。当前最优 Tokens:Params 比例可以在 `base_train.py` 的默认值里看到（是 10.5）。根据当前测得的 scaling laws，10.5 会产生“计算最优”模型。不过 GPT-2 能力目前位于 d24 和 d26 之间。因此为了精确达到它，我们要么过训练 d24，要么欠训练 d26。在这个例子中，作者选择略微欠训练 d26。注意，不太推荐使用奇数 depth（如 d25），因为 Transformer 尺寸和 head 维度的数学不会那么整齐。
- `--fp8` 开启 fp8 训练。如果你的 GPU 不支持 fp8，可以去掉这个参数，代码会改用 bf16 训练。bf16 精度高于 fp8，因此你实际上可能可以用更少 step（降低 `target-param-data-ratio`）达到同等能力。

启动 run 后，等待约 1.5 小时，最后会看到类似输出：

```text
wandb: Run summary:
wandb:          core_metric 0.25851
wandb:                 step 16704
wandb: total_training_flops 4.330784131228946e+19
wandb:  total_training_time 10949.46713
```

你的 CORE 指标必须大于 GPT-2 的 0.256525。然后报告 `total_training_time`（例如 10949），它表示单纯训练迭代的时间，单位为秒，不包含评测和日志时间。因此这里大约是 `10949/60/60 ~= 3.04` 小时。也请同时记录并报告该 run 的 validation bpb，因为 CORE 指标会有一些噪声。

如果你超过 GPT-2，并且耗时低于排行榜当前 SOTA，就可以提交 PR。除了原始收益外，是否合并还会考虑一些定性和审美因素。例如，如果改动很难看、显著膨胀代码，或看起来过于冷门，就会把这些因素和改进幅度一起权衡。此外，nanochat 不只关心单个模型目标，也关心完整的模型迷你系列。因此你的改动必须足够有原则，能轻松推广到其他模型 depth，以便扫出一个迷你系列。

创建 commit 后，可用以下命令获取当前短 git commit hash：

```bash
git log -1 --format="%h"
```

## Run 1

于 2026 年 1 月 29 日在 commit `348fbb3` 上达成。启动命令为：

```bash
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=24 \
    --run=d24-jan29 \
    --model-tag=d24_jan29 \
    --device-batch-size=16 \
    --sample-every=-1 \
    --save-every=-1 \
    --core-metric-max-per-task=-1 \
    --core-metric-every=3000 \
    --target-param-data-ratio=12
```

结果为：

```text
wandb: Run summary:
wandb:          core_metric 0.25851
wandb:                 step 16704
wandb: total_training_flops 4.330784131228946e+19
wandb:  total_training_time 10949.46713
```

validation bpb 为 0.74833。

详细记录：[Beating GPT-2 for <<$100: the nanochat journey](https://github.com/karpathy/nanochat/discussions/481)

## Run 2

于 2026 年 2 月 2 日在 commit `a67eba3` 上达成。启动命令为：

```bash
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=26 \
    --run="d26-feb2-fp8-ratio8.5" \
    --model-tag="d26_feb2_fp8_ratio8.5" \
    --device-batch-size=16 \
    --sample-every=-1 \
    --save-every=-1 \
    --core-metric-max-per-task=-1 \
    --core-metric-every=999999 \
    --target-param-data-ratio=8.5 \
    --fp8
```

结果为：

```text
core_metric 0.2578
step 14889
total_training_time 10493
Minimum validation bpb: 0.745036
```

这次 run 的主要变化是 `--fp8`：它会把所有 Linear 层（gates 之外）切换为使用 `torchao` 和 tensorwise fp8 scaling 的 fp8 训练。每一步的质量略低，但速度快很多，总体仍然获益。没有 fp8 的用户（例如 GPU 不支持）可以直接去掉 `--fp8` 参数，改用 bfloat16 训练。这样也能正常工作，但由于 fp8 -> bf16 的精度升级，会得到一个比 GPT-2 略强的模型。后续也许还可以进一步调优哪些层应纳入 fp8 转换，例如一些较小的 matmul 可能应该保留 bf16 等。

前一个记录是 3.04 小时，所以 2.91 小时对应 `(3.04 - 2.91)/3.04*100 ~= 4.3%` 的速度提升。

## Run 3

于 2026 年 2 月 5 日在 commit `2c062aa` 上达成。启动命令：

```bash
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=26 \
    --run="d26_feb4_double_batch_ratio8.25" \
    --model-tag="d26_feb4_double_batch_ratio8.25" \
    --device-batch-size=16 \
    --total-batch-size=1048576 \
    --sample-every=-1 \
    --save-every=-1 \
    --core-metric-max-per-task=-1 \
    --core-metric-every=999999 \
    --target-param-data-ratio=8.25 \
    --fp8
```

结果：

```text
core_metric 0.26024
step 7226
total_training_time 9922
Minimum validation bpb: 0.74645
```

这里的主要变化是 batch size 从 0.5M 翻倍到 1M。它对 d26 模型效果更好，并允许作者把优化步数通过 `--target-param-data-ratio` 从 8.5 略降到 8.25。简而言之，原始 0.5M batch size 是为 d12 调优的，但更大的模型（如 d26）更偏好更大的总 batch size。作者在实验中确认 d26 偏好 1M，然后实现并合并了一个根据 depth 有原则地计算最优 batch size 的方法，使所有 depth 的 nanochat 模型受益。更多细节见 [dev/LOG.md](LOG.md) 中的 “2026-02-05: Auto Batch Size Scaling” 条目。

## Run 4

于 2026 年 3 月 3 日在 commit `324e69c` 上达成。主要变化是从 HuggingFace FineWeb-EDU 切换到 NVIDIA ClimbMix 数据集。`@karpathy` 之前多次尝试替换数据集，每次结果都变差（FineWeb、DCLM、Olmo），但 ClimbMix 带来了清晰且立刻可见的提升。感谢 `@ddudek` 最初为 nanochat 发现 ClimbMix 并报告改进，从而触发后续调查。

复现方式：使用上述 commit，至少下载 150 个数据 shard，训练 tokenizer：

```bash
python -m nanochat.dataset -n 150
python -m scripts.tok_train
```

然后按常规方式启动 run，使用略低于计算最优的 ratio 9.5（而不是计算最优 10.5），也就是略微欠训练 d24。

```bash
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=24 \
    --run="d24-climbmix" \
    --model-tag="d24-climbmix" \
    --sample-every=-1 \
    --save-every=-1 \
    --core-metric-max-per-task=-1 \
    --core-metric-every=999999 \
    --target-param-data-ratio=9.5 \
    --device-batch-size=16 \
    --fp8
```

作者独立运行了 7 次。由于训练存在轻微非确定性，CORE 分数会有一定分布，例如：

```text
0.25373
0.2584
0.25489
0.2568
0.25732
0.26765
0.25119
```

均值为 0.25714（高于所需的 GPT-2 阈值），最大最小差为 0.01646。未来值得研究的一点是：随机打乱数据 shard（即使用不同顺序）还能得到略好结果。这有些意外，因为文档在数据构建时已完全打乱，因此理论上应有较均匀的数据分布。事实上，当前默认顺序很不幸地属于不同 shuffle seed 中较差的顺序之一，但目前已经足以超过 GPT-2，因此作者选择合并。TODO：之后再投入更多研究。

注意：由于数据分布变化，本次 run 的 `val_bpb` 和前三次 run **不可比较**。本 run 的 validation bpb 恰好是 `0.71854`。如果数据集不变，`val_bpb` 是追踪相对性能的优秀、平滑指标，噪声也小于 CORE。

## Run 5

于 2026 年 3 月 9 日在 commit `6ed7d1d` 上达成。启动命令与 Run 4 完全相同，只是 `--target-param-data-ratio=8.7`。作者运行了 5 次相同实验，平均 CORE 为 0.2690，显著高于所需阈值 0.2565。没有进一步降低 ratio（即缩短训练）的原因是：虽然 CORE 的“安全余量”很大，但 val_loss 的安全余量较小，为 0.71808；我们希望低于 Run 4 的 val loss 0.71854。可能还能把 ratio 降到更低，比如 8.6，但此时不值得为了极小差异继续纠结。

这个 commit 很特别，因为其中所有改进都来自一个私有版本 [autoresearch](https://github.com/karpathy/autoresearch) 在 d12 模型上完成的完全自主“研究”。作者在 [这条推文](https://x.com/karpathy/status/2031135152349524125) 中写了更多。改动很容易从 d12 迁移到 d24，因此刷新排行榜记录，把 “time to GPT-2” 从 2.02 小时降到 1.80 小时。

## Run 6

于 2026 年 3 月 14 日在 commit `a825e63` 上达成。启动命令与 Run 4 完全相同，只是 `--target-param-data-ratio=8`。架构改进让我们能够越来越短地训练。作者尝试用过训练 d22 代替欠训练 d24，但结果更差。这组改动来自 autoresearch round 2，当时作者要求它参考 modded-nanogpt 仓库寻找灵感。因此探索尝试了多个想法，特别是找到了一种融合 backout 和 smear 的方式，使它们变得有用（作者很久前手动尝试过这些想法，当时导致了回归）。其中 smear 尤其稍微重一些、也更膨胀，因为它本质上是跨 token 上下文的“早期融合”，产生一种 bigram 输入，让网络更早关注更高阶 ngram。因此代码也更复杂，并且需要改推理。作者用单元测试验证了 Engine 推理相对 `GPT.generate()` 的 naive 推理是正确的。5 次 run 的平均 CORE 为 0.262634，每次耗时 1.65 小时（99 分钟）。
