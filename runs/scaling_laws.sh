#!/bin/bash

LABEL="jan26"

FLOPS_BUDGETS=(
    1e18
    2.15e18
    4.64e18
    1e19
)
DEPTHS=(10 12 14 16 18 20)

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
WANDB_RUN="${WANDB_RUN:-scaling_${LABEL}}"
EVAL_TOKENS=$((100 * 524288))  # 最终评估约 100M tokens（默认约 10M）

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
source .venv/bin/activate

RESULTS_DIR="$NANOCHAT_BASE_DIR/scaling_laws_results_${LABEL}"
mkdir -p "$RESULTS_DIR"
RESULTS_FILE="$RESULTS_DIR/results.csv"

# 仅在文件不存在时写入 CSV 表头
if [ ! -f "$RESULTS_FILE" ]; then
    echo "flops_budget,depth,model_dim,params_wte,params_value_embeds,params_lm_head,params_transformer,params_scalars,params_total,num_iterations,tokens_trained,val_bpb,core_score,train_time_sec" > "$RESULTS_FILE"
fi

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# 检查结果中是否已有某个 run
run_exists() {
    local flops=$1
    local depth=$2
    grep -q "^${flops},${depth}," "$RESULTS_FILE" 2>/dev/null
}

# =============================================================================
# 主循环
# =============================================================================

for flops in "${FLOPS_BUDGETS[@]}"; do
    log "=============================================="
    log "计算预算：$flops FLOPs"
    log "=============================================="

    for d in "${DEPTHS[@]}"; do

        # 如果已经完成则跳过
        if run_exists "$flops" "$d"; then
            log "跳过 d=$d at $flops FLOPs（结果中已存在）"
            continue
        fi

        log "训练 d=$d at $flops FLOPs..."

        # 此 run 的唯一 tag
        TAG="scaling_${flops}_d${d}"

        # 在更大 depth 下减小 --device-batch-size 以避免 OOM
        if [ $d -ge 28 ]; then
            DEVICE_BATCH_SIZE_ARG="--device-batch-size=8"
        elif [ $d -ge 20 ]; then
            DEVICE_BATCH_SIZE_ARG="--device-batch-size=16"
        else
            DEVICE_BATCH_SIZE_ARG="--device-batch-size=32"
        fi

        # 记录开始时间
        START_TIME=$(date +%s)

        # 使用固定 FLOPs 预算训练模型
        # 脚本会自动计算 num_iterations 以达到 target_flops
        # CORE eval 只在最后发生一次（999999 确保只有最终 step）
        torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train -- \
            --depth=$d \
            --target-flops=$flops \
            --target-param-data-ratio=-1 \
            --run="${WANDB_RUN}_${TAG}" \
            --model-tag="${TAG}" \
            --eval-tokens=$EVAL_TOKENS \
            --core-metric-every=999999 \
            --core-metric-max-per-task=-1 \
            --sample-every=-1 \
            --save-every=-1 \
            $DEVICE_BATCH_SIZE_ARG \
            2>&1 | tee "$RESULTS_DIR/${TAG}_train.log"

        END_TIME=$(date +%s)
        TRAIN_TIME=$((END_TIME - START_TIME))

        # 从日志中提取训练统计信息
        LOG_FILE="$RESULTS_DIR/${TAG}_train.log"

        # 提取详细参数计数（用于采用不同约定的 scaling law 分析）
        # 注意：日志格式有 padding，例如 "wte                     : 25,165,824"
        # 因此 grep "^key "（行首 key 后跟空格）以避免误匹配
        PARAMS_WTE=$(grep "^wte " "$LOG_FILE" | tail -1 | grep -oP '[\d,]+' | tr -d ',')
        PARAMS_VE=$(grep "^value_embeds " "$LOG_FILE" | tail -1 | grep -oP '[\d,]+' | tr -d ',')
        PARAMS_LM=$(grep "^lm_head " "$LOG_FILE" | tail -1 | grep -oP '[\d,]+' | tr -d ',')
        PARAMS_TRANSFORMER=$(grep "^transformer_matrices " "$LOG_FILE" | tail -1 | grep -oP '[\d,]+' | tr -d ',')
        PARAMS_SCALARS=$(grep "^scalars " "$LOG_FILE" | tail -1 | grep -oP '[\d,]+' | tr -d ',')
        PARAMS_TOTAL=$(grep "^total " "$LOG_FILE" | tail -1 | grep -oP '[\d,]+' | tr -d ',')

        NUM_ITERS=$(grep "Calculated number of iterations" "$LOG_FILE" | tail -1 | sed 's/.*: //' | tr -d ',')
        # 从日志中提取实际 batch size（自动计算，随模型大小变化）
        BATCH_SIZE=$(grep "Total batch size" "$LOG_FILE" | tail -1 | grep -oP 'Total batch size \K[\d,]+' | tr -d ',')
        TOKENS_TRAINED=$((NUM_ITERS * BATCH_SIZE))
        # 模型维度
        MODEL_DIM=$((d * 64))
        # 来自最终评估的 Val BPB
        VAL_BPB=$(grep "Validation bpb:" "$LOG_FILE" | tail -1 | grep -oP '[\d.]+$')

        # 从训练日志中提取 CORE 分数（在最终 step 评估）
        CORE_SCORE=$(grep "CORE metric:" "$LOG_FILE" | tail -1 | awk '{print $NF}')
        if [ -z "$CORE_SCORE" ]; then
            log "警告：无法提取 d=$d 的 CORE 分数"
            CORE_SCORE="0.0"
        fi

        log "  Params: $PARAMS_TOTAL (transformer: $PARAMS_TRANSFORMER), Iters: $NUM_ITERS, Val BPB: $VAL_BPB, CORE: $CORE_SCORE"

        # 追加到 CSV
        echo "$flops,$d,$MODEL_DIM,$PARAMS_WTE,$PARAMS_VE,$PARAMS_LM,$PARAMS_TRANSFORMER,$PARAMS_SCALARS,$PARAMS_TOTAL,$NUM_ITERS,$TOKENS_TRAINED,$VAL_BPB,$CORE_SCORE,$TRAIN_TIME" >> "$RESULTS_FILE"
    done
done

log "=============================================="
log "Scaling Laws sweep 完成"
log "=============================================="
log "结果已保存到：$RESULTS_FILE"
echo ""
echo "结果："
column -t -s',' "$RESULTS_FILE"
