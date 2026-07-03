#!/bin/bash
# Run all 16 LongBench datasets for one (model, sparsity_mode) variant of
# SketchWalk.  Datasets are split across 4 GPUs (4 per GPU).
#
# Usage:
#   bash scripts/longbench/SketchWalk/run_longbench.sh <model_tag> <sparsity_mode> [gpu_base]
#
#   model_tag       : Llama-3.2-1B-Instruct | Llama-3.1-8B-Instruct | Qwen3-8B
#   sparsity_mode   : prefilling | decoding | both
#   gpu_base        : first GPU index (default 0).  Uses gpu_base..gpu_base+3.
#
# Output: $OUT_ROOT/sketchwalk-<model_tag>-<sparsity_mode>/{results,logs}/

set -euo pipefail

MODEL_TAG=${1:?model_tag required: Llama-3.2-1B-Instruct | Llama-3.1-8B-Instruct | Qwen3-8B}
SPARSITY_MODE=${2:?sparsity_mode required: prefilling | decoding | both}
GPU_BASE=${3:-0}

case $SPARSITY_MODE in
    prefilling|decoding|both) ;;
    *) echo "Invalid sparsity_mode: $SPARSITY_MODE (expected prefilling | decoding | both)"; exit 1 ;;
esac

BASE=${SKETCHWALK_BASE:-$(cd "$(dirname "$0")/../../.." && pwd)}
OUT_ROOT=${OUT_ROOT:-$BASE/experiment-results/longbench}

CFG=$BASE/config/pipeline_config/SketchWalk/${MODEL_TAG}/${MODEL_TAG}-inference-${SPARSITY_MODE}.json
if [ ! -f "$CFG" ]; then
    echo "Missing config: $CFG"
    exit 1
fi

OUT_MODEL=$OUT_ROOT/sketchwalk-${MODEL_TAG}-${SPARSITY_MODE}
mkdir -p "$OUT_MODEL/logs" "$OUT_MODEL/results"

cd "$BASE"

run_slice() {
    local GPU=$1
    shift
    local DATASETS="$@"
    for ds in $DATASETS; do
        echo "[$(date '+%H:%M:%S')] GPU$GPU starting $ds ($MODEL_TAG/$SPARSITY_MODE)"
        CUDA_VISIBLE_DEVICES=$GPU \
        PYTHONPATH=$BASE \
        TRITON_CACHE_DIR=/tmp/triton_cache_gpu${GPU} \
        python pipeline/sketchwalk/main.py \
            --exp_desc "longbench_${ds}_${MODEL_TAG}_sketchwalk_${SPARSITY_MODE}" \
            --pipeline_config_dir "$CFG" \
            --eval_config_dir "$BASE/config/eval_config/longbench/${ds}.json" \
            --output_folder_dir "$OUT_MODEL/results/$ds" \
            2>&1 | tee "$OUT_MODEL/logs/${ds}.log"
        echo "[$(date '+%H:%M:%S')] GPU$GPU done $ds"
    done
}

# 4 GPUs × 4 datasets each = 16 total.  Stagger by 5s to avoid HF login race.
run_slice $((GPU_BASE+0)) narrativeqa qasper multifieldqa_en hotpotqa    >> "$OUT_MODEL/logs/_gpu0.log" 2>&1 &
sleep 5
run_slice $((GPU_BASE+1)) 2wikimqa musique gov_report qmsum              >> "$OUT_MODEL/logs/_gpu1.log" 2>&1 &
sleep 5
run_slice $((GPU_BASE+2)) multi_news trec triviaqa samsum                >> "$OUT_MODEL/logs/_gpu2.log" 2>&1 &
sleep 5
run_slice $((GPU_BASE+3)) passage_retrieval_en lcc repobench-p passage_count >> "$OUT_MODEL/logs/_gpu3.log" 2>&1 &

echo "Launched 4 GPU workers ($((GPU_BASE))..$((GPU_BASE+3))).  Waiting..."
wait
echo "[$(date '+%H:%M:%S')] LongBench ${MODEL_TAG} ${SPARSITY_MODE} — all done."
