#!/bin/bash
# Run all 16 LongBench datasets on a single GPU, decode-only sparsity.
# Prefill uses dense FlashAttention; sparse attention kicks in at decode time.
#
# Usage:
#   bash scripts/longbench/SketchWalk/run_decoding.sh <model_tag> [gpu_id]
#
#   model_tag : Llama-3.2-1B-Instruct | Llama-3.1-8B-Instruct | Qwen3-8B
#   gpu_id    : which GPU to use (default 0)
#
# Output: $OUT_ROOT/sketchwalk-<model_tag>-decoding/{results,logs}/

set -euo pipefail

MODEL_TAG=${1:?model_tag required: Llama-3.2-1B-Instruct | Llama-3.1-8B-Instruct | Qwen3-8B}
GPU=${2:-0}
MODE=decoding

BASE=${SKETCHWALK_BASE:-$(cd "$(dirname "$0")/../../.." && pwd)}
OUT_ROOT=${OUT_ROOT:-$BASE/experiment-results/longbench}
CFG=$BASE/config/pipeline_config/SketchWalk/${MODEL_TAG}/${MODEL_TAG}-inference-${MODE}.json

if [ ! -f "$CFG" ]; then
    echo "Missing config: $CFG"
    exit 1
fi

OUT_MODEL=$OUT_ROOT/sketchwalk-${MODEL_TAG}-${MODE}
mkdir -p "$OUT_MODEL/logs" "$OUT_MODEL/results"

DATASETS=(
    narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique
    gov_report qmsum multi_news trec triviaqa samsum
    passage_retrieval_en passage_count lcc repobench-p
)

cd "$BASE"
for ds in "${DATASETS[@]}"; do
    echo "[$(date '+%H:%M:%S')] running $ds"
    CUDA_VISIBLE_DEVICES=$GPU \
    PYTHONPATH=$BASE \
    TRITON_CACHE_DIR=/tmp/triton_cache_gpu${GPU} \
    python pipeline/sketchwalk/main.py \
        --exp_desc "longbench_${ds}_${MODEL_TAG}_sketchwalk_${MODE}" \
        --pipeline_config_dir "$CFG" \
        --eval_config_dir "$BASE/config/eval_config/longbench/${ds}.json" \
        --output_folder_dir "$OUT_MODEL/results/$ds" \
        2>&1 | tee "$OUT_MODEL/logs/${ds}.log"
done

echo "[$(date '+%H:%M:%S')] LongBench ${MODEL_TAG} ${MODE} — all done."
