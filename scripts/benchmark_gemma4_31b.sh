#!/bin/bash
exp_config_path=${1:-"benchmark_configs/gemma4_31b_config.json"}
# Aggregation method: zeroshot (baseline) | majority (Self-Consistency) | confidence (Sample-and-Rank) | selector (Self-Selector)
method=${2:-"majority"}
gpu=${3:-"1"}

case "$method" in
    zeroshot)   agg_method="" ;;
    majority)   agg_method="answer_level_greedy_majority_vote" ;;
    confidence) agg_method="answer_level_greedy_confidence_scores" ;;
    selector)   agg_method="answer_level_greedy_mllm_selector" ;;
    *)
        echo "Unknown method '$method'. Use: zeroshot | majority | confidence | selector" >&2
        exit 1
        ;;
esac

use_openai=False # True, False, "Local" Note: True is too expensive!, For local judge, use local_judge.sh script.
export CUDA_VISIBLE_DEVICES=$gpu
export AUTO_SPLIT=0
export SUBSET_LEN=1000
export USE_COT=1
export TOKENIZERS_PARALLELISM=false
export DIST_TIMEOUT=99999999999
export UNSLOTH_DISABLE_FAST_GENERATION="1"
# export SAVE_VISUAL_SAMPLES=true

exp_config_stem=$(basename "$exp_config_path" .json)
workdir="benchmark_results/n_samples_${SUBSET_LEN}/${exp_config_stem}_${method}/"

# if [ -d "$workdir" ]; then
#     read -p "The directory '$workdir' already exists. Do you want to delete it? (y/n): " confirm
#     if [ "$confirm" = "y" ]; then
#         rm -r "$workdir"
#     fi
# fi

mkdir -p "$workdir"
if [ "$method" = "zeroshot" ]; then
    cp "$exp_config_path" "$workdir"
else
    patched_config="$workdir/$(basename "$exp_config_path")"
    python - "$exp_config_path" "$patched_config" "$agg_method" <<'PY'
import json, sys
src, dst, agg = sys.argv[1], sys.argv[2], sys.argv[3]
with open(src) as f:
    cfg = json.load(f)
for m in cfg.get("model", {}).values():
    m["token_selection_aggregation_method"] = agg
with open(dst, "w") as f:
    json.dump(cfg, f, indent=4)
PY
    exp_config_path="$patched_config"
fi

if [ "$use_openai" = True ]; then
    openai_file_path="/zhome/88/8/215456/openai_key.txt"
    export OPENAI_API_KEY=$(<"$openai_file_path")
elif [ "$use_openai" = "Local" ]; then
    export OPENAI_API_KEY="sk-123456"
    export OPENAI_API_BASE="http://0.0.0.0:23333/v1/chat/completions"
    export LOCAL_LLM="internlm/internlm2_5-7b-chat"

    for _ in {1..3}; do echo "================================================================"; done
    echo "Local judge is enabled. Make sure you have the local judge running."
    for _ in {1..3}; do echo "================================================================"; done
fi

export PYTHONPATH=$PWD:$PYTHONPATH
number_of_gpus=$(($(grep -o "," <<<"$CUDA_VISIBLE_DEVICES" | wc -l) + 1))
echo "Number of GPUs: $number_of_gpus"
if [ "$number_of_gpus" -gt 1 ]; then
    if [ "$AUTO_SPLIT" = 1 ]; then
        number_of_gpus=1
    fi
    torchrun --nproc-per-node=$number_of_gpus --master-port 29555 run.py --work-dir "$workdir" --verbose --reuse --config "$exp_config_path"
else
    python run.py --work-dir "$workdir" --verbose --reuse --config "$exp_config_path"
fi
