#!/bin/bash
# Run all three pure-TTS methods (majority/confidence/selector) for one Gemma4 model
# on a single GPU, sequentially.
#
# Usage:
#   bash scripts/run_gemma4_tts.sh <e2b|31b> [gpu]
#
# Examples:
#   bash scripts/run_gemma4_tts.sh e2b 0
#   bash scripts/run_gemma4_tts.sh 31b 1
#
# Optional env: SUBSET_RATIO (e.g. 0.05) to run on a fraction of each dataset
# instead of the default SUBSET_LEN=1000 cap.
set -e

model=${1:?"usage: bash scripts/run_gemma4_tts.sh <e2b|31b> [gpu]"}
gpu=${2:-0}

case "$model" in
    e2b)
        script="scripts/benchmark_gemma4_e2b.sh"
        config="benchmark_configs/gemma4_e2b_config.json"
        ;;
    31b)
        script="scripts/benchmark_gemma4_31b.sh"
        config="benchmark_configs/gemma4_31b_config.json"
        ;;
    *)
        echo "Unknown model '$model'. Use: e2b | 31b" >&2
        exit 1
        ;;
esac

methods=(majority confidence selector)

for method in "${methods[@]}"; do
    echo "============================================================"
    echo "Gemma4 $model | method=$method | GPU=$gpu"
    echo "============================================================"
    bash "$script" "$config" "$method" "$gpu"
done

echo "============================================================"
echo "Done. Results under benchmark_results/.../gemma4_${model}_config_{majority,confidence,selector}/"
echo "============================================================"
