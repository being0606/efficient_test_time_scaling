#!/bin/bash
# Run Gemma4 zero-shot baselines for both E2B and 31B sequentially on a single GPU.
# Usage: bash scripts/run_gemma4_zeroshot.sh [gpu]
#   gpu: GPU index (default: 0)
set -e

gpu=${1:-0}

echo "============================================================"
echo "Gemma4 E2B zero-shot on GPU $gpu"
echo "============================================================"
bash scripts/benchmark_gemma4_e2b.sh \
    benchmark_configs/gemma4_e2b_zeroshot_config.json \
    zeroshot \
    "0"

echo "============================================================"
echo "Gemma4 31B zero-shot on GPU $gpu"
echo "============================================================"
bash scripts/benchmark_gemma4_31b.sh \
    benchmark_configs/gemma4_31b_zeroshot_config.json \
    zeroshot \
    "0"

echo "============================================================"
echo "Done. Results:"
echo "  benchmark_results/n_samples_1000/gemma4_e2b_zeroshot_config_zeroshot/"
echo "  benchmark_results/n_samples_1000/gemma4_31b_zeroshot_config_zeroshot/"
echo "============================================================"
