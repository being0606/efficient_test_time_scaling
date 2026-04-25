<p align="center">
<h1 align="center"><strong>Efficient Test-Time Scaling for Small Vision-Language Models</strong></h1>
  <p align="center">
    <em><a href="https://monurcan.github.io/">Mehmet Onurcan Kaya</a><sup>1,2</sup>, <a href="https://elliottd.github.io/">Desmond Elliott</a><sup>3,2</sup>, <a href="https://dimipapa.github.io/">Dim P. Papadopoulos</a><sup>1,2</sup></em>
  </p>
  <p align="center">
    <em><sup>1</sup> Technical University of Denmark &nbsp;&nbsp;&nbsp; <sup>2</sup> Pioneer Center for AI &nbsp;&nbsp;&nbsp; <sup>3</sup> University of Copenhagen</em>
  </p>
</p>

<div id="top" align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2510.03574-b31b1b)](http://arxiv.org/abs/2510.03574)
[![](https://img.shields.io/badge/%F0%9F%9A%80%20-Project%20Page-blue)](https://monurcan.github.io/efficient_test_time_scaling/)

</div>

<div align="center">
    <img src="assets/teaser.png">
</div>

Our framework consists of two main pipelines: (1) Test-Time Augmentation: Given an input image and text prompt, we apply various transformations to create multiple augmented versions. VLM processes each augmented input to produce next token probability distributions, which are then aggregated at the token level to generate the final response. (2) Test-Time Adaptation: We create pseudolabels through test-time augmentation and fine-tune the VLM parameters, then repeat the process.  Both methods demonstrate effectiveness across nine diverse benchmarks as shown in (b).

## 🔎 Abstract
Small Vision-Language Models (VLMs) provide a computationally efficient alternative to larger models, at the cost of weaker generalization abilities and downstream task performance. These shortcomings could be addressed by test-time scaling techniques, but existing methods are typically computationally demanding, contradicting the resource-efficient design goals of small models. To address these limitations, we propose two novel and efficient test-time scaling strategies that leverage the model-internal features rather than external supervision: (i) Test-Time Augmentation (TTAug), which generates multiple augmented inputs and aggregates outputs at the token level without parameter updates, and (ii) Test-Time Adaptation (TTAdapt), which adapts model parameters during inference using consensus-based pseudolabels from TTAug. Through extensive experiments across nine benchmarks, we demonstrate consistent performance improvements while maintaining computational efficiency suitable for resource-constrained environments. The generality of our approach is demonstrated both within models at different scales and across different VLMs without additional tuning.


## 🔧 Installation
```bash
git clone https://github.com/monurcan/efficient_test_time_scaling.git
cd efficient_test_time_scaling
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt --no-deps
pip install -e . --no-deps
```

Note that the code has been tested with Python 3.10.12 and CUDA 12.5.

## 💻 Inference: Run an Experiment
```bash
bash scripts/benchmark.sh benchmark_configs/test_config.json
```
This will execute the experiment configuration defined in [`benchmark_configs/test_config.json`](./benchmark_configs/test_config.json).

For customizing experiments, refer to the configuration system documentation: [`docs/en/ConfigSystem.md`](./docs/en/ConfigSystem.md)

Results will be automatically saved to the [`benchmark_results`](./benchmark_results) directory as specified in [`scripts/benchmark.sh`](./scripts/benchmark.sh).

### Gemma4: Compare Zero-shot / Self-Consistency / Sample-and-Rank / Self-Selector

The Gemma4 adapter ([`vlmeval/vlm/tta/tta_gemma4.py`](./vlmeval/vlm/tta/tta_gemma4.py)) supports three answer-level aggregation strategies on top of the plain zero-shot baseline. Each can be selected via a CLI argument to the benchmark scripts.

**Script usage** — `bash <script> <config> <method> <gpu>`:

| Arg | Values |
|---|---|
| `<config>` | Path to a Gemma4 config (`benchmark_configs/gemma4_e2b_config.json`, `benchmark_configs/gemma4_31b_config.json`, `benchmark_configs/gemma4_e2b_zeroshot_config.json`, ...) |
| `<method>` | `zeroshot` (baseline) \| `majority` (Self-Consistency) \| `confidence` (Sample-and-Rank) \| `selector` (Self-Selector) |
| `<gpu>` | GPU index, e.g. `0`, `1`, `4` |

> `zeroshot` runs plain inference on the base `Gemma4` class with **no augmentation and no aggregation** — use it as the baseline. Pair it with `benchmark_configs/gemma4_e2b_zeroshot_config.json` / `gemma4_31b_zeroshot_config.json`, which omit the TTA fields and point to `"class": "Gemma4"` directly.

Example:
```bash
bash scripts/benchmark_gemma4_e2b.sh benchmark_configs/gemma4_e2b_config.json selector 0
```

Results are saved under `benchmark_results/n_samples_${SUBSET_LEN}/<config_stem>_<method>/`, so multiple methods do not overwrite each other.

**Zero-shot baselines** — run these first to get the no-TTA reference numbers:
```bash
# E2B baseline on GPU 0
bash scripts/benchmark_gemma4_e2b.sh     benchmark_configs/gemma4_e2b_zeroshot_config.json     zeroshot 0
# 31B baseline on GPU 1
bash scripts/benchmark_gemma4_31b.sh benchmark_configs/gemma4_31b_zeroshot_config.json zeroshot 1
```
Outputs go to `benchmark_results/n_samples_1000/gemma4_e2b_zeroshot_config_zeroshot/` and `..._31b_zeroshot_config_zeroshot/`.

**Quick timing run with `SUBSET_RATIO`** — evaluate on a fraction of each dataset (5% here) to measure inference time without full-scale runs. The override env var takes precedence over `SUBSET_LEN` and writes results to a separate `benchmark_results/ratio_005/...` tree, so it does not collide with full-cap runs. Per-dataset timing is dumped to `<workdir>/<model>/T*/<model>_<dataset>_timing.json`.
```bash
# E2B zeroshot, 5% subset
SUBSET_RATIO=0.05 bash scripts/benchmark_gemma4_e2b.sh \
    benchmark_configs/gemma4_e2b_zeroshot_config.json zeroshot 0

# 31B zeroshot, 5% subset (run separately)
SUBSET_RATIO=0.05 bash scripts/benchmark_gemma4_31b.sh \
    benchmark_configs/gemma4_31b_zeroshot_config.json zeroshot 0
```

**Full 6-run TTA sweep on 3 GPUs (E2B + 31B × 3 methods)** — run each block in its own terminal:

<details>
<summary>Terminal 1 — GPU 0 (majority / Self-Consistency)</summary>

```bash
bash scripts/benchmark_gemma4_31b.sh benchmark_configs/gemma4_31b_config.json majority 0 && \
bash scripts/benchmark_gemma4_e2b.sh     benchmark_configs/gemma4_e2b_config.json     majority 0
```
</details>

<details>
<summary>Terminal 2 — GPU 1 (confidence / Sample-and-Rank)</summary>

```bash
bash scripts/benchmark_gemma4_31b.sh benchmark_configs/gemma4_31b_config.json confidence 1 && \
bash scripts/benchmark_gemma4_e2b.sh     benchmark_configs/gemma4_e2b_config.json     confidence 1
```
</details>

<details>
<summary>Terminal 3 — GPU 4 (selector / Self-Selector)</summary>

```bash
bash scripts/benchmark_gemma4_31b.sh benchmark_configs/gemma4_31b_config.json selector 4 && \
bash scripts/benchmark_gemma4_e2b.sh     benchmark_configs/gemma4_e2b_config.json     selector 4
```
</details>

Each terminal runs the large **31B** model first, then the smaller **E2B** on the same GPU once VRAM is freed. The three terminals run in parallel across GPUs 0 / 1 / 4.

**Single-terminal alternative** (all in background, logs per job):
```bash
(bash scripts/benchmark_gemma4_31b.sh benchmark_configs/gemma4_31b_config.json majority 0 && \
 bash scripts/benchmark_gemma4_e2b.sh     benchmark_configs/gemma4_e2b_config.json     majority 0) > logs_gpu0_majority.txt 2>&1 &

(bash scripts/benchmark_gemma4_31b.sh benchmark_configs/gemma4_31b_config.json confidence 1 && \
 bash scripts/benchmark_gemma4_e2b.sh     benchmark_configs/gemma4_e2b_config.json     confidence 1) > logs_gpu1_confidence.txt 2>&1 &

(bash scripts/benchmark_gemma4_31b.sh benchmark_configs/gemma4_31b_config.json selector 4 && \
 bash scripts/benchmark_gemma4_e2b.sh     benchmark_configs/gemma4_e2b_config.json     selector 4) > logs_gpu4_selector.txt 2>&1 &

wait && echo "All 6 runs finished"
```

After completion you will have six TTA result directories (plus two zero-shot baselines if you ran them):
```
benchmark_results/n_samples_1000/
├── gemma4_e2b_zeroshot_config_zeroshot/  ├── gemma4_31b_zeroshot_config_zeroshot/
├── gemma4_e2b_config_majority/           ├── gemma4_31b_config_majority/
├── gemma4_e2b_config_confidence/         ├── gemma4_31b_config_confidence/
└── gemma4_e2b_config_selector/           └── gemma4_31b_config_selector/
```

## 🚀 Development
The core logic of our methods is located in [`vlmeval/vlm/tta`](./vlmeval/vlm/tta)

Utility scripts for analysis and visualization are available in [`scripts`](./scripts):
- [`figure_create.ipynb`](./scripts/figure_create.ipynb) - Figure generation, saves them to [`benchmark_visualizations`](./benchmark_visualizations) directory
- [`table_create.ipynb`](./scripts/table_create.ipynb) - Results table generation

## 🙏 Acknowledgement
This project builds upon [VLMEvalKit](https://github.com/open-compass/VLMEvalKit).
For more details, refer to [`README_VLMEVALKIT.md`](./README_VLMEVALKIT.md).

## 📚 Citation
```
@article{Kaya2025EfficientTTS,
  title={Efficient Test-Time Scaling for Small Vision-Language Models},
  author={Mehmet Onurcan Kaya and Desmond Elliott and Dim P. Papadopoulos},
  journal={arXiv preprint arXiv:2510.03574},
  year={2025},
  url={https://monurcan.github.io/efficient_test_time_scaling}
}
```

## 💬 Contact
For questions, please open an issue or contact me at monka@dtu.dk