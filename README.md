# Scout Before You Attend: Sketch-and-Walk Sparse Attention for Efficient LLM Inference

**Hoang Anh Duy Le**¹²   **Sahil Joshi**¹   **Zeyu Yang**¹   **Zhaozhuo Xu**²   **Anshumali Shrivastava**¹

¹ Department of Computer Science, Rice University   ² Workato

*Published at the 43rd International Conference on Machine Learning (**ICML 2026**), Seoul, South Korea.*

[📄 Paper (OpenReview)](https://openreview.net/forum?id=uCLVPafHqd)

---

## TL;DR

Which tokens matter at layer *L*? Existing sparse attention methods answer from layer *L* alone. **Sketch&Walk** simulates attention's forward flow — a Hadamard sketch builds a block transition matrix at each layer, and a random walk through these matrices forecasts where importance will emerge in later layers, surfacing tokens that look unimportant when viewed one layer at a time. Matches dense LongBench at 20% attention density and delivers up to **4.7× end-to-end attention speedup** over FlashAttention-2.

## Overview

![Sketch&Walk pipeline](figures/sketchwalk_pipeline.png)

*(1) Queries and keys are sketched with Small-World Sketching to obtain lightweight block-level attention estimates. (2) These estimates are accumulated across layers with Sketch-Determined Walk to approximate cross-layer attention influence. (3) The resulting walk scores are used to select top-τ blocks for sparse attention.*

## Abstract

Self-attention dominates the computational and memory cost of long-context LLM inference across both prefill and decode phases. To address this challenge, we introduce **Sketch&Walk Attention**, a training-free sparse attention method that determines sparsity with lightweight sketches and deterministic walk. Sketch&Walk applies Hadamard sketching to get inexpensive approximations of attention scores, then aggregates these estimates across layers via a walk mechanism that captures attention influence beyond direct interactions between tokens. The accumulated walk scores are used to select top-*k* attention blocks, enabling dynamic sparsity with a single training-free algorithm that applies uniformly to both the prefill and decode phases, together with custom sparse attention kernels. Across a wide range of models and tasks, Sketch&Walk maintains near-lossless accuracy at 20% attention density and can slightly outperform dense attention in some settings, while achieving up to **4.7×** end-to-end attention speedup over FlashAttention-2.

## Installation

**Requirements:** Python 3.12, CUDA 12.8, a Hopper or Ampere GPU (H100/H200/A100 recommended for long-context runs).

```bash
git clone https://github.com/Escanord/SketchWalk.git
cd SketchWalk

# Create a fresh conda environment
conda create -n sketchwalk python=3.12 -y
conda activate sketchwalk

# Install a matching PyTorch build for your CUDA version, e.g. CUDA 12.8:
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# Install everything else pinned in requirements.txt
pip install -r requirements.txt
```

### HuggingFace access token

The evaluation pipeline downloads gated models (e.g. `meta-llama/Llama-3.1-8B-Instruct`) from the HuggingFace Hub, which requires an access token. Before running anything:

1. Create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (a "Read" token is sufficient).
2. Open [`config/access_tokens.py`](config/access_tokens.py) and set:

   ```python
   hf_access_token = 'hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
   ```

   Or export `HF_TOKEN` as an environment variable — the code prefers the env var if both are set.
3. Make sure you have accepted the license for each gated model on its HuggingFace model page.

## Quick Start

Run Sketch&Walk on a single LongBench task with one GPU:

```bash
python pipeline/sketchwalk/main.py \
    --exp_desc "sketchwalk_multifieldqa_en_llama8b_both" \
    --pipeline_config_dir config/pipeline_config/SketchWalk/Llama-3.1-8B-Instruct/Llama-3.1-8B-Instruct-inference-both.json \
    --eval_config_dir     config/eval_config/longbench/multifieldqa_en.json \
    --output_folder_dir   experiment-results/quickstart/
```

The command runs Llama-3.1-8B-Instruct with sparse attention in both prefill and decode phases on the `multifieldqa_en` split of LongBench, writing predictions and metrics under `experiment-results/quickstart/`.

**Pick a sparsity mode by choosing a different pipeline config:**

| mode | when sparse attention is active | pipeline config |
|---|---|---|
| `prefilling` | prefill only (dense decode) | `…-inference-prefilling.json` |
| `decoding`   | decode only (dense prefill) | `…-inference-decoding.json` |
| `both`       | end-to-end sparse           | `…-inference-both.json` |

**Supported models** (`meta-llama/Llama-3.1-8B-Instruct`, `meta-llama/Llama-3.2-1B-Instruct`, and `Qwen/Qwen3-*`) have their configs under [`config/pipeline_config/SketchWalk/`](config/pipeline_config/SketchWalk/).


For each `(model, sparsity_mode)` combination, we ship a single-GPU launcher that runs all 16 LongBench tasks sequentially:

```bash
# Sketch&Walk with sparse prefill (dense decode)
bash scripts/longbench/SketchWalk/run_prefilling.sh Llama-3.1-8B-Instruct  0

# Sketch&Walk with sparse decode (dense prefill)
bash scripts/longbench/SketchWalk/run_decoding.sh   Llama-3.1-8B-Instruct  0

# Sketch&Walk end-to-end sparse (both prefill and decode)
bash scripts/longbench/SketchWalk/run_both.sh       Llama-3.1-8B-Instruct  0
```

Arguments: `<model_tag>` and an optional `<gpu_id>` (default `0`). Substitute `Llama-3.2-1B-Instruct` or `Qwen3-8B` to reproduce those rows.

If you have multiple GPUs and want a 4× speedup across the 16 tasks, the [`run_longbench.sh`](scripts/longbench/SketchWalk/run_longbench.sh) launcher shards them across 4 GPUs:

```bash
bash scripts/longbench/SketchWalk/run_longbench.sh Llama-3.1-8B-Instruct both 0
```

Results land in `experiment-results/longbench/sketchwalk-<model_tag>-<sparsity_mode>/`. Each task's `raw_results.json` contains the per-task LongBench score and predictions.

### Default hyperparameters

We provide default hyperparameters below, but you may need to tune them a bit to get the best performance on your setting:

```
random_walk_hadamard_dim   = 128     # Hadamard sketch dimension
random_walk_window         = 0.05    # recent-window fraction of context
random_walk_sink           = 0.025   # sink-block fraction of context
random_walk_kblocks_frac   = 0.125   # top-k block fraction (≈ 20% total kept)
random_walk_degree         = 3       # walk-composition depth
random_walk_query_window   = 32      # decode query pooling window
walk_damping               = 0.25    # Damping factor β
```

## Citation

If you find this work useful, please consider citing:

```bibtex
@inproceedings{le2026sketchwalk,
  title     = {Scout Before You Attend: Sketch-and-Walk Sparse Attention for Efficient {LLM} Inference},
  author    = {Le, Hoang Anh Duy and Joshi, Sahil and Yang, Zeyu and Xu, Zhaozhuo and Shrivastava, Anshumali},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {306},
  year      = {2026},
  publisher = {PMLR},
  url       = {https://openreview.net/forum?id=uCLVPafHqd}
}
```
