---
title: "Reasoning-Aware Compression"
metatags:
    description: "Prune reasoning LLMs without breaking them: calibrate SparseGPT on on-policy chain-of-thought activations collected with SGLang, then serve the sparse checkpoint."
---

Pruning a reasoning model with a standard calibration set does more damage than pruning a
conventional LLM — and it can make the model *slower*. Reasoning-Aware Compression (RAC) fixes this
by changing what the pruning solver calibrates on: the model's own chain of thought, generated with
SGLang.

From [*Reasoning Models Can be Accurately Pruned Via Chain-of-Thought Reconstruction*](https://arxiv.org/abs/2509.12464)
(ICLR 2026).

## The problem

One-shot pruning methods such as SparseGPT and Wanda choose which weights to remove by minimizing a
layer-wise reconstruction error against a calibration activation matrix `X`:

```
min_{W'} || W X - W' X ||_F^2    s.t.  ||W'||_0 <= S
```

`X` is conventionally built from **prompt** tokens — a slice of C4, or a set of task prompts. That
is a fair proxy for a typical serving workload, where the prompt dominates the token count.

Reasoning models invert that ratio. They emit thousands of chain-of-thought tokens per query, so
almost every forward pass the pruned model will ever run is over a token it generated itself.
Calibrating only on prompts optimizes the pruned weights for a distribution the model barely visits.

The result is not a graceful accuracy decay. The pruned model starts to ramble: it produces longer
chains of thought *and* answers less accurately, so pruning increases end-to-end latency instead of
reducing it. At 50% sparsity on MATH-500, C4-calibrated DeepSeek-R1-Distill-Qwen-7B takes almost six
times as long to evaluate as the dense model it was meant to accelerate.

## The fix

RAC samples the dense model's own on-policy rollout during calibration and reconstructs the prompt
and decode activations jointly:

```
X_RAC = [ X_prompt , X_decode ]
```

The solver is untouched, so this is a drop-in change to any existing SparseGPT or Wanda workflow.

DeepSeek-R1-Distill-Qwen-7B, MATH-500, SparseGPT at 50% sparsity, 1M calibration tokens:

| Calibration set | acc@1 | Eval wall clock |
| --- | --- | --- |
| Dense (no pruning) | 0.936 | 23.3 min |
| C4 | 0.744 | 135.0 min |
| Task prompts only | 0.812 | 115.6 min |
| **RAC (prompts + on-policy CoT)** | **0.900** | **35.3 min** |

Across DeepSeek-R1-Distill-Qwen (1.5B–32B) and Qwen3 (1.7B–14B), the paper reports that RAC keeps up
to 95% of dense accuracy at 50% sparsity, improving on prompt-only calibration by up to 17 points.

## Using it

SGLang ships the recipe as a runnable example at
[`examples/usage/reasoning_aware_compression`](https://github.com/sgl-project/sglang/tree/main/examples/usage/reasoning_aware_compression),
in three phases:

| Phase | Script | What it does |
| --- | --- | --- |
| I | `rac_collect_traces.py` | `sgl.Engine` samples on-policy CoT traces into a calibration set |
| II | `rac_prune.py` | `llm-compressor` runs SparseGPT/Wanda against those activations |
| III | `rac_serve_and_eval.py` | SGLang serves the sparse checkpoint and scores MATH-500 |

Phase I is the expensive step — the paper's budget is 1M on-policy CoT tokens — and is where SGLang's
batched generation does the work. Phase II delegates the pruning solver to
[`llm-compressor`](https://github.com/vllm-project/llm-compressor), which is **not** an SGLang
dependency; install it separately with `pip install "llmcompressor>=0.12.0"`.

```bash
cd examples/usage/reasoning_aware_compression

python rac_collect_traces.py \
    --model-path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --dataset open-r1/OpenR1-Math-220k --prompt-column problem \
    --target-tokens 1000000 --output-dir ./rac_traces_math

python rac_prune.py \
    --model-path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --calibration ./rac_traces_math/traces.jsonl \
    --sparsity 0.5 --output-dir ./rac_pruned_50

python -m sglang.launch_server --model-path ./rac_pruned_50
```

The example README walks through building the paper's prompt-only baseline from the same prompts so
you can compare calibration strategies head to head.

## Evaluating a pruned reasoning model

Accuracy alone will hide the failure mode described above. Always report **mean completion length**
and **wall clock** alongside accuracy when comparing pruned reasoning checkpoints — a model that
scores two points lower while emitting three times the chain of thought is not a good trade.
`rac_serve_and_eval.py` reports all three.

## Related

- [Quantization](/docs/advanced_features/quantization) — the other axis of model compression, applied
  at serving time.
