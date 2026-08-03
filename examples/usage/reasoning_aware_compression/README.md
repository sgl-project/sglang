# Reasoning-Aware Compression (RAC)

One-shot pruning of reasoning models, calibrated on the model's own chain of thought.

Implements the recipe from [*Reasoning Models Can be Accurately Pruned Via Chain-of-Thought
Reconstruction*](https://arxiv.org/abs/2509.12464) (Lucas, Behdin, Wang, Tang, Song, Mazumder;
ICLR 2026). Reference implementation: [RyanLucas3/Reasoning-Aware-Compression](https://github.com/RyanLucas3/Reasoning-Aware-Compression).

## Why

Layer-wise one-shot pruning picks weights by minimizing a reconstruction error against a
calibration activation matrix `X`:

```
min_{W'} || W X - W' X ||_F^2    s.t.  ||W'||_0 <= S
```

Every standard pipeline builds `X` from **prompt** tokens — C4 text, or task prompts. That is a
reasonable proxy when `|prompt| >> |output|`. Reasoning models invert the ratio: they emit
thousands of chain-of-thought tokens per query, so nearly all of the forward passes the pruned
model will ever run are over tokens it generated itself. Calibrating on prompts alone leaves the
solver optimizing for a distribution the model barely visits.

The failure mode this produces is worse than a plain accuracy drop. A poorly calibrated pruned
reasoning model **rambles** — it emits more thinking tokens and still answers less accurately, so
pruning makes it *slower*. From the paper (DeepSeek-R1-Distill-Qwen-7B, MATH-500, SparseGPT at 50%
sparsity, 1M calibration tokens):

| Calibration set | acc@1 | Eval wall clock |
| --- | --- | --- |
| Dense (no pruning) | 0.936 | 23.3 min |
| C4 | 0.744 | 135.0 min |
| Task prompts only | 0.812 | 115.6 min |
| **RAC (prompts + on-policy CoT)** | **0.900** | **35.3 min** |

RAC's fix is one line of the algorithm: sample the dense model's own rollout, and calibrate on the
prompt *and* decode activations,

```
X_RAC = [ X_prompt , X_decode ]
```

The solver is untouched — RAC is a drop-in calibration-set swap for SparseGPT, Wanda, and friends.

## Why this lives in SGLang

Collecting the rollout is Phase I of the paper's Algorithm 1, and it is the expensive half: the
paper's budget is 1M on-policy CoT tokens per calibration set. That is batched autoregressive
generation, which is what SGLang does. The pruning solver itself is not an inference-engine
concern, so Phase II delegates to [`llm-compressor`](https://github.com/vllm-project/llm-compressor),
and SGLang serves the result.

```
rac_collect_traces.py   Phase I   sgl.Engine samples on-policy CoT  -> traces.jsonl
rac_prune.py            Phase II  llm-compressor SparseGPT/Wanda    -> pruned checkpoint
rac_serve_and_eval.py   Phase III sgl.Engine scores MATH-500        -> acc + CoT length + runtime
```

## Setup

Phases I and III need only SGLang. Phase II additionally needs `llm-compressor`, which is **not** an
SGLang dependency:

```bash
pip install "llmcompressor>=0.12.0"
```

Tested against `llmcompressor` 0.12.0.

## Full run

Reproduces the paper's DeepSeek-R1-Distill-Qwen-1.5B row at 50% sparsity. The paper runs all
one-shot pruning experiments on a single H100.

```bash
cd examples/usage/reasoning_aware_compression

# Phase I -- 1M on-policy CoT tokens (the paper's budget), T_max = 8192, T = 0.6, top_p = 0.95.
python rac_collect_traces.py \
    --model-path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --dataset open-r1/OpenR1-Math-220k \
    --prompt-column problem \
    --target-tokens 1000000 \
    --output-dir ./rac_traces_math

# Phase II -- SparseGPT at 50% unstructured sparsity, calibrated on those traces.
python rac_prune.py \
    --model-path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --calibration ./rac_traces_math/traces.jsonl \
    --sparsity 0.5 \
    --output-dir ./rac_pruned_50

# Phase III -- accuracy *and* CoT length *and* wall clock.
python rac_serve_and_eval.py --model-path ./rac_pruned_50 --num-problems 500
```

To see what RAC actually buys, build the paper's prompt-only baseline from the same prompts and
compare the two checkpoints directly:

```bash
python rac_collect_traces.py \
    --model-path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --dataset open-r1/OpenR1-Math-220k --prompt-column problem \
    --calibration-mode prompt_only \
    --target-tokens 1000000 \
    --output-dir ./prompt_only_traces_math

python rac_prune.py \
    --model-path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --calibration ./prompt_only_traces_math/traces.jsonl \
    --sparsity 0.5 --output-dir ./prompt_only_pruned_50

python rac_serve_and_eval.py \
    --model-path ./prompt_only_pruned_50 ./rac_pruned_50 \
    --num-problems 500
```

`prompt_only` mode skips generation entirely, so it costs nothing but the tokenization pass.

## Smoke test

A few minutes on one GPU, to check the plumbing before committing to a 1M-token run:

```bash
python rac_collect_traces.py --model-path Qwen/Qwen3-0.6B \
    --dataset open-r1/OpenR1-Math-220k --prompt-column problem \
    --target-tokens 20000 --max-new-tokens 1024 --output-dir /tmp/rac_traces
python rac_prune.py --model-path Qwen/Qwen3-0.6B \
    --calibration /tmp/rac_traces/traces.jsonl --sparsity 0.5 --output-dir /tmp/rac_pruned
python rac_serve_and_eval.py --model-path /tmp/rac_pruned --num-problems 50 --max-new-tokens 2048
```

Phase I should report a decode share well above 50% — that gap is the activation mass prompt-only
calibration discards. Phase II should report a realized sparsity within a hair of the target.

## Models and datasets

The paper evaluates DeepSeek-R1-Distill-Qwen at 1.5B/7B/14B/32B and Qwen3 at 1.7B/8B/14B, pruned at
20–50% sparsity. Any of them work here; pass `--tp-size` to shard the larger ones.

Calibration prompts follow the paper: [`open-r1/OpenR1-Math-220k`](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k)
with `--prompt-column problem` for math, and a CodeForces prompt set with `--prompt-column prompt`
for code. `--dataset` also accepts a local `.jsonl` path.

## Notes

- **Chat template.** Traces are generated through the model's own chat template with the open-r1
  system prompt, which is what the reference implementation's published traces use. The calibration
  distribution *is* the method, so changing `--system-prompt` changes the result.
- **Token ids, not text.** Phase I emits token ids and Phase II consumes them directly, so the
  sequence the pruner reconstructs is exactly the sequence the model produced — no
  detokenize/retokenize drift.
- **Batch size 1 during calibration.** Padding tokens would enter the layer-wise Hessian as if they
  were real activations, which is precisely the contamination RAC exists to avoid.
- **`2:4` masks.** Pass `--mask-structure 2:4` for a semi-structured mask. The paper's headline
  results are unstructured (`0:0`).
- **Magnitude pruning** is in the reference implementation but not exposed here: `llm-compressor`'s
  magnitude modifier is a gradual, training-time modifier rather than a one-shot solver, and RAC is
  a one-shot method.
- **Grading.** `rac_serve_and_eval.py` does lightweight boxed-answer matching, enough to rank
  checkpoints. For paper-grade numbers use the `lighteval` harness that the RAC and open-r1 repos
  use.

## Citation

```bibtex
@inproceedings{lucas2026reasoning,
  title     = {Reasoning Models Can be Accurately Pruned Via Chain-of-Thought Reconstruction},
  author    = {Lucas, Ryan and Behdin, Kayhan and Wang, Zhipeng and Tang, Shao and Song, Qingquan and Mazumder, Rahul},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026}
}
```
