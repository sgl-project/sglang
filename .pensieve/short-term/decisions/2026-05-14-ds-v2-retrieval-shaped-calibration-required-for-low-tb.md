---
id: 2026-05-14-ds-v2-retrieval-shaped-calibration-required-for-low-tb
type: decision
title: Double Sparsity calibration must be retrieval-shaped to pass NIAH below token_budget=8192
status: active
created: 2026-05-14
updated: 2026-05-14
tags: [double-sparsity, calibration, niah, retrieval, llama]
---

# Double Sparsity calibration must be retrieval-shaped to pass NIAH below token_budget=8192

## Decision

For 70B/Llama-3.1-Instruct at 128K context with the native DS sparse-decode
path, **calibrate K-channels on NIAH-shaped (needle + question) prompts**
when targeting `token_budget <= 4096`. Wikitext calibration is fine for
the headline `tb=8192` operating point but fails the NIAH quality gate
at lower budgets even though the perf gate is well above the 0.90×
ratio threshold.

## Evidence (this branch, 2026-05-14)

| operating point | calibration | TBT ratio | NIAH (n=10) | Quality gate |
|---|---|---:|---:|:---|
| conc=16 / tb=2048 | wikitext  | 0.84× PASS | 4/10 (0.40) | **FAIL** (delta −0.40) |
| conc=16 / tb=2048 | retrieval | 0.8035× PASS | **10/10 (1.00)** | **PASS** (delta +0.20) |
| conc=16 / tb=8192 | wikitext  | 0.996× FAIL | 9/10 (0.90) | PASS |
| conc=32 / tb=8192 | wikitext  | 0.8800× PASS | 10/10 (1.00) | PASS |

Same kernel shape, same hyperparameters, same operating point — only
the K-channel selection differs. Retrieval-shaped calibration shifts
19% of the heavy-channels picks at `heavy_channels=32`; the shifted
channels are the ones that activate on Note/Question retrieval patterns
in long context.

## How to produce the retrieval calibration

```bash
# 1. Generate synthetic NIAH-shaped prompts.
PYTHONPATH=python python3 scripts/double_sparsity/make_retrieval_calib_prompts.py \
    --output /workspace/ds_retrieval_calib_prompts.txt \
    --n-prompts 128 --target-chars 20000 --seed 0

# 2. Run calibrate.py with --prompts-file.
PYTHONPATH=python python3 scripts/double_sparsity/calibrate.py \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --output /workspace/calib_llama_3_1_70b_retrieval_s32.json \
    --heavy-channels 32 --n-samples 64 --seq-len 4096 \
    --prompts-file /workspace/ds_retrieval_calib_prompts.txt \
    --device-map auto
```

## Why this is a project-specific decision (not knowledge)

The "calibrate on retrieval-shaped prompts" choice trades calibration
data shape against the quality gate. It's specific to the DS branch's
NIAH-as-quality-gate definition; it would not transfer to a different
quality metric. The model and the heavy_channels=32 setting are also
project-specific.

## Three exploration-reduction items

* **Fewer questions next time**: "do we need a calibration JSON for
  every `token_budget`?" → No, retrieval calib works at every tb >= 2048.
* **Fewer lookups**: result JSONs at
  `benchmark/double_sparsity/repro_session/conc16_move_left/*.json`,
  prompt file at `/workspace/ds_retrieval_calib_prompts.txt`,
  calibration JSON at `/workspace/calib_llama_3_1_70b_retrieval_s32.json`.
* **Invalidation conditions**: this decision is scoped to the native
  sparse-decode path on 70B/Llama-3.1. If we adopt a different model
  family, swap the K-projection lifecycle, or change `heavy_channels`,
  re-measure. Note: the calibration JSON's `heavy_channels` must match
  `--double-sparsity-heavy-channels` at server start — startup fails
  loud if they differ.

## Context Links
- Based on: [[knowledge/ds-native-sparse-decode-pareto/content]]
- Leads to: [[decisions/2026-05-14-ds-v2-replace-fa3-adaptor-with-native-sparse-decode]]
- Related: [[knowledge/ds-flashinfer-top-k-page-table-boundaries/content]]
