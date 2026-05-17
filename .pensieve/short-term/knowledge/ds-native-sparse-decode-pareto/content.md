---
id: ds-native-sparse-decode-pareto
type: knowledge
title: DS native sparse-decode perf/quality Pareto at 70B/TP=8/128K
status: active
created: 2026-05-14
updated: 2026-05-14
tags: [double-sparsity, benchmark, niah, tbt, calibration]
---

# DS native sparse-decode perf/quality Pareto at 70B/TP=8/128K

## Source
- Bench artifacts: `benchmark/double_sparsity/repro_session/sweep_70b_128k_tbt_win/`
  and `benchmark/double_sparsity/repro_session/conc16_move_left/`
- Design + status: `benchmark/double_sparsity/DESIGN.md`
- Native path: `python/sglang/srt/layers/attention/triton_ops/double_sparsity_native_decode.py`

## Summary
At the wikitext calibration baseline, the TBT visible-win gate
(`tbt_on ≤ 0.90 × tbt_off`) and the NIAH quality-guard gate
(`niah_on ≥ niah_off − 0.02`) cannot both pass at `concurrency=16` for any
single `token_budget`; both pass cleanly at `concurrency=32` with
`token_budget=8192`. **Retrieval-shaped calibration unlocks conc=16 /
tb=2048** — both gates pass (TBT 0.8035×, NIAH +0.20). The kernel
shape is unchanged; only the K-channel selection differs.
See [[decisions/2026-05-14-ds-v2-retrieval-shaped-calibration-required-for-low-tb]].

## Content

### Bench points (70B/TP=8, 128K ctx, output_len 256–512, FA3 backend)

| tb   | conc | calib     | TBT(off) ms | TBT(on) ms | TBT ratio | NIAH(off) | NIAH(on) | NIAH delta |
|-----:|-----:|-----------|------------:|-----------:|----------:|----------:|---------:|-----------:|
| 512  | 16   | wikitext  | 27.94 | 22.99 | **0.82×** PASS | 0.80 | 0.00 (n=5) | −0.60 FAIL |
| 2048 | 16   | wikitext  | 27.94 | 23.38 | **0.84×** PASS | 0.80 | 0.40 (n=10) | −0.40 FAIL |
| 8192 | 16   | wikitext  | 27.94 | 27.83 | 0.996× FAIL | 0.80 | 0.90 (n=10) | **+0.10** PASS |
| 8192 | 32   | wikitext  | 34.68 | 31.19 | **0.8995×** PASS | 0.80 | 0.90 (n=10) | **+0.10** PASS |
| **2048** | **16** | **retrieval** | **27.94** | **22.45** | **0.8035×** **PASS** | **0.80** | **1.00** (n=10) | **+0.20** **PASS** |
| 8192 (recheck) | 32 | wikitext  | 34.68 | 30.52 | **0.8800×** PASS | 0.80 | 1.00 (n=10) | **+0.20** PASS |

### Why conc=16 needs `tb ≤ 4096` (and therefore retrieval calib)

- Dense decode TBT is **KV-bandwidth-bound** at 128K bs ≥ 4; it scales
  ~linearly with `batch × seq_len` (27.94 ms at conc=16 → 34.68 ms at
  conc=32, +24%).
- Native DS-on TBT is bounded by `total_selected = top_k + sink + recent`,
  not by `seq_len`. From conc=16 to conc=32 it grows only +12% (driven by
  per-batch fixed costs: `_compute_q_label`, `set_kv_buffer`, etc.).
- With wikitext calibration, `tb=8192` is the only budget that surfaces
  enough needle tokens for NIAH at conc=16 — but at that budget
  `total_selected` is large enough that the TBT ratio bumps up to 0.996×
  (perf fail).
- The crossover where DS-on/DS-off TBT ratio drops below 0.90 at
  `tb=8192` is right of conc=16; conc=32 lands it.
- With retrieval calibration, `tb=2048` surfaces needle tokens reliably
  (NIAH 10/10) AND keeps `total_selected` small enough that TBT ratio is
  0.8035× — both gates pass simultaneously at conc=16.

### Why narrow `token_budget` breaks NIAH

`token_budget=512` = 0.4% of 128K context. The needle-in-a-haystack probe
places a 10-token needle at position ~65K. For DS-on to retrieve it, the
needle's K_label must score in the top 512 of ~131K positions under
Q_label = "What was the magic phrase?". Wikitext calibration shapes
K_label for **next-token language modeling**, not for retrieval — the
heavy channels don't strongly flag needle-like K patterns at narrow
budgets. Quality recovers monotonically as `token_budget` widens (0.00 →
0.40 → 0.90), saturating around `tb=8192` (~6% coverage).

### What the synthetic profile shows is fast and what is slow

`benchmark/double_sparsity/repro_session/profile_native_decode.py`, at
70B/TP=8 shape (bs=1, h_kv=1, h_q=8, d=128, S=32):

| phase | µs/layer | ms × 80 layers |
|---|---:|---:|
| score (Triton)              |  16 | 1.3 |
| `torch.topk`                |  62 | 5.0 |
| build_selected_physical (Triton) | 16 | 1.3 |
| sparse attn stage2+3 (Triton)    | 36 | 2.9 |
| inter-op overhead                 | —  | 1.6 |
| **TOTAL**                   | 151 | **12.07** |

- All phases are essentially flat in `seq_len` (microbench:
  `benchmark/double_sparsity/repro_session/microbench_sparse_attn.py`
  — 37 µs at selected=512 across 32K/64K/128K, ≤2% jitter).
- `torch.topk` dominates and isn't user-code. The remaining wins are
  fusion (e.g. inline Q_label into the score kernel saves Python launch
  cost, was stashed when synthetic showed neutral).

## When to Use

Read this before:

- Adjusting `--double-sparsity-token-budget` or `--double-sparsity-sink-tokens`
  / `--double-sparsity-recent-tokens` for a published claim.
- Concluding "DS doesn't beat dense" from a single bench point at
  conc ≤ 8. The DS win lives at high concurrency / long context where
  KV bandwidth dominates dense.
- Investigating a NIAH-quality regression. Bumping `token_budget` is
  the first lever; calibration shape is the deeper fix.

## Context Links
- Based on: [[decisions/2026-05-14-ds-v2-replace-fa3-adaptor-with-native-sparse-decode]]
- Leads to: [[maxims/cuda-graph-capture-rejects-host-syncs-in-eligibility-gates]]
- Related: [[maxims/preserve-user-visible-behavior-as-a-hard-rule]]
