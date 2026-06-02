# M8 — Served lifted-budget 4K recall recovery (task15)

## Result
The opt-in lifted-budget decode path **materially recovers 4K NIAH recall** on the
served path: **DS-lifted `lifted_budget_top_k=4096` = 95% (19/20)** vs
**DS-default `top_k=2048` = 75% (15/20)**, **+20 pp**, same node, same prompts,
both EAGER, N=20. The lifted point (0.95) **exceeds the DS-default 95% Clopper–Pearson
CI high (0.9134)** → MATERIAL by the plan's directional rule.

| variant | hits/N | recall | 95% CP CI | admission_fail | prompt_tokens |
|---|---|---|---|---|---|
| DS-default top_k=2048 | 15/20 | 75% | [50.9%, 91.34%] | 0 | 4368–4408 |
| DS-lifted lifted_budget_top_k=4096 | 19/20 | **95%** | [75.13%, 99.87%] | 0 | 4368–4408 |
| **uplift** | | **+20 pp** | lifted 0.95 > base_hi 0.9134 → **material** | | |

This confirms the M0 oracle's 4K attribution (budget-limited: score-only
recall@2048≈44% → recall@4096≈86% → recall@8192≈100%, `m0_oracle_finding_r4.md`)
**on the served decode path**: widening the decode budget from 2048 to 4096 lets
the selector keep ~4096 of the ~4400-token 4K context, so the needle (oracle
score-rank ~2208) lands inside the budget.

## Why both servers are EAGER (clean comparison)
The lifted path is eager-only (its `dequantize_k_cache_paged` is not graph-safe),
so the DS-default baseline was **re-measured EAGER on the same node** rather than
reusing a CUDA-graph number. This isolates the **budget** (2048 vs 4096) from the
eager-vs-graph upstream-numerics gap (BL-20260602-eager-vs-graph-recall-differs).
Both numbers are eager-mode and must be quoted as such; the graph-mode production
recall would differ and is a separate (task16-gated) measurement.

## Scope of the recovery (bounded-secondary, per M0)
Tier-2.A remains **bounded-secondary**: the M0 oracle showed 4K is budget-limited
(recovered here) but **16K is budget-partial (~46% cap) and 64K is scorer-limited**
(rank ≫ 8192). A wider decode budget cannot recover 16K/64K — those are served by
the **landed Tier-2.B hybrid scorer** (AC-3, 16K 6%→38% material). So this 4K
served recovery is the expected, characterized win for the wider-budget lever, not
a long-context fix. A `lifted_budget_top_k=8192` run (M0 predicted ~100% at 4K) was
not run this round; 4096 already clears the materiality bar.

## Provenance
- **Commit**: R14 working tree on `2ba4dafc1` (the R13 wired lifted branch +
  the R14 `serve_double_sparsity.sh` `LIFTED_BUDGET` knob).
- **GPU**: 8× NVIDIA H200 (sm90), TP=8.
- **Op-point**: DS int8 compact table, `mem_fraction_static=0.7`, page 64, fp8 KV,
  `--dsa-prefill-backend/--dsa-decode-backend flashmla_kv`, `--disable-overlap-schedule`,
  `--disable-piecewise-cuda-graph`, radix-off, **`--disable-cuda-graph` (eager)**.
- **DS config (lifted)**: `{"top_k":2048, "signature_dtype":"int8", "page_size":64,
  "enable_lifted_budget_decode":true, "lifted_budget_top_k":4096, ...}`.
- **DS config (default)**: `{"top_k":2048, "signature_dtype":"int8", "page_size":64,
  "enable_lifted_budget_decode":false, "lifted_budget_top_k":0, ...}`.
- **Harness**: `development/loop7/niah_ds_baseline.py` (use_chat, `max_new_tokens=64`),
  NIAH 4096 words, N=20, served-vs-admission separated.
- **Launch**: lifted = `LIFTED_BUDGET=1 LIFTED_BUDGET_TOP_K=4096 TOP_K=2048
  LOOP7_MEASUREMENT=1 bash development/serve_double_sparsity.sh`; default =
  `TOP_K=2048 LOOP7_MEASUREMENT=1 EXTRA_SERVER_ARGS=--disable-cuda-graph bash
  development/serve_double_sparsity.sh`.
- **Servability**: served 20/20, **0 admission failures** on both (the wider
  selection/output buffers at `ds_max_top_k=4096` did not break admission at mem 0.7).
- **Artifacts**: `niah_ds_lifted4096.json`, `niah_ds_default2048_eager.json`,
  `ds_lifted_vs_default_recall_4k.json` (+ `lifted_recall_matrix.py`).

## Runtime confirmation that the lifted path engaged
A short `/generate` smoke on the lifted server returned coherent text
(" Paris. The capital of the United States is Washington, D.C. …") with the
`double_sparsity` response meta non-None (`dense_fallback=0`) — the lifted decode
branch served real traffic, not a silent fallback.

## Determinism + correctness backing this served result
- Backend-level GPU test of the actual wired `_forward_lifted_budget` at 4096/8192
  (prefix-sharing, duplicate slot, `valid_lengths < width`) matches a reference
  (`test_lifted_budget_decode.py::TestLiftedBudgetBackendDecode`).
- TP=8 lifted-width selected-index/valid-length equality at 4096 and 8192 across 8
  gloo ranks (`test_ds_scorer_tp_determinism.py::TestTP8LiftedWidthDeterminism`).

## Remaining (task16/17)
The landed path is eager-required (the dequant allocates). Production graph use
needs the alloc-free `out=` dequant + scratch (task16); the task17 disposition then
records this served recall evidence + the graph-safety decision with the DSA
default untouched.
