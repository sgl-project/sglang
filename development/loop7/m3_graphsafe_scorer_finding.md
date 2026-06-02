# Loop 7 — Tier-2.B scorer ported to the graph-safe path (AC-3 landed path)

The winning Tier-2.B scorer (R5: material 16K uplift) was eager-only and required
`--disable-cuda-graph`. R6 ports it onto the **production CUDA-graph path** and
validates it. This is the AC-3 "landed graph-safe path."

## What was ported
`scorer_norm ∈ {cosine, hybrid}` (with `scorer_norm_hybrid_threshold`) and
`head_agg ∈ {max, mean}` are implemented directly in the graph-safe Triton
scorer `_logical_score_kernel` (3 new `tl.constexpr`: `SCORER_NORM`,
`HEAD_AGG_MEAN`, `HYBRID_THRESHOLD`):
- **cosine** = unit-normalize the weighted query and the token signature per head
  (scale ignored, it cancels under normalization) — `dot/((‖q_proj‖+eps)·(‖sig‖+eps))`
  computed normalize-then-sum to match the eager form.
- **hybrid** = per-request switch read from `seq_len_i` in-kernel: cosine when
  `seq_len > threshold`, raw (scaled) channel-dot otherwise.
- **head_agg mean** = sum-then-divide over heads (vs the default cross-head max).
- The R17 full-context early-exit and the int8 dequant-scale handling are
  preserved; default (`off`/`max`) is byte-identical to the prior kernel.

Flags thread config-borne through `retrieve_topk_graph_safe`, the deepseek_v2
graph-safe call site, and the DS `capture_decode_step` path. The startup
validator + `_force_eager_select` + `capture_decode_step` guard now use a new
`ds_scorer_is_graph_safe(config)` (only a non-default **anchor_mode** still
requires `--disable-cuda-graph`; anchor force-include is a post-topK per-row op
not yet graph-safe — queued).

## Eager-vs-graph equality (the correctness proof)
GPU test `TestGraphSafeScorerEqualsEager` (and a standalone sweep): the graph-safe
Triton scorer produces **bit-identical** `selected_indices` + `valid_lengths` to
the eager `retrieve_topk_via_labels` for **all 12 combos**
`scorer_norm{off,cosine,hybrid} × head_agg{max,mean}` on **both fp16 and int8**
signatures, with short/long requests crossing the hybrid threshold. So the port
changes the kernel implementation, not the selection.

## Live: the hybrid scorer SERVES under CUDA graph
A `scorer_norm=hybrid` server now boots with **CUDA graph ON** (no
`--disable-cuda-graph`; validator allows it; `Capture cuda graph begin` confirmed
on all 8 TP ranks) — previously impossible.

## Production (graph-mode) served recall — N=20, the honest production number
| length | graph DS-default | graph DS-hybrid (Tier-2.B) | note |
|--------|------------------|-----------------------------|------|
| 1024w (≤budget) | 100% | 100% [83,100] | parity |
| 4K | 75% | 75% [50.9,91.3] | hybrid uses RAW ≤8192 ⇒ == default (expected) |
| 16K | 5% [0.1,24.9] | **25% [8.7,49.1]** | cosine regime: +20pp over default |

**Finding:** under the production CUDA-graph path the hybrid scorer **still
improves 16K recall (5% → 25%, +20 pp)** over the default, and is unchanged at 4K
(the ≤8192 raw regime is identical to default — as designed). The 16K production
uplift is **marginally material** (25% just exceeds the default CI high of 24.9%
at N=20); a binding claim needs N≥50 (queued).

**Honest correction vs the eager R5 numbers.** The R5 eager-mode measurement
(`--disable-cuda-graph`) reported hybrid 4K=85% / 16K=40%. The graph-safe
production path gives 75% / 25%. Selection is bit-identical between the eager and
graph-safe *scorer code* (proven above), so the difference is **upstream
eager-vs-graph model-forward numerics** (the query projection feeding the scorer
differs slightly under CUDA-graph capture vs eager), which shifts a few needles
at both lengths and affects the default too (graph default 4K=75% vs the eager
regime). The **binding production recall is the graph-mode number**; the eager
research number over-stated it. This is exactly why AC-3 requires a landed
graph-safe path measured at the production op-point, not an eager research path.

## Artifacts
`niah_ds_hybrid_graphsafe.json` (graph-mode N=20). Code: `selection_kernel.py`
(kernel + threading + `ds_scorer_is_graph_safe`), `deepseek_v2.py`,
`cuda_graph.py`, `validator.py`, `serve_double_sparsity.sh`. Test:
`test_scorer_variants.py::TestGraphSafeScorerEqualsEager` (+ guard/predicate tests).

## Remaining for AC-3 (queued)
- **N≥50 binding 16K** at the graph-mode op-point (firm up the marginal 25%).
- **MMLU ≤1.0pp re-anchor** (single-node mem0.7) for non-regression.
- **graph-vs-eager perf delta** (AC-6) now that the hybrid runs under graph.
- **anchor_mode graph-safe port** (still eager-only).
