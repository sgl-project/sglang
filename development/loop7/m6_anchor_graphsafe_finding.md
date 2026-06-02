# Loop 7 — anchor-budget variant ported to the graph-safe path (AC-3 completion)

R6 put `scorer_norm` + `head_agg` on the graph-safe path; the third AC-3 variant
(`anchor_mode`) was still eager-only (post-topK force-include was a per-row Python
loop). R9 ports it, completing AC-3's variant coverage on the production
CUDA-graph path.

## What was done
- **Tensorized `_force_include_anchor`.** The per-row Python loop (`.item()`,
  per-`b` `for`) is replaced by a fully tensorized, fixed-shape, host-sync-free
  implementation: `effective_budget = min(anchor_budget, valid_count, seq_len)`;
  anchor positions via `_anchor_positions_tensor` (recency/global/strided, with
  strided's ascending set-dedup); evict the k lowest-score non-anchor selected
  (stable score-asc / position-asc tie-break via `_stable_argsort_ascending`);
  insert the first k missing anchors; re-sort. It is **bit-identical** to the
  former reference (fuzz: **2000/2000** random cases match) and is now used by
  BOTH the eager logical path and the graph-safe path — so they cannot diverge.
- **Graph-safe integration.** `retrieve_topk_graph_safe` calls it after the
  top-K (when `anchor_mode != off`); `anchor_mode`/`anchor_budget` thread through
  the deepseek_v2 graph-safe call site and `capture_decode_step`. Under CUDA-graph
  capture the extra ops are captured once and replay reuses their memory.
- **Guards relaxed.** `ds_scorer_is_graph_safe()` now returns `True` (ALL
  non-learned variants are graph-safe); the validator / `_force_eager_select` /
  capture guard no longer force eager for anchor; the serve script only adds
  `--disable-cuda-graph` for the recall-oracle diagnostic.

## Evidence (GPU)
- **Eager-vs-graph bit-identical selection** over the full matrix
  `scorer_norm{off,cosine,hybrid} × head_agg{max,mean} × anchor_mode{off,recency,
  global,strided}` (24 combos) on **fp16 + int8**
  (`TestGraphSafeScorerEqualsEager`), incl. short/long requests + over-budget.
- **Real CUDA-graph capture/replay**: a hybrid+recency-anchor selection captured
  in a `torch.cuda.CUDAGraph` replays **byte-identical to eager** and with **zero
  new allocations** (`test_anchor_graph_safe_replay_zero_alloc`).
- **TP=8 cross-rank determinism** holds with the tensorized anchor
  (`test_ds_scorer_tp_determinism.py`).
- Default (anchor off) byte-identical; **346 DS unit tests pass**.

## AC-3 status
All three AC-3 non-learned variants are now flag-gated, graph-safe, and
non-regressing: channel-normalization (cosine/hybrid) + head-aggregation [R6] and
anchor-budget (recency/global/strided) [R9]; default byte-identical when off;
within-budget parity + MMLU ≤1.0pp + binding 16K uplift [R7]; TP=8 equality
[R3/R9]. **AC-3 variant coverage complete on the production path.** (The winning
landed config remains the hybrid scorer; anchor is the now-graph-safe exploratory
variant.)

## Artifacts
`selection_kernel.py` (tensorized `_force_include_anchor` + `_anchor_positions_tensor`
+ `_stable_argsort_ascending`; graph-safe threading), `deepseek_v2.py`,
`cuda_graph.py`, `validator.py`, `serve_double_sparsity.sh`,
`test_scorer_variants.py` (24-combo matrix + anchor replay no-alloc).

## Remaining (queued)
- **AC-4 lifted-budget** (task13–17): the Tier-2.A workstream.
- **AC-6 perf consolidation + final strategic-gate decision record** (task19–20).
