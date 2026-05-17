# FTKA evaluation: integrate / defer / reject

**Scope.** Evaluate `tsinghua-ideal/flash-topk-attention` (FTKA, commit
`d8803b29961c44d77a747636ad4282bd7a9094af`) as an **implementation
substitute** for the DS native sparse-decode path's *score* and *top-k*
phases. No algorithm change: same K-channel calibration, same token
budget, same sink/recent retention, same selected-token contract. The
Twilight adaptive-budget / top-p pruner / weight estimator family is
explicitly out of scope for this pass.

**Method.** A new microbench
(`benchmark/double_sparsity/repro_session/microbench_ftka_backends.py`)
times four paths at 76 DS shapes (3 contexts × 5 bs × 5 top_k + an
h_kv=8 stress row). For each (path, shape) entry it records mean µs/call,
selected-physical-id set parity vs the torch baseline, and CUDA-graph
capture/replay status under an isolated probe. Raw artifacts:
[`results.json`](./results.json), [`results.md`](./results.md).

The four paths:

| ID | Path | What it substitutes |
|----|------|---------------------|
| **P1** | `score_triton + torch.topk + build_selected_physical` | — (current production baseline) |
| **P2** | `score_triton + flashinfer.top_k_page_table_transform + append_sink_recent` | top-k step only |
| **P3** | `score_triton + ftka.cuda_ops.raft_topk + build_selected_physical` | top-k step only |
| **P4** | `ftka.cuda_ops.batched_sparse_gemv + ftka.cuda_ops.raft_topk + build_selected_physical` | score + top-k |

## Environment of this run

| Item | Value |
|------|-------|
| GPU | NVIDIA H200 |
| torch | 2.11.0+cu130 |
| flashinfer | `0.6.11.post1` (available) |
| ftka | **not installed** — paths P3 and P4 are reported as `skipped` |

The skip-gracefully path is the design contract. `ftka` is treated as an
optional dep; the script and the new `ftka_raft_topk` selector backend
both raise a clear `RuntimeError` with installation guidance when missing
(see `selector_backends.FtkaRaftTopKSelector.__init__`).

## Component-level recommendation

| Component | Decision | Evidence in this run |
|-----------|----------|----------------------|
| `torch.topk` selector (current default) | **KEEP** | P1 ran clean at all 76 shapes; parity is the oracle; isolated capture probe passes. |
| `flashinfer_topk_page_table` selector | **NO STATE CHANGE** | 46 P1↔P2 pairs (only those at `top_k ≤ 2048` per the FlashInfer ceiling); median speedup **1.22×**, range **0.87×–1.38×**. Two slowdown cells at small `bs × top_k`. Production constraint unchanged: capture-unsafe inside SGLang's nested-stream graph context (this microbench's isolated probe does pass capture; that delta is real and previously documented). |
| **`ftka_raft_topk` (P3)** | **DEFER — harness ready, dep not installed** | Backend wired (`FtkaRaftTopKSelector`), parity test added (skipped when ftka missing), microbench will fill in the row when ftka is present. No evidence to integrate today. |
| **`ftka_gemv+ftka_topk` (P4)** | **DEFER — *and* expect REJECT in current form** | Score-kernel substitution requires building a paged-cache view (`kv_indices` / `kv_indptr` / `kv_last_page_len`) over our flat K-label cache. `req_to_token` changes every decode step, so the transform is *per-step*, not amortized. The microbench includes the transform cost in a dedicated `layout_transform_us` field so the future measurement will surface whether the FTKA GEMV speedup is large enough to absorb that cost. The expected answer is no — but the harness will measure it instead of guess. |

## Gate evaluation (from PLAN.md decision rules)

### FTKA `raft_topk` → DS selector integration gates

| Gate | Status today | Notes |
|------|--------------|-------|
| Output set parity with `torch.topk` | **PENDING** | Parity test added; runs when ftka is present (`test_selector_backends.TestSelectorParity.test_ftka_raft_topk_matches_torch`). |
| ≥1.15× selector speedup at `bs ≥ 16` for `top_k ∈ {1024, 2048}` | **PENDING** | At those shapes today, FlashInfer P2 reaches 1.20–1.31×. RAFT top-k is similar-class radix-based, so plausible target — needs the measurement. |
| Supports `top_k ≥ 8192` (or cleanly documented lower ceiling) | **PROBABLE OK** | RAFT `decode_select_k` is radix-based, no fixed-k ceiling apparent in the source. Microbench grid includes `top_k = 8192` so this would surface. |
| Survives CUDA graph capture/replay | **PENDING** | The microbench's `probe_graph` function will test this and emit `ok` / `capture_fail:<reason>` / `replay_fail:<reason>` per shape when ftka is installed. |
| Does not pin SGLang to an incompatible dep stack | **MOSTLY OK** | FTKA's `requirements.txt` declares `flashinfer-python==0.2.0.post1`, `torch==2.5.0`, `numpy==2.1.2`. The installed SGLang env runs `torch==2.11`, `flashinfer==0.6.11.post1`. Either FTKA installs cleanly against the newer deps (likely — RAFT top-k itself is plain CUDA/C++) or it pins us back; the latter is a reject. |

### FTKA `batched_sparse_gemv` → DS score integration gates

| Gate | Status today | Notes |
|------|--------------|-------|
| Score parity within bf16/fp32 tol vs current score kernel | **PENDING** | Microbench compares P4's selected-id set vs P1; tighter score-value parity would need a per-row dot-product comparison (out of scope today). |
| ≥1.10× score-kernel speedup at 128K for `bs ≥ 16` | **PENDING + UNLIKELY** | Our score kernel is already at **~25µs median, ~87µs at bs=32/ctx=128K**. FTKA's GEMV has comparable arithmetic intensity. The gap to close is small, and the layout-transform cost (next gate) eats most of it. |
| Consumes our K-label side cache, or layout-transform cost is included | **STRUCTURALLY ADVERSE** | Our K-label cache is `[T_pool, H_kv, S]` indexed by physical row id from `req_to_token`. FTKA expects a paged KV cache with explicit `kv_indices/kv_indptr/kv_last_page_len`. With `page_size=1`, the transform is a per-request flatten + cumsum; small at low bs but materializing `kv_indices = req_to_token.flatten()` allocates ~bs × max_ctx int32 cells *every step*. This is structurally adverse for production. |
| Survives CUDA graph capture/replay | **PENDING** | Same `probe_graph` will tell us. |

## Observations on the FlashInfer rerun

Reconfirmed properties:

- **P2 median speedup 1.22×** at the 46 shapes where `top_k ≤ 2048`. The
  PLAN ≥1.15× gate is met at every `bs ≥ 16` shape except the largest two
  (ctx=128K, k=2048 at bs=16 gives 1.06×; ctx=128K, k=1024 at bs=16 gives
  1.13×). At bs=32 across all contexts the gate is met cleanly
  (1.22×–1.31×).
- **P2 slowdown cases**: bs=8/ctx=32K/k=512 → 0.87×, bs=4/ctx=128K/k=2048
  → 0.98×. Both are at small batch × small budget where FlashInfer's
  kernel-launch fixed cost is not amortized.
- **Isolated CUDA-graph probe passes for P2 at every shape**, while
  inside SGLang's nested-stream capture region the kernel still fails
  with `Triton load_binary [CUDA] illegal memory access`. The delta is
  context-dependent JIT binding, documented in
  `python/sglang/srt/mem_cache/sparsity/algorithms/selector_backends.py`
  (the block comment above `_row_to_batch_buffer`). This run does not
  change that posture: production still must use `selector_backend=torch`.

## Twilight-side ideas: explicit reject for this pass

For traceability, the algorithmic items in `tsinghua-ideal/Twilight`
that **change token-selection semantics** are not part of this
evaluation:

- Top-p pruner after the top-k step (`twilight/pyimpl/top_p.py`).
- Adaptive token budget / `HistoryBudgetInfo` accumulator.
- INT8 weight-estimator over K_label.
- The hierarchical elementwise-threshold variant.

These can be added later as offline oracle/control experiments labeled
`twilight_algorithmic_control`, never as "same algorithm" DS results.

## How to extend this evaluation when FTKA is installed

```bash
# Install FTKA against this repo's torch/flashinfer (try-then-verify):
pip install "git+https://github.com/tsinghua-ideal/flash-topk-attention.git@d8803b29961c44d77a747636ad4282bd7a9094af#egg=ftka"

# Verify FTKA imports without forcing a dependency rollback.
python3 -c "import ftka, ftka.cuda_ops; print(ftka.__file__)"

# Run the microbench (full grid, ~5 minutes on H200):
PYTHONPATH=python python3 benchmark/double_sparsity/repro_session/microbench_ftka_backends.py
#   --quick                  # 7 shapes, ~30s smoke test
#   --shape-limit N          # cap the grid to first N entries

# The parity test will then run (skipped today):
PYTHONPATH=python python3 -m pytest \
    test/registered/unit/mem_cache/sparsity/test_selector_backends.py \
    -k ftka -v
```

If results.json shows **set parity = ok AND graph_status = ok AND
speedup ≥ 1.15× at bs ∈ {16, 32} / top_k ∈ {1024, 2048}**, escalate to the
e2e step in PLAN.md: rerun the two README headline operating points
(conc=16/tb=2048 retrieval; conc=32/tb=8192 wikitext) with
`--double-sparsity-selector-backend=ftka_raft_topk` and verify NIAH is
preserved and TBT improves or matches.

## Production default — unchanged

`--double-sparsity-selector-backend=torch` remains the default. No
runtime config change required.
