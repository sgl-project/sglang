# M10 — Lifted-budget decode under production CUDA graph (task16 close)

## Result
The opt-in lifted-budget decode path now runs **under production CUDA graph** (no
longer eager-only) and recovers 4K recall in graph mode:

- **DS-lifted `lifted_budget_top_k=4096`, CUDA-graph mode, NIAH 4K, N=20 = 95%
  (19/20)** [`niah_ds_lifted4096_graph.json`], served 20/20, **0 admission failures**.
- This **matches the eager 95%** (R14, `m8_lifted_recall_finding.md`) — at 4K the
  4096 budget keeps ~all of the ~4400-token context, so the recall is robust to the
  eager-vs-graph upstream-numerics shift. Per the eager-vs-graph lesson, **the
  graph-mode number is the binding production recall**; here it confirms the eager result.
- vs the DS-default `top_k=2048` 4K recall (~75%, eager same-node R14 / the mem-0.7
  baseline) → the **+20pp recovery holds in production graph mode**.

## Production-graph evidence (the task16 landing)
- **Live boot WITH CUDA graph** (no `--disable-cuda-graph`): the full V3.2 forward —
  including the lifted decode — captures and the server boots ("fired up").
- **Decode runs under the captured graph**: server log shows
  `Decode batch ... #token: 4416 ... cuda graph: True ... gen throughput ~14.5 tok/s`
  (conc-1, 4K context). (`double_sparsity` response meta is `None` under graph — the
  per-request summary host-syncs and is eager-only, so its absence CONFIRMS the
  decode ran captured, not eager.)
- **3.4× faster than eager**: the N=20 sweep took **13.8 s** (graph) vs **46.8 s**
  (eager, R14) — the captured graph removes the eager per-op launch overhead.
- **Memory**: at `--cuda-graph-max-bs 8`, mem 0.7, the run used ~114 GB/GPU on H200
  (143 GB) — the model + KV pool dominate; the lifted scratch
  (`lifted_compact_kv [max_bs*4096, 1, 576] bf16`, ~70 MB summed over the captured
  batch sizes ≤ 8) is small. The lifted scratch scales with the capture batch, so a
  high-concurrency capture would need a larger budget; the 4K recall lever is a
  low-concurrency long-context use, so a bounded `--cuda-graph-max-bs` is the op-point.

## Graph-safety (offline proof backing the live run)
- `dequantize_k_cache_paged_out` (alloc-free) + `build_lifted_compact_kv_fixed`
  (fixed `[bs*width]` shape) + a preallocated `DSGraphState` lifted scratch + a q
  head-padding scratch: the wired backend `_forward_lifted_budget` replays
  **zero-alloc** under a real `torch.cuda.CUDAGraph` at **4096 and 8192**
  (`test_lifted_budget_decode.py::TestLiftedBudgetBackendGraphSafe`,
  `::TestLiftedBudgetGraphSafe`), matching the eager reference.

## Provenance
- **Commit**: R17 tree (graph wiring) at `6453562e9`.
- **GPU**: 8× NVIDIA H200 (sm90), TP=8.
- **Op-point**: DS int8 / mem 0.7 / page 64 / fp8-KV / flashmla_kv backends /
  radix-off / **CUDA graph ON** / `--cuda-graph-max-bs 8`.
- **DS config**: `{"top_k":2048, "signature_dtype":"int8", "enable_lifted_budget_decode":true,
  "lifted_budget_top_k":4096, ...}`.
- **Launch**: `LIFTED_BUDGET=1 LIFTED_BUDGET_TOP_K=4096 TOP_K=2048 LOOP7_MEASUREMENT=1
  EXTRA_SERVER_ARGS="--cuda-graph-max-bs 8" bash development/serve_double_sparsity.sh`
  (the serve script no longer forces `--disable-cuda-graph` for `LIFTED_BUDGET=1`).
- **Artifacts**: `niah_ds_lifted4096_graph.json` (graph), `niah_ds_lifted4096.json` +
  `niah_ds_default2048_eager.json` (eager comparison), `ds_lifted_vs_default_recall_4k.json`.

## Default-path non-regression
Every lifted code path is behind a default-off `getattr(self,
"ds_lifted_budget_decode", False)` guard; the default `flashmla_kv` decode and its
`indices.shape[-1] == dsa_index_topk` (2048) assert are untouched. The full DS unit
suite (347 + 9 subtests) passes.
