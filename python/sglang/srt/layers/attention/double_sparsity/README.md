# Double Sparsity

Double Sparsity (DS) is a query-aware KV sparsifier for the DeepSeek/GLM **MLA**
attention path. Each decode step it scores the request's KV positions from an
offline **channel mask** and selects the top-K to attend, serving them through
the FlashMLA sparse (`flashmla_kv`) path. It targets GLM-5.1-FP8 (which reuses
the DeepSeek MLA path) and is enabled with `--enable-double-sparsity`.

Calibration, serving, and the performance eval are documented in
[`benchmarks/DOUBLE_SPARSITY.md`](../../../../../../benchmarks/DOUBLE_SPARSITY.md).

## How selection works

The score reuses the **absorbed-latent identity**: for the MLA latent `c_kv`, the
no-PE attention logit equals `v_h · c_kv` where `v_h = W_UK_nope^T · q_nope`. So
DS scores directly off the *resident* MLA latent — no separate label cache:

```
score[b, t] = agg_h( v_h[b] · c_kv[t] )      # agg_h = max (default) or mean
```

- **scorer_norm** — `cosine` (default) divides each per-head dot by the query/key
  norms; `off` is the raw channel-dot. Same numerator, both graph-safe.
- **include_current_slot** (default on) — force-includes the current decode
  token's own slot, which the slot-validity mask would otherwise hold out.
- **rope_aware_score** (opt-in) — adds the RoPE term `q_pe·k_pe[t]` to recover
  long-context accuracy; fail-closed on any non-validated runtime.

The channel mask (`channel_selection`, `channel_weights`, `[L, H, label_dim]`) is
calibrated offline, TP-agnostic, sliced per rank at bind time, and pinned by a
content SHA-256 the loader verifies.

## CUDA-graph strategy

Decode runs under CUDA-graph capture, so selection is **replay-stable**: all
scratch is pre-allocated in `DSGraphState` (zero allocation on replay) and every
device branch is in-kernel (no host syncs in the captured region).

The cost lever is the **selector-width ladder**: scoring the full `req_to_token`
width (~200k) every step is wasteful, so DS captures graphs at a small set of
compact widths (`selector_width_buckets`, default `[5120]`) plus the full width
as fallback. Decode graphs are keyed by `(batch_size, selector_width)`; a live
sequence replays into the smallest width that covers it. Selection then scores
only that width's columns, recovering the decode cost of full-width selection.

Cross-TP the per-rank scores are all-reduced (custom all-reduce with a pinned
algorithm for deterministic summation order) so every rank's top-K agrees.

## File map

| File | Role |
|------|------|
| `config.py` | `DoubleSparsityConfig` parsing + validation (data model) |
| `channel_mask.py` | Load / verify / per-rank slice the calibrated mask |
| `selector.py` | Bound runtime state + `bind_runtime_data`; eager reference path |
| `selection_kernel.py` | `retrieve_topk_graph_safe`: score → reduce → current-slot → radix top-k |
| `absorbed_latent_kernel.py` | Paged fp8 absorbed-latent score Triton kernel |
| `topk_kernel.py` | Sequence-order deterministic radix top-k |
| `cuda_graph.py` | `DSGraphState` scratch + `allocate_graph_state` |
| `page_table_adapter.py` | Logical → physical KV-slot gather (`logical_to_physical`) |
| `validator.py` | Startup fail-closed gates (`validate_double_sparsity`) |
| `calibrate.py` | Offline channel-mask calibration tool |
| `metrics.py` | `channel_mask_valid` Prometheus gauge |

Production selection runs through `deepseek_v2._select_topk_indices` →
`selection_kernel.retrieve_topk_graph_safe`; the eager `selector.retrieve_topk`
and the CPU score kernels in `absorbed_latent.py` are unit-test references.
