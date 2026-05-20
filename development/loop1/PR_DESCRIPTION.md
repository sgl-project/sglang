# Standalone Double Sparsity for DeepSeek-V3.2 (FP8)

## Summary

This PR adds a standalone Double Sparsity (DS) path for DeepSeek-V3.2 (FP8) under a new top-level package `python/sglang/srt/layers/attention/double_sparsity/`, gated by two new server args:

```
--enable-double-sparsity
--double-sparsity-config '{"top_k": 2048, "page_size": 64,
                           "channel_mask_path": "/path/to/channel_mask.safetensors",
                           "device_buffer_size": 4096}'
```

The DS selector replaces NSA's per-step token-selection role on the V3.2 attention path with an offline-calibrated channel-importance projection plus a runtime page-signature top-K. Everything else stays untouched: paged FlashMLA, FP8 KV cache, NSA's quant / dequant kernels, and the model code outside the indexer call site.

**Explicitly NOT a HiSparse algorithm.** DS does not register with `_ALGORITHM_REGISTRY`, does not require `--enable-hisparse`, and does not require PD-disaggregation. The two features are mutually exclusive at runtime (startup error if both are enabled) but both ship in the codebase.

## Architecture

### Two artifacts

* **Channel mask file** — offline `safetensors` produced by `python -m sglang.srt.layers.attention.double_sparsity.calibrate`. Ships once per model revision. Contains per-(layer, head) channel selection + weights + a `content_sha256` over the payload. Validated at server startup.
* **Page signature table** — GPU tensor, allocator-owned metadata sitting next to the KV page table. Per-rank shape `[num_layers_local, max_pages, num_heads_local, label_dim]` at `fp16`. Populated by a scale-aware FP8 dequant + channel projection on every page assignment, invalidated on free / eviction / abort. ~480 MB / rank at 1 M context, page=64, TP=8.

### Single edit site

`DeepseekV2AttentionMLA.__init__` constructs `self.double_sparsity_selector` when DS is enabled. A new helper method `DeepseekV2AttentionMLA._select_topk_indices` contains the single config-gated branch:

```python
if self.use_double_sparsity:
    return self.double_sparsity_selector.retrieve_topk(...)
return self.indexer(...)
```

The two existing `self.indexer(...)` call sites in `forward_absorb_prepare` (the V3.2 MLA absorb prepare mixin in `models/deepseek_common/attention_forward_methods/forward_mla.py`) delegate to the helper. No monkey-patching, no model class fork, no new attention backend.

### TP rank synchronization

Per the DEC-9 implementation contract: page signatures stay TP/head-sharded. Each rank computes scalar page scores from its local head shard; an `all_reduce(SUM)` on the `[max_pages]`-shaped score tensor produces shared scores; each rank runs deterministic top-K independently → bit-equal `selected_indices` by construction. No signature all-gather.

### Hot-page rule

The active in-fill page (and a configurable local window of N most-recent pages, default 1) is forced into the selected set regardless of score, addressing the "fresh tokens invisible to selection" problem (CMT-14).

## Acceptance Criteria Status

| AC | Status | Notes |
|----|--------|-------|
| AC-1 (server args) | ✅ | `--enable-double-sparsity` + `--double-sparsity-config`; DEC-8 mutual-exclusion; missing-config startup error |
| AC-2 (hook + selector ABI) | ✅ scaffold; ⚠ depth | Hook site + helper + selector ABI shape (`int32[bs, max_top_k]`, `int32[bs]`); end-to-end FP8 attention output tolerance requires real channel mask |
| AC-3 (page size + backend/dtype) | ✅ | Unsupported page-size rejected; `fp8_e4m3 ↔ flashmla_kv`, `bfloat16 ↔ flashmla_sparse` |
| AC-4 (channel mask file) | ✅ | safetensors loader + content_sha256 + NIAH-min sanity probe (probe is inconclusive with placeholder selector) |
| AC-5 (calibration) | ✅ | `python -m ...double_sparsity.calibrate` CLI; synthetic NIAH default; CI tiny-fixture path |
| AC-6 (CUDA graph) | ⚠ scaffold | Static buffers + preallocated scratch + device-side branching; capture-time test infra outlined; full conc 16/32/64 capture verification requires weights |
| AC-7 (baseline comparator) | ✅ scaffold | `benchmark_baseline.sh` + `benchmark.sh` two-column scaffolding; `benchmark_compare.py` enforces match on {GPU id, TP size, page size, radix-cache, concurrency} |
| AC-8 (SLO 30 tok/s P50 / 22s P99 TTFT) | ⚠ untested | Hardware: 2-node H200 (DEC-1); requires calibrated channel mask + real V3.2 server boot |
| AC-9 (NIAH ≤ 5 pp / MMLU ≤ 1.0 pp) | ⚠ untested | Test scaffold lives at `test/manual/test_double_sparsity_v32.py`; needs weights |
| AC-10 (observability) | ✅ | `sglang_double_sparsity_*` Prometheus surface + `meta_info` helper (`sparsity_rate`, `selected_pages`, `dense_fallback`) |
| AC-11 (ABI lock-in) | ✅ | `retrieve_topk(queries, layer_id, req_pool_indices, sparse_mask, seq_lens, hot_pages) -> (selected_indices, valid_lengths)`; `selection_mode` / `top_p` rejected at parse time |
| AC-12 (upstream hygiene) | ✅ | `development/pre-commit-blocklist.sh` blocks `HANDOFF*.md`, `SESSION_REPORT*.md`, pensieve installs, humanize state |

## Deferred (per refined plan)

* **GLM-5.1 support** — DEC-6 deferred-but-hard. Schema is shaped to admit it; the capability-check validator (`is_deepseek_nsa(hf_config)` proxy) will be extended once GLM-5.1 ships the same indexer interface.
* **PD-Disagg / HiCache integration** — DEC-6 "kept in mind" downstream; no customer ask on DS yet.
* **128K ISL** — DEC-6 deferred; schema has no length-dependent fields.
* **FP4 (nvfp4 / mxfp4) weights** — DEC-6 deferred; schema is dtype-agnostic.
* **Twilight / top-p selection** — DEC-6 / CMT-10 / AC-11: dropped from initial scope. The selector ships single-mode top-K. A future top-p ABI is its own plan.
* **HiSparse algorithm registration** — DEC-6 + DEC-8: OUT of all scope, initial AND downstream. The DS validator and the HiSparse validator are independently invoked from `check_server_args`; they refuse to coexist at runtime.

## Testing

Unit tests at `test/registered/unit/layers/attention/test_double_sparsity_unit.py`:
* Config parser: minimum-required fields, extra-dict, `selection_mode`/`top_p` rejection, missing `channel_mask_path`, invalid JSON, invalid `top_k`.
* Selector ABI: shapes + dtypes, sequence-ascending invariant, `valid_lengths` clipping.
* Validator: disabled no-op, HiSparse mutual-exclusion, missing config, disaggregation rejection, page-size mismatch, unsupported page size, backend/dtype mismatch, file-missing, radix-gate without fixture, dev override.
* Placeholder guard: refuses without env var, allowed with env var, real selector passes.
* Hook branch dispatch: DS tuple return, indexer fallback, env-guard refusal.
* Channel mask loader: save / load round-trip, content-hash mismatch, schema-version mismatch, dtype not in supported set.
* Page signature table: lifecycle (assign / free / evict / retract), hot-page tracking, memory-budget estimator.
* Selection kernel: project_query_onto_channels, compute_page_scores (invalid-page handling), select_topk_sequence_order (ascending + hot-page rule).
* Metrics: meta_info_for_request shape; counter increments observable.

Manual tests at `test/manual/test_double_sparsity_v32.py` for NIAH (4K / 16K / 64K) and MMLU 5-shot; skip cleanly when weights or a calibrated channel mask file are unavailable.

## Reviewer Notes

See `development/loop1/REVIEWER_GUIDE.md` for a tour of the load-bearing files, the AC-12 path-whitelist amendment, and the V3.2-only scope per DEC-10's capability check seam.
