# Reviewer Guide — Standalone Double Sparsity

A short tour for an upstream reviewer. The PR description lives at
`development/PR_DESCRIPTION.md`; this file is the where-to-look companion.

## Load-bearing files

| File | What's in it |
|------|--------------|
| `python/sglang/srt/server_args.py` | New flags `--enable-double-sparsity` and `--double-sparsity-config`; `validate_double_sparsity(self)` plugged into `check_server_args` immediately after `validate_hisparse(self)`. |
| `python/sglang/srt/layers/attention/double_sparsity/__init__.py` | Re-exports the package surface. |
| `.../double_sparsity/config.py` | `DoubleSparsityConfig` dataclass + strict JSON parser. Allowed fields: `top_k`, `page_size`, `channel_mask_path`, `device_buffer_size`, `extra`. Rejects `selection_mode` / `top_p` (these belong to the deferred Twilight ABI). |
| `.../double_sparsity/selector.py` | `DoubleSparsitySelector.retrieve_topk(queries, layer_id, req_pool_indices, sparse_mask, seq_lens, hot_pages)`. Placeholder by default; flips to real mode after `bind_runtime_data(page_signature_table, channel_mask)`. Module-level placeholder guard refuses production traffic with placeholder selector. |
| `.../double_sparsity/channel_mask.py` | `safetensors` loader + content-hash recompute + startup NIAH-min sanity probe. Schema: `channel_selection[L, H, label_dim]` int32 + `channel_weights[L, H, label_dim]` fp32 + metadata. |
| `.../double_sparsity/page_signature_table.py` | Allocator-owned GPU table + lifecycle hooks (assign / free / evict / retract) + `valid_mask` invalidation + hot-page bookkeeping. |
| `.../double_sparsity/selection_kernel.py` | `compute_page_scores` (signature → max-over-heads scalar score), `all_reduce_page_scores` (DEC-9 SUM across attention TP group), `select_topk_sequence_order` (top-K + ascending sort + `-1` padding). |
| `.../double_sparsity/validator.py` | Full startup validation: HiSparse mutex (DEC-8), no PD-disagg, channel-mask file existence + content-hash, schema/dtype/page-size pairing, DEC-3 backend/dtype pairing, capability check (DEC-10), DEC-2 radix-cache gate. |
| `.../double_sparsity/metrics.py` | `sglang_double_sparsity_*` Prometheus surface + `meta_info_for_request` helper. |
| `.../double_sparsity/calibrate.py` | Offline calibration CLI. Default dataset is NIAH-shaped synthetic; emits `content_sha256` into the file metadata. |
| `python/sglang/srt/models/deepseek_v2.py` | `DeepseekV2AttentionMLA.__init__` constructs `self.double_sparsity_selector` inside the existing `if self.use_nsa:` block. The single config-gated branch lives in `_select_topk_indices` (helper method on `DeepseekV2AttentionMLA`). |
| `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` | Two `self.indexer(...)` call sites in `forward_absorb_prepare` (the V3.2 absorb-mode prepare mixin) now call `self._select_topk_indices(...)`. |
| `development/benchmark.sh`, `benchmark_baseline.sh`, `benchmark_compare.py`, `serve_native_nsa.sh`, `serve_double_sparsity.sh` | Two-column comparator scaffolding + paired server-launch references for the 2-node H200 operating point (DEC-1). |
| `development/pre-commit-blocklist.sh` | Hook that blocks `HANDOFF*.md`, `SESSION_REPORT*.md`, pensieve installs, humanize state from the git index. |
| `test/registered/unit/layers/attention/test_double_sparsity_unit.py` | Unit-test surface (config parser, selector ABI, validator, placeholder guard, hook branch, channel mask loader, page signature lifecycle, selection kernel, metrics). |
| `test/manual/test_double_sparsity_v32.py` | NIAH @ 4K/16K/64K + MMLU 5-shot scaffolding; skips when weights unavailable. |

## Path-whitelist amendment

The refined plan's AC-12 whitelist did not include `python/sglang/srt/models/deepseek_common/`. The V3.2 attention path is split across `deepseek_v2.py` (class definition) and the absorb-mode prepare mixin in `deepseek_common/attention_forward_methods/forward_mla.py`. Any DS hook into the per-step indexer call site touches the mixin. The PR keeps the touch surgical: two line replacements at the two existing `self.indexer(...)` call sites; no new files in the mixin directory.

## V3.2-only scope today; capability seam for GLM-5.1

The validator uses `is_deepseek_nsa(hf_config)` as a proxy for "exposes the NSA `Indexer` hook surface" (DEC-10). Today this matches DeepSeek-V3.2 only; the capability check is the seam at which GLM-5.1 (DEC-6 deferred-but-hard) will fall in once it ships the same `nsa.Indexer` interface. The selector ABI, channel-mask schema, and metric names are all dtype- and length-agnostic so 128K ISL and FP4 weights also fall in without rewrite.

## What I'd review first

1. **Server-args wiring**: confirm `validate_double_sparsity(self)` is called from `check_server_args` after `validate_hisparse(self)`, and that the two argparse entries do not collide with anything (`--double-sparsity-config` is a new top-level flag).
2. **Hook site**: confirm the single config-gated branch lives in `DeepseekV2AttentionMLA._select_topk_indices` and that the two call sites in `forward_absorb_prepare` (forward_mla.py:242, 261) both delegate to it. Confirm the indexer is otherwise unchanged.
3. **TP rank synchronization**: read `compute_page_scores` + `all_reduce_page_scores`. Confirm that nothing all-gathers the page signatures themselves (per DEC-9); only the scalar per-page scores are reduced.
4. **Placeholder guard**: `assert_real_selector_or_placeholder_allowed` refuses production traffic when the selector is placeholder-built. `SGLANG_DS_ALLOW_PLACEHOLDER=1` exists only for unit/smoke tests.
5. **Channel mask file**: read `load_channel_mask` — schema validation, content-hash recompute, and `validate_against_runtime` shape pairing.
6. **Validator paths**: read `validate_double_sparsity` top-to-bottom. Each branch corresponds to one acceptance criterion or one decision record.

## Known gaps for the integration that the deploying team must close

* End-to-end AC-8 SLO verification: requires a calibrated channel mask file + V3.2 FP8 weights on the deploying hardware.
* AC-9 quality runs (NIAH / MMLU): same.
* AC-6 conc 16/32/64 CUDA-graph capture verification: requires real V3.2 boot.
* DEC-2 radix-cache permission: the validator gates radix cache until `_double_sparsity_radix_fixture_passed` is recorded as True (set by the M3-B page-stability fixture, which the deploying team runs alongside calibration).

## meta_info integration (Round 2 wiring point)

The DS stats helper lives at:

```python
from sglang.srt.layers.attention.double_sparsity.metrics import (
    DoubleSparsityRequestStats, customized_info_for_request,
)
```

The least-invasive surface point is `tokenizer_manager.py`'s existing `customized_info` hook (around line 1739). The DS path in the scheduler should:

```python
# scheduler side (per-request, at the point where ds stats become known):
stats = DoubleSparsityRequestStats(
    sparsity_rate=selected_pages / total_valid_pages,
    selected_pages=selected_pages,
    dense_fallback=0,
)
recv_obj.customized_info.setdefault(rid, {}).update({
    "double_sparsity": customized_info_for_request(stats),
})
```

This avoids modifying the central `meta_info` constructor; the existing tokenizer-manager loop will auto-surface the `"double_sparsity"` key inside the request's `meta_info` payload. This wiring is the deploying team's integration glue — the helper API + the docstring locator make the change one paragraph long.

## Round 2 perf path (Triton kernels)

`compute_page_scores` and `page_signature_write` both ship Triton kernels that auto-select on CUDA + Triton-available, with torch-reference fallback on CPU / non-CUDA / when Triton is absent. Numerical equivalence is asserted in unit tests:

* `compute_page_scores_triton` vs torch reference: max diff ~2e-6 on synthetic inputs.
* `page_signature_write_triton` vs torch reference: max diff ~2e-3 in fp16 on synthetic inputs.

NSA cross-validation (`test_quant_kcache_roundtrip`) drives `nsa.quant_k_cache.quantize_k_cache_separate` end-to-end through `dequant_nope_fp8_to_bf16` and asserts ~2.6% relative L2 error per tile — i.e. the FP8 byte-layout contract DS depends on is verified against NSA's source-of-truth quantizer.
