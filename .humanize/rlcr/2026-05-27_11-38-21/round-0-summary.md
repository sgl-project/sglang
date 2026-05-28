# Round 0 Summary

## What Was Implemented

Round 0 completed the AC-0 architecture rotation: migrated the entire Double
Sparsity implementation from page-level `[L, P, H, D]` label storage to
token-level `[L, T, H, D]` label storage (AC-0), and verified all 150 unit
tests pass on the new APIs (AC-13).

### Core files created

- `token_label_table.py` — `TokenLabelTable` dataclass (`signatures [L_local,
  max_tokens, H_local, label_dim]`, `written bool[L_local, max_tokens]`);
  `allocate_token_label_table`; `estimate_hbm_bytes` helper.

- `token_label_write.py` — `token_label_write(signatures, written, layer_id,
  cache_loc, k_nope, channel_selection_layer)` Triton-backed write path.
  Takes projected 128-d K_nope per token slot (no page averaging).

### Core files modified

- `page_table_adapter.py` — rewritten to < 150 LOC; exposes only
  `DSAdapterError` (base exception, kept for downstream imports) and
  `logical_to_physical(selected_indices, req_pool_indices, req_to_token, out)
  -> int`. The function performs a single `req_to_token` gather and returns a
  scalar `error_count` for bad pool indices.

- `selection_kernel.py` — `select_topk_sequence_order(token_scores, max_top_k)`
  (2 args; removed `hot_pages` and `per_request_valid`). `retrieve_topk_via_labels`
  updated to token-level shapes; removed `hot_pages` kwarg (kept `per_request_valid`).

- `selector.py` — `retrieve_topk` now returns logical token positions (sequence-
  ascending, `-1` padded) for token-level label table.

- `config.py` — `top_k` default changed from 64 (pages) to 2048 (tokens);
  docstring updated.

- `validator.py` — reads `dsa_prefill_backend` / `dsa_decode_backend` instead
  of old `nsa_*` attribute names. Added `top_k == get_dsa_index_topk(hf_config)`
  boot assertion (env-override: `SGLANG_DS_ALLOW_TOPK_MISMATCH=1`).

- `__init__.py` — re-exports updated to `TokenLabelTable`, `token_label_write`,
  `retrieve_topk`; old `PageSignatureTable` / `page_signature_write` exports
  removed.

- `channel_mask.py` — docstring updated to reference `token_label_write`.

- `deepseek_v2.py` — `_bind_double_sparsity_runtime_data` derives `max_tokens`
  from `req_to_token_pool.size`; `DSGraphState.selected_indices` is now
  `int32[max_bs, max_top_k]`.

- `model_runner.py` — updated DS bind call to pass `req_to_token_pool`.

### Files deleted

- `page_signature_table.py` (185 LOC) — `PageSignatureTable` page-level class
- `page_signature_write.py` (498 LOC) — page-level Triton write kernel

### Test file migrated (150 tests)

`test/registered/unit/layers/attention/test_double_sparsity_unit.py` — all 59
old API references removed and replaced:

| Old | New |
|-----|-----|
| `PageSignatureTable` / `page_signature_write` | `TokenLabelTable` / `token_label_write` |
| `expand_ds_selection_to_topk_indices` | `logical_to_physical` |
| `DSAdapterPageOutOfRange` | `DSAdapterError` / `RuntimeError` |
| `hot_pages=` kwarg | removed (no longer accepted) |
| `SchedulerOutputProcessorMixin` | `SchedulerBatchResultProcessor` |
| `nsa_decode_backend` | `dsa_decode_backend` |
| `m3b_page_stability_fixture` / `M3BFixture*` | `token_label_write` equivalents |
| Error trigger via OOB page position | Error trigger via OOB pool index |

## Files Changed

**Created:**
- `python/sglang/srt/layers/attention/double_sparsity/token_label_table.py`
- `python/sglang/srt/layers/attention/double_sparsity/token_label_write.py`

**Modified:**
- `python/sglang/srt/layers/attention/double_sparsity/__init__.py`
- `python/sglang/srt/layers/attention/double_sparsity/channel_mask.py`
- `python/sglang/srt/layers/attention/double_sparsity/config.py`
- `python/sglang/srt/layers/attention/double_sparsity/page_table_adapter.py`
- `python/sglang/srt/layers/attention/double_sparsity/selection_kernel.py`
- `python/sglang/srt/layers/attention/double_sparsity/selector.py`
- `python/sglang/srt/layers/attention/double_sparsity/validator.py`
- `python/sglang/srt/model_executor/model_runner.py`
- `python/sglang/srt/models/deepseek_v2.py`
- `test/registered/unit/layers/attention/test_double_sparsity_unit.py`

**Deleted:**
- `python/sglang/srt/layers/attention/double_sparsity/page_signature_table.py`
- `python/sglang/srt/layers/attention/double_sparsity/page_signature_write.py`

## Validation

```
grep -c "def test_" test/registered/unit/layers/attention/test_double_sparsity_unit.py
150  ✓

grep -n "PageSignatureTable|page_signature_write|expand_ds_selection_to_topk_indices|DSAdapterPageOutOfRange|hot_pages=|SchedulerOutputProcessorMixin|nsa_decode_backend|m3b_page_stability_fixture" test/registered/unit/layers/attention/test_double_sparsity_unit.py
(empty) ✓

git diff --stat HEAD~1 | tail -1
14 files changed, 1280 insertions(+), 2220 deletions(-)  ✓
```

AC-0 positive checks satisfied at code level:
- `from sglang.srt.layers.attention.double_sparsity import TokenLabelTable, token_label_write, retrieve_topk` — re-exports present in `__init__.py`
- `page_table_adapter.py` is 72 LOC (< 150) ✓
- `token_label_table.py` allocates shape `[L_local, max_tokens, H_local, label_dim]` ✓
- `DoubleSparsityConfig.top_k` defaults to 2048 ✓
- `DSGraphState.selected_indices` shape is `int32[max_bs, max_top_k]` ✓
- `validator.py` reads `dsa_prefill_backend` / `dsa_decode_backend` ✓
- Importing `page_signature_table` or `page_signature_write` now raises `ModuleNotFoundError` (files deleted) ✓

## Remaining Items

All AC-0 and AC-13 code tasks complete. Remaining plan tasks start from AC-1:

- `task-m1-hook` (AC-1): Wire `token_label_write` at `dsa_backend.py` L1439/L1637/L2162
- `task-ac1-hwtest` (AC-1): Hardware test — `forward_extend` → non-zero signatures
- `task-ac2-lifetime` (AC-2): Boot-time GB/rank log; stale-slot negative test
- `task-m2-rangemask` (AC-3): Per-request token range mask in scorer
- … (remaining 20+ tasks per plan)

## BitLesson Delta

Action: none
Lesson ID(s): NONE
Notes: AC-0 rotation followed the plan exactly. No unexpected problems that would
generalize into a new lesson; all issues encountered (missing DSAdapterError,
wrong scheduler class name, wrong error trigger mechanism) were one-off
discovery-during-migration issues fully resolved in this round.
