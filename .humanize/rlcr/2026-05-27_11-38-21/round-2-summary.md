# Round 2 Summary

## Work Completed

### AC-0 slot-count authority fix

`TokenLabelTable.max_tokens` was incorrectly bound from `req_to_token_pool.size`
(request-row count, ~256). Any `out_cache_loc` value beyond that count would produce
an out-of-bounds write. Fixed by deriving `max_tokens = kv_pool.size + kv_pool.page_size`
where `kv_pool` is `token_to_kv_pool` — the object whose first tensor axis exactly
spans the physical KV slot address space addressed by `out_cache_loc`.

The physical pool is published to `server_args._ds_token_to_kv_pool` in the
`model_runner.py` DS bind block so it is available when `finalize_double_sparsity_bind()`
runs. A `RuntimeError` guard ensures the bind fails loudly if called before pool init.

Three regression tests confirm: (1) non-contiguous large-slot writes succeed with
correct sizing, (2) the old small sizing raises `IndexError`, (3) logical→physical
round-trip via `req_to_token` is bit-exact.

### AC-1 hook wiring

`_write_token_labels(layer, cache_loc, k)` added to `NativeSparseAttnBackend`:
- Projects `k_latent [T, kv_lora_rank]` through `layer.kv_b_proj` (no_grad) to
  get full `kv_proj_out [T, H_local*(nope+v)]`, slices the noPE K columns and
  reshapes to `[T, H_local, 128]`.
- Calls `token_label_write(signatures, written, layer_id, cache_loc, k_nope,
  channel_selection_layer)`.
- Guard: no-op if `enable_double_sparsity=False`, table/channel_selection is None,
  or `layer.kv_b_proj` is absent.

The hook is wired at all three `set_mla_kv_buffer` sites:
- Site 1: `forward_extend` native path
- Site 2: `forward_decode` native path
- Site 3: TRT-LLM extend path

`channel_selection` and `qk_nope_head_dim` are published from
`_bind_double_sparsity_runtime_data` via `server_args` attributes so the backend
can capture them at construction time without traversing the model hierarchy.

Two unit tests confirm: hook populates `signatures` and sets `written`, and is a
no-op when DS is disabled.

## Files Changed

- `python/sglang/srt/model_executor/model_runner.py` — publish `_ds_token_to_kv_pool`
- `python/sglang/srt/models/deepseek_v2.py` — use `kv_pool.size + kv_pool.page_size` for `max_tokens`; publish `_ds_channel_selection`, `_ds_qk_nope_head_dim`, `_double_sparsity_token_label_table`
- `python/sglang/srt/layers/attention/double_sparsity/token_label_table.py` — fix docstrings
- `python/sglang/srt/layers/attention/dsa_backend.py` — capture table+channel selection in `__init__`; add `_write_token_labels`; wire at 3 sites
- `test/registered/unit/layers/attention/test_double_sparsity_unit.py` — add `TestAC0RealSlotRegression` (3 tests) and `TestAC1HookUnit` (2 tests)

## Validation

```
PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q
155 passed, 0 failed
```

Commit: `65cbd28e0`

## Remaining Items

Next round candidates (from queued side issues and active task table):
- AC-2: boot-time GB/rank log; stale-slot lifetime test
- AC-3: per-request range mask (M2); multi-request boundary test
- AC-7: short-seq MHA bypass
- AC-4: V3.2 channel mask calibration
- AC-6 graph helper: `capture_decode_step` still calls `retrieve_topk` without logical→physical conversion (blocks AC-6)
- AC-8 observability: `_publish_ds_request_summary` reports page-named fields (blocks AC-8 quality metrics)

## BitLesson Delta

Action: none
Lesson ID(s): NONE
Notes: Implementation corrections (slot-count sizing and hook wiring). No new generalizable pattern emerged.
