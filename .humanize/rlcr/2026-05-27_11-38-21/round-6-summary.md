# Round 6 Summary

## Work Completed

### AC-2 stale-slot protection — production code

**`invalidate_token_label_slots` in `token_label_write.py`**
New function `invalidate_token_label_slots(written, layer_id, cache_loc)` sets `written[layer_id, cache_loc] = False` via `index_fill_`. This is capture-safe and O(1). Called by `_select_topk_indices` in `deepseek_v2.py` before `retrieve_topk` so that a newly-allocated physical KV slot cannot be selected based on a stale label left by a previously-evicted request.

**Wire in `deepseek_v2.py:_select_topk_indices._run()`**
Before `selector.retrieve_topk` is called, the `_run()` closure now retrieves `forward_batch.out_cache_loc` and `selector.token_label_table`, then calls `invalidate_token_label_slots`. Both guards are None-safe so the call is a no-op in unit tests that use placeholder selectors or SimpleNamespace forward batches.

**`validate_table_covers_kv_pool` in `token_label_table.py`**
New function `validate_table_covers_kv_pool(table, kv_pool_size, page_size)` raises `ValueError` if `table.max_tokens != kv_pool_size + page_size`. Called in the reuse branch of `_bind_double_sparsity_runtime_data` (the `else:` block that fires for layers 2+ when the table was already created by layer 0). Guards against a mis-sized pre-existing table reaching production.

### AC-2 regression tests in `TestAC2Lifetime`

**`test_invalidate_makes_stale_slot_unselectable`**
Writes a high-score stale label (1000.0) at physical slot 7, creates a new request with `req_to_token[0, 0] = 7` (logical pos 0 → slot 7). Runs `retrieve_topk_via_labels` in logical-domain mode WITHOUT invalidation → confirms stale slot IS selectable (shows the bug). Then calls `invalidate_token_label_slots` → runs again → confirms slot is NOT selectable.

**`test_after_invalidation_new_write_restores_selectability`**
Invalidates slot 7, then writes a new label. Confirms `written[0, 7]` is restored to True and slot is selectable again. Verifies the invalidation → write lifecycle.

**`test_validate_table_size_rejects_wrong_max_tokens`**
Creates a table with `max_tokens=100`. Calls `validate_table_covers_kv_pool(table, 36, 64)` → passes (36+64=100). Calls with `(64, 64)` → raises ValueError mentioning `max_tokens=100` and `128`.

### AC-3 production-path test in `TestAC3RangeMask`

**`test_logical_domain_req_to_token_isolates_per_request`**
Uses `req_to_token[0, 0..9] = [0..9]` and `req_to_token[1, 0..9] = [10..19]`. Sets signatures so physical slots 10..19 have score 1000 and 0..9 have score 0. Runs `retrieve_topk_via_labels` in logical-domain mode (with `req_pool_indices`, `req_to_token`, `seq_lens`). Converts logical output to physical via `logical_to_physical`. Asserts req-0 physical slots ⊆ [0,10) and req-1 physical slots ⊆ [10,20). This is the production path: logical-domain mode scores only each request's own `req_to_token` row, so foreign high-score slots are invisible to req-0.

## Files Changed

- `python/sglang/srt/layers/attention/double_sparsity/token_label_write.py` — add `invalidate_token_label_slots`
- `python/sglang/srt/layers/attention/double_sparsity/token_label_table.py` — add `validate_table_covers_kv_pool`
- `python/sglang/srt/models/deepseek_v2.py` — invalidation call in `_select_topk_indices._run()`; size guard in bind reuse branch
- `test/registered/unit/layers/attention/test_double_sparsity_unit.py` — 4 new tests in `TestAC2Lifetime`; 1 new test in `TestAC3RangeMask`

## Validation

```
PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q
170 passed, 0 failed
```

Commit: `178427b75`

## Remaining Items

- `task-ac1-hwtest`: H200 hardware population test — pending hardware access.
- `task-ac7-bypass` (AC-7): short-seq MHA bypass implementation + tests.
- `task-ac4-calibrate` (AC-4): Method 1 Q+K joint hooks in calibrate.py.
- `task-ac5-tp` (AC-5): TP=2 multiprocess all-reduce test.
- `task-ac6-cuda-graph` (AC-6): decode-path graph capture.

## BitLesson Delta

Action: none
Lesson ID(s): NONE
Notes: Stale-slot invalidation is a standard pre-selection invalidation pattern (mark invalid before selection, restore valid after write). No new generalizable project-specific lesson.
