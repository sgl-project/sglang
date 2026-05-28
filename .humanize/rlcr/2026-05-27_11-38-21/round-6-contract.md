# Round 6 Contract

## Mainline Objective

Close AC-2 and AC-3 properly by (a) implementing the stale-slot invalidation mechanism that prevents the selector from picking a reused slot before its new label is written, (b) adding a bind-time size guard, and (c) replacing the physical-domain-only AC-3 test with a logical-domain `req_to_token` production-path test.

## Target ACs

- **AC-2**: Add `invalidate_token_label_slots` in `token_label_write.py`; call it in `_select_topk_indices` before `retrieve_topk`; add `validate_table_covers_kv_pool` guard in `deepseek_v2.py`; add regression tests proving the stale-slot invariant.
- **AC-3**: Add logical-domain production-path test using `req_to_token` + `seq_lens` + `logical_to_physical`; negative fixture drops `req_to_token` (physical-domain) and confirms contamination (already in R5 test).

## Tasks

### task-ac2-invalidate (coding, claude)
Add `invalidate_token_label_slots(written, layer_id, cache_loc)` to `token_label_write.py`.
In `deepseek_v2.py:_select_topk_indices._run()`, call it on `forward_batch.out_cache_loc` before `selector.retrieve_topk`.
Add `validate_table_covers_kv_pool(table, kv_pool_size, page_size)` to `token_label_table.py` and call it in the reuse branch of `_bind_double_sparsity_runtime_data`.

### task-ac2-invalidate-test (coding, claude)
In `TestAC2Lifetime`:
- `test_invalidate_makes_stale_slot_unselectable`: write A to slot 7 via `req_to_token`, score WITHOUT invalidation to confirm it IS selectable (documents the bug), then invalidate and confirm it is NOT selectable.
- `test_after_invalidation_new_write_restores_selectability`: invalidate slot 7, then write B, confirm slot IS selectable.
- `test_validate_table_size_rejects_wrong_max_tokens`: `validate_table_covers_kv_pool` raises when `max_tokens != kv_pool_size + page_size`.

### task-ac3-logical-domain-test (coding, claude)
In `TestAC3RangeMask`:
- `test_logical_domain_req_to_token_isolates_per_request` (positive): bs=2, req-0 req_to_token maps logical 0..9 → physical 0..9, req-1 maps 0..9 → physical 10..19; set signatures so physical 10..19 outscore 0..9 for all queries; run in logical-domain mode; convert via `logical_to_physical`; assert req-0 physical output ⊆ [0,10) and req-1 physical output ⊆ [10,20).
- Negative is already covered by `test_without_mask_cross_request_contamination_occurs` (R5).

## Blocking Issues

None. All work is unit-testable without hardware.

## Success Criteria

1. `PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q` → ≥ 170 passed, 0 failed.
2. `invalidate_token_label_slots` exists in `token_label_write.py`.
3. `_select_topk_indices._run()` calls invalidation before `retrieve_topk`.
4. `validate_table_covers_kv_pool` exists in `token_label_table.py`; called in reuse branch of bind.
5. `TestAC2Lifetime` has stale-slot before/after invalidation tests plus size-guard test.
6. `TestAC3RangeMask` has logical-domain `req_to_token` production-path test.
