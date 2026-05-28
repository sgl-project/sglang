# Round 5 Summary

## Work Completed

### AC-2 lifetime/stale-slot — `TestAC2Lifetime`

Added four tests that verify the token label table's lifetime and slot budget invariants:

**`test_boot_log_emits_gb_per_rank`**
Calls `allocate_token_label_table` inside `assertLogs` and verifies that the INFO-level message contains `token_label_table:`, `GB/rank`, and all four dimension fields (`L=`, `T=`, `H=`, `D=`). This confirms operators can read the sizing audit from boot logs.

**`test_slot_budget_covers_all_physical_kv_slots`**
Creates a table with `max_tokens = kv_pool_size + page_size` (= 192) and calls `token_label_write` targeting the last valid physical slot (`max_tokens - 1 = 191`). Asserts `written[0, 191]` is True and `table.max_tokens == 192`. Verifies the table covers the full `out_cache_loc` address space without OOB.

**`test_stale_slot_overwrite_replaces_prior_label`**
Writes label A (all-1.0) to slot 7, confirms it reads 1.0, then writes label B (all-2.0) to the same slot, and asserts the slot now reads 2.0 with no trace of 1.0. Verifies `token_label_write` performs an unconditional overwrite rather than accumulating.

**`test_label_visible_immediately_after_write`**
Writes sentinel value 42.0 to slot 3 and immediately reads it back. Asserts all label values equal 42.0 and `written[0, 3]` is True. Confirms no phantom/stale state is visible before the next overwrite.

### AC-3 range mask — `TestAC3RangeMask`

Added two tests verifying per-request token range ownership:

**`test_multi_request_picks_within_own_range_with_mask`** (positive)
Constructs bs=2, max_tokens=20 with disjoint ownership: req-0 owns slots 0..9, req-1 owns slots 10..19. Sets signatures so req-1's slots would outscore req-0's for any query without a mask. Passes `per_request_valid` (bool [2, 20]) to `retrieve_topk_via_labels`. Asserts all indices for req-0 are < 10 and all for req-1 are >= 10. Both requests produce at least one valid pick.

**`test_without_mask_cross_request_contamination_occurs`** (negative — mask is load-bearing)
Same setup but passes `per_request_valid=None`. Slots 10..19 have score 1000, slots 0..9 have score 0. With all-ones queries, req-0's top picks come from the high-score region 10..19 (which belongs to req-1). Asserts at least one index in req-0's results is >= 10. This confirms that `per_request_valid` is the mechanism preventing cross-request contamination and cannot be removed.

## Files Changed

- `test/registered/unit/layers/attention/test_double_sparsity_unit.py` — add `TestAC2Lifetime` (4 tests) and `TestAC3RangeMask` (2 tests)

## Validation

```
PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q
166 passed, 0 failed
```

## Remaining Items

- `task-ac1-hwtest`: hardware population test on H200 — pending hardware access.
- `task-ac7-bypass` (AC-7): short-seq MHA bypass; confirm save_kv_cache=True still fires.
- `task-ac4-calibrate` (AC-4): Method 1 Q+K joint hooks in calibrate.py.
- `task-ac5-tp` (AC-5): TP=2 multiprocess test.
- `task-ac6-cuda-graph` (AC-6): decode-path graph capture.

## BitLesson Delta

Action: none
Lesson ID(s): NONE
Notes: AC-2 tests use standard Python assertLogs and direct tensor writes; AC-3 tests use the existing per_request_valid parameter of retrieve_topk_via_labels. No project-specific new pattern emerged.
