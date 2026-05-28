# Round 5 Review Result

Mainline Progress Verdict: ADVANCED

Round 5 added useful unit coverage and the local unit suite is green:

```text
PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q
166 passed, 24 warnings in 11.46s
```

However, AC-2 and AC-3 are not closed. The new tests verify boot-log text, direct overwrites, and a physical-domain `per_request_valid` helper mask, but they do not prove the allocator-lifetime stale-slot invariant or the production `req_to_token` ownership path required by the original plan.

## Mainline Gaps

1. **AC-2 stale-slot protection is missing; a reused slot can be selected before the new label write.**

   Evidence:
   - The new stale-slot test only writes A and then immediately writes B to the same slot (`test_double_sparsity_unit.py:5020`). It never simulates free/reallocation followed by selector read before B is written, which is the plan's negative fixture.
   - `token_label_write.py:53-54` sets `written[layer_id, slot] = True` and there is no corresponding lifecycle invalidation on slot reuse.
   - `_compute_logical_token_scores` gathers labels from `req_to_token` and masks only by `written_layer[safe_phys]` and `seq_lens` (`selection_kernel.py:383-400`). If the old request left `written=True`, the stale label is valid to the selector.
   - In the live decode path, `_select_topk_indices` is called before `attn_mqa` writes the current KV/label (`forward_mla.py:283`, then `dsa_backend.py:1703-1709`). That means the current step can read a reused slot before `_write_token_labels` overwrites it.

   I verified this with a synthetic fixture: old request writes `1000.0` at physical slot 7, new `req_to_token` maps logical position 0 to slot 7, no new write occurs, and `retrieve_topk_via_labels` returns `[[0]]` with `valid_lengths=[1]`. That is exactly the stale read AC-2 was meant to prevent.

2. **AC-2 slot-budget/fail-fast coverage is still incomplete.**

   `test_slot_budget_covers_all_physical_kv_slots` (`test_double_sparsity_unit.py:4994`) allocates an arbitrary `max_tokens = kv_pool_size + page_size` and writes the last slot. It does not exercise a server/bind-time invariant that token-label capacity cannot exceed the KV pool slot budget, and there is no boot-time HBM/fail-fast gate in `token_label_table.py` or `validator.py` that would reject an independently over-sized table as the plan requires.

3. **AC-3 production ownership is not tested.**

   Both new AC-3 tests call `retrieve_topk_via_labels` without `req_pool_indices`, `req_to_token`, or `seq_lens` (`test_double_sparsity_unit.py:5119` and `:5157`). That forces the physical-domain branch (`selection_kernel.py:440-468`), so the tests only prove that an explicit `[bs, max_tokens]` `per_request_valid` mask works.

   The production path is different: `DeepseekV2AttentionMLA._select_topk_indices` obtains `forward_batch.req_to_token_pool.req_to_token`, calls `selector.retrieve_topk(..., req_to_token=req_to_token)`, then maps logical positions back through `logical_to_physical` (`deepseek_v2.py:2061-2114`). Round 5 does not add a boundary test for that path, so a regression that drops `req_to_token` from the live call could still escape.

4. **Remaining original-plan work is still pending.**

   Claude's "Remaining Items" are unfinished tasks, not accepted deferrals: AC-1 H200 population, AC-1b chunked-prefill probe, AC-4 calibration and mask generation, AC-5 TP harness, AC-6 CUDA graph capture, AC-7 short-seq bypass, AC-8 bench/quality smoke, AC-12 full quality gate, and AC-9 through AC-11 stretch rows all remain open.

## Blocking Side Issues

None outside the mainline. The stale-slot bug above is itself an AC-2 mainline gap and must be fixed before AC-2 can be marked complete.

## Queued Side Issues

1. Before AC-6, update `capture_decode_step` to pass `req_to_token` and avoid the physical-domain fallback during graph capture.
2. Before AC-8, fix DS observability metrics that still use page-named fields and page-count denominators.
3. Clean stale `deepseek_v2.py` comments that still describe table sizing as `req_to_token_pool.size`.

## Goal Alignment Summary

```text
ACs: 5/15 addressed | Forgotten items: 0 | Unjustified deferrals: 0
```

Met: AC-0, AC-13. Partial: AC-1, AC-2, AC-3. All other ACs remain not met. The tracker has no explicit deferred tasks; AC-9 through AC-11 remain active stretch work, not accepted deferrals.

## Goal Tracker Update

I updated `.humanize/rlcr/2026-05-27_11-38-21/goal-tracker.md`:
- Added a Round 5 Review plan-evolution row reopening AC-2/AC-3 as partial.
- Kept `task-ac2-lifetime`, `task-m2-rangemask`, and `task-ac3-test` in Active with `partial-r5-review`.
- Removed the pending AC-2/AC-3 rows from Completed and Verified.

## Required Implementation Plan

1. Fix AC-2 stale-slot lifetime now. Add a capture-safe invalidation helper for token-label slots, e.g. `invalidate_token_label_slots(table, layer_id, cache_loc)`, that sets `table.written[layer_id, cache_loc] = False` before the selector can read newly allocated slots. Call it from the DS branch of `_select_topk_indices` before `selector.retrieve_topk`, using `forward_batch.out_cache_loc` and the current `layer_id`. This makes reused slots unselectable until `_write_token_labels` rewrites them later in `dsa_backend.py`; it also protects `save_kv_cache=False` fused paths by leaving the slot invalid instead of stale.
2. Add the missing AC-2 regression: write old label A to slot N, reassign N through `req_to_token` for a new request, run selection before the new write and assert the old label is not selectable; then write label B and assert B becomes selectable. Add a bind-time shape guard so reused or preexisting token-label tables must have `max_tokens == kv_pool.size + kv_pool.page_size`, and test the oversize/undersize failure path.
3. Replace the Round 5 AC-3 helper-only test with a production-path boundary test. Build a real bound `DoubleSparsitySelector` or `_select_topk_indices` fixture with two `req_to_token` rows mapping logical positions to disjoint physical ranges; make request 1's physical slots outscore request 0's globally; assert the adapter output for each request stays inside that request's `req_to_token` row. The negative test should deliberately drop `req_to_token` or call the physical helper without a mask and show contamination.
4. Rerun the full unit suite. Only after AC-2/AC-3 are fixed and verified should the tracker move those tasks to Completed and Verified.
5. Continue in dependency order: AC-7 short-seq bypass including `save_kv_cache=True`/fused-path verification; AC-4 Method 1 calibration by capturing Q-noPE and K-noPE in the same forward pass and computing `mean(abs(Q_nope * K_nope))`; AC-5 TP=2 multiprocess logical-domain all-reduce; AC-6 graph capture with preallocated buffers and `req_to_token`; AC-1 H200 population, AC-1b chunked-prefill probe, AC-8 bench/quality smoke, AC-12 full NIAH/MMLU quality gate, then AC-9 through AC-11 stretch measurements.

NOT COMPLETE
