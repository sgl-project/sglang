# Round 5 Contract

## Mainline Objective

Implement AC-2 lifetime/stale-slot verification and AC-3 per-request token range ownership mask. Both are unit-testable without hardware access.

## Target ACs

- **AC-2** (sole unit-testable target): Boot-time GB/rank log already emits; add unit tests that cover the 2× KV-slot budget invariant and the stale-slot overwrite-before-read semantics.
- **AC-3**: Attach per-request token range to the scoring path so `retrieve_topk` only picks tokens within a request's own slots; add multi-request boundary test and negative fixture.

## Tasks (in order)

### task-ac2-lifetime (coding, claude)

AC-2 requires:
1. **Boot-time log already exists** in `token_label_table.py`. Verify format in a test.
2. **2× KV-slot budget test**: table `max_tokens` must equal `kv_pool.size + kv_pool.page_size` (not larger); allocating with `max_tokens` that exceeds this must raise or misindex.
3. **Stale-slot overwrite test**: write label A at slot N; overwrite with label B at slot N; assert slot N now reads B (not A). Verifies that `token_label_write` unconditionally overwrites rather than accumulating.
4. **Negative**: read slot N immediately after write A but before write B; assert it reads A (data is visible, no phantom state).

### task-m2-rangemask (coding, claude)

AC-3 ownership mask — cross-request contamination prevention:
1. In `selection_kernel.py` (or `selector.py`), the scorer must receive, per request, the token index range `[req_start, req_end)` that belongs to that request.
2. Any label slot outside `[req_start, req_end)` must receive score `−inf` before top-K, so `retrieve_topk` cannot select cross-request tokens.
3. The range must be derivable from `ForwardBatch` (e.g. from `req_to_token_pool` offsets) and passed through to the scorer.

### task-ac3-test (coding, claude)

Multi-request boundary test and negative fixture:
1. **Positive test**: Construct a batch with 2 requests occupying non-overlapping slot ranges; run `retrieve_topk`; assert all selected indices for req-0 fall within req-0's range, and all for req-1 fall within req-1's range.
2. **Negative test**: Remove the ownership mask; run the same fixture; assert cross-request picks occur (i.e. if it were pure top-K, some req-0 high-score tokens would land in req-1's results).

## Blocking Issues

None currently. All tasks are unit-testable.

## Success Criteria

1. `PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q` → all tests pass (≥ 160 currently, plus new AC-2/AC-3 additions).
2. `TestAC2Lifetime` class: boot-log format test, 2× budget test, stale-overwrite test, negative read test.
3. `TestAC3RangeMask` class: positive multi-request boundary test, negative (mask-off) fixture.
4. No new imports of removed names (no `PageSignatureTable`, `page_signature_write`).
