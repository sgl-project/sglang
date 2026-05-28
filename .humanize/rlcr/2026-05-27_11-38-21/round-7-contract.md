# Round 7 Contract

## Tasks This Round

### task-ac2-live-wiring (coding)
Add `TestSelectTopkIndicesHookBranch` that tests the production `_select_topk_indices` invalidation hook (lines 2087-2093 in `deepseek_v2.py`). The test must:
- Construct a `DeepseekV2AttentionMLA`-like fixture with `use_double_sparsity=True`
- Set `selector.token_label_table.written[0, 7] = True` (stale)
- Set `forward_batch.out_cache_loc = torch.tensor([7])`
- Monkey-patch `selector.retrieve_topk` with a spy that asserts `written[0, 7] is False` when called
- Verify the spy was called (i.e., the hook fired and invalidated before selection)
- The test MUST fail if lines 2087-2093 of `deepseek_v2.py` are removed

AC-2 completion gate: this test closes the gap Codex identified — existing tests call `invalidate_token_label_slots` directly; this test exercises the production wiring path.

### task-ac7-bypass (coding)
Add short-seq MHA bypass in `forward_absorb_prepare` (`forward_mla.py:~283`):
- When `use_double_sparsity=True` AND `use_mha=True`, skip `_select_topk_indices` (return `topk_indices = None` for the layer)
- Keep `_write_token_labels` firing (label write must still happen during MHA prefill)
- Add tests:
  - `test_mha_bypass_skips_retrieve_topk`: spy confirms `retrieve_topk` NOT called when `use_mha=True`
  - `test_mha_bypass_label_write_still_fires`: label write hook fires even when selection is bypassed
  - `test_first_decode_after_mha_prefill_calls_retrieve_topk`: after bypass, decode path calls retrieve_topk normally
  - `test_no_bypass_when_use_mha_false`: negative — `retrieve_topk` IS called when `use_mha=False`

## Files To Touch
- `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` — AC-7 bypass
- `test/registered/unit/layers/attention/test_double_sparsity_unit.py` — new test classes

## Invariants
- Round contract written before any code change ✓
- bitlesson-selector run before each task
- Tests must be hermetic (CPU tensors, no CUDA required)
- goal-tracker.md updated before writing summary

## Exit Criteria
All new tests pass; `170 + N` tests pass where N = new tests added this round.
