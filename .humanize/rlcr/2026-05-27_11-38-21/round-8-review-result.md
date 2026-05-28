# Round 8 Review Result

Mainline Progress Verdict: ADVANCED

Round 8 fixed the two concrete AC-7 production bugs identified in Round 7:

- `_select_topk_indices` now reads `use_mha` from the active `ForwardContext` backend and unwraps `TboAttnBackend` (`python/sglang/srt/models/deepseek_v2.py:2071-2089`).
- `_set_mla_kv_buffer` now calls the active backend's `_write_token_labels` after the MHA KV write when `use_double_sparsity=True` (`python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py:477-486`).

I verified the claimed local suite:

```text
PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q
178 passed, 24 warnings in 11.60s
```

This is real progress, but the original plan is still not complete and AC-7 should remain open until the missing decode-after-prefill proof lands.

## Mainline Gaps

1. **AC-7 still lacks the required first-decode-after-short-prefill proof.**

   Evidence:
   - The original AC-7 requires: "First decode step after short prefill: `retrieve_topk` IS invoked and returns a non-empty, non-trivial selection."
   - The Round 8 contract explicitly required `test_decode_after_mha_prefill_calls_retrieve_topk`.
   - No such test exists. The closest test is `test_no_bypass_when_forward_context_use_mha_false` (`test/registered/unit/layers/attention/test_double_sparsity_unit.py:5586-5617`), but it only wraps a generic `ForwardContext(use_mha=False)` and mocks `retrieve_topk` to return one synthetic index. It does not run after a dense-prefill label write, does not use a real bound selector/table, and does not prove the selection is non-empty because `_set_mla_kv_buffer` populated labels.
   - `test_mha_label_write_fires_in_set_mla_kv_buffer` (`test_double_sparsity_unit.py:5674-5719`) proves a spy was called with `kv_a.unsqueeze(1)`, but it does not chain that write into the next decode selector call.

   Required fix: add the missing AC-7 regression before moving to AC-4. Use one test fixture that performs these steps in order:
   - Allocate a real `TokenLabelTable` and bind a `DoubleSparsitySelector` with a synthetic `ChannelMask`.
   - Build a fake DSA backend with `enable_double_sparsity=True`, `_ds_token_label_table`, `_ds_channel_selection`, and `_ds_qk_nope_head_dim`, then use the real `NativeSparseAttnBackend._write_token_labels` implementation through `_set_mla_kv_buffer`.
   - In a `ForwardContext` with `backend.use_mha=True`, call `_select_topk_indices` and assert it returns `None` without invoking `retrieve_topk`; then call `_set_mla_kv_buffer` for short-prefill slots and assert `table.written[layer, slots]` is true and signatures are non-zero.
   - Flip the same backend to `use_mha=False` for the first decode step. Set `req_to_token` so logical positions map to the prefilled physical slots plus the new decode slot, set `out_cache_loc` to only the new decode slot, and call `_select_topk_indices` with a real query that scores at least one prefill slot.
   - Assert `retrieve_topk` runs and the adapter output contains a non-`-1` physical prefill slot. This test must fail if either the MHA label-write hook or the `use_mha=False` decode path is broken.

2. **The lower-bound plan remains incomplete. These are pending tasks, not accepted deferrals.**

   Evidence by AC:
   - AC-1: `task-ac1-hwtest` is still active; no H200 real `forward_extend` population result or AC-8 selector-read smoke is recorded.
   - AC-1b: chunked-prefill probe is still active and unrun.
   - AC-4: `calibrate.py` still documents and computes K-only L2 statistics (`pow(2)`) rather than same-forward-pass Method 1 `mean(abs(Q_nope * K_nope))` (`python/sglang/srt/layers/attention/double_sparsity/calibrate.py:1-5`, `:196-205`).
   - AC-5: `test/registered/integration/test_double_sparsity_tp_multiprocess.py` does not exist.
   - AC-6: `capture_decode_step` still has no `req_to_token` parameter and calls `selector.retrieve_topk` without it in both eager and CUDA paths (`python/sglang/srt/layers/attention/double_sparsity/cuda_graph.py:109-183`), so graph capture would exercise the physical-domain fallback.
   - AC-8: the required lightweight quality smoke file is absent, and the DS observability path still publishes token selections through page-named counters/fields.
   - AC-12: `test/manual/test_double_sparsity_v32.py` remains a skip-only scaffold, not a NIAH/MMLU quality gate.
   - AC-9 through AC-11: baseline JSON, radix-cache validation, and comparator rows are not complete.

## Blocking Side Issues

None outside the mainline gaps above. The missing AC-7 decode-after-prefill proof is a mainline gap, not a side issue.

## Queued Side Issues

1. Before AC-6, thread `req_to_token` through `capture_decode_step`; otherwise CUDA graph validation uses the wrong selector domain.
2. Before AC-8, rename/fix DS observability so token selections are not reported as pages and sparsity is not divided by page counts.
3. Clean stale comments/docstrings about `req_to_token_pool.size` sizing and overwrite-before-read label lifetime when touching those modules.

## Goal Alignment Summary

```text
ACs: 6/15 addressed | Forgotten items: 0 | Unjustified deferrals: 0
```

Verified/met: AC-0, AC-2, AC-3, AC-13. Partial: AC-1, AC-7. Not met: AC-1b, AC-4, AC-5, AC-6, AC-8, AC-9, AC-10, AC-11, AC-12. There are no accepted deferrals in the tracker; the remaining tasks are active pending work.

## Goal Tracker Update

I updated `.humanize/rlcr/2026-05-27_11-38-21/goal-tracker.md`:

- Added a Round 8 Review plan-evolution row reopening AC-7 first-decode proof.
- Moved `task-ac7-bypass` back to Active as `partial-r8-review`.
- Removed AC-7 from Completed and Verified.
- Kept all other pending lower-bound and stretch tasks Active.

## Required Implementation Plan

1. Finish AC-7 first. Add the first-decode-after-short-prefill regression described above, using a real bound selector/table and a real `_write_token_labels` call from `_set_mla_kv_buffer`. Rerun `PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q`. Move AC-7 back to Completed only after that test proves the dense-prefill labels feed decode selection.

2. Implement AC-4 calibration next. Replace the K-only L2 hook in `calibrate.py` with paired same-forward-pass Q-noPE and K-noPE capture per layer, compute Method 1 `mean(abs(Q_nope * K_nope))` over the 128-d noPE channel axis, keep `qk_nope_head_dim=128`, and reject 512-d channel indices. Add fixture tests for Q/K pairing, label-dim mismatch, and 512-d rejection. Then run the H200 command from the plan and validate `/models/dsv32-fp8-channel-mask.safetensors` with `load_channel_mask`.

3. Implement AC-5. Create `test/registered/integration/test_double_sparsity_tp_multiprocess.py`, spawn two `torch.multiprocessing` ranks, initialize a process group, compute logical-domain `[bs, max_seq_len]` score tensors, all-reduce them, and assert bit-equal logical `selected_token_indices`. Include the physical-slot permutation case and the no-op all-reduce negative.

4. Implement AC-6. Add `req_to_token` to `capture_decode_step` and pass it through every `selector.retrieve_topk` call. Preallocate output and scratch buffers before capture, add eager-vs-replay and allocation-negative tests, then run the H200 conc=64 capture plus 100 replay steps.

5. Run the hardware and quality gates in dependency order: AC-1 H200 population, AC-1b chunked-prefill probe, AC-8 `bench_serving` and lightweight quality smoke, AC-12 full NIAH/MMLU gate, then AC-9 through AC-11 stretch measurements. Record exact commands, commit SHA, server args, chunked-prefill setting, and result JSON paths in the round summary and goal tracker.

NOT COMPLETE
