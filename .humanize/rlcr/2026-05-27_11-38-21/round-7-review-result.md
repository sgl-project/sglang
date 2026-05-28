# Round 7 Review Result

Mainline Progress Verdict: ADVANCED

Round 7 closes the narrow AC-2 live-wiring gap: `_select_topk_indices` still invalidates `forward_batch.out_cache_loc` before `retrieve_topk`, and `TestAC2LiveWiring.test_production_hook_invalidates_before_retrieve_topk` would fail if that hook were removed.

I verified the claimed local suite:

```text
PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q
176 passed, 24 warnings in 11.62s
```

AC-7 is not closed. The new tests exercise a synthetic `forward_batch.attn_backend` attribute, but production stores the MHA decision on the active `ForwardContext` backend.

## Mainline Gaps

1. **AC-7 bypass reads the wrong production state, so the new guard does not fire in the real forward path.**

   Evidence:
   - The new guard is `getattr(forward_batch, "attn_backend", None)` at `python/sglang/srt/models/deepseek_v2.py:2077-2079`.
   - `ForwardBatch` has no `attn_backend` field (`python/sglang/srt/model_executor/forward_batch_info.py:273-289`); `dataclasses.fields(ForwardBatch)` confirms `attn_backend` is absent.
   - Production publishes the backend via `ForwardContext(attn_backend=self.attn_backend)` in `model_runner.py:2715`, and the DSA dispatcher reads `get_attn_backend().use_mha` in `attention_backend_handler.py:164-168`.
   - The Round 7 tests attach `attn_backend=SimpleNamespace(use_mha=True)` directly to a `SimpleNamespace` forward batch at `test_double_sparsity_unit.py:5545-5552`, which is not the production data path.
   - Reproduction with a real `ForwardContext(attn_backend=SimpleNamespace(use_mha=True))` and a forward batch without `attn_backend` produced:

   ```text
   retrieve_topk_called= True
   result_is_none= False
   ```

   Required fix: remove the dead `forward_batch.attn_backend` dependency. Either implement the bypass at the call site in `forward_absorb_prepare` or keep a defensive guard in `_select_topk_indices` that reads the active backend from `get_attn_backend()` and unwraps `TboAttnBackend` the same way `handle_attention_dsa` does. Add a regression that runs under `forward_context(ForwardContext(attn_backend=backend_with_use_mha_true))` with a forward batch that has no synthetic `attn_backend` attribute, then asserts `retrieve_topk` is not called.

2. **AC-7 label-write coverage is still missing for the actual dense prefill path.**

   Evidence:
   - The summary says `dsa_backend._write_token_labels` fires before the `if self.use_mha:` branch at `dsa_backend.py:1510-1513`.
   - However, when the DSA dispatcher sees `backend.use_mha=True`, it returns `AttnForwardMethod.MHA_ONE_SHOT` (`attention_backend_handler.py:167-168`), and `DeepseekV2AttentionMLA.forward_prepare` routes to `forward_normal_one_shot_prepare` (`deepseek_v2.py:1755-1758`), not `forward_absorb_prepare`.
   - `forward_normal_prepare` writes KV through `_set_mla_kv_buffer` at `forward_mha.py:253`; `_set_mla_kv_buffer` only calls `set_mla_kv_buffer` (`forward_mha.py:451-475`) and does not call `_write_token_labels`.
   - The later MHA attention call uses `self.attn_mha(..., save_kv_cache=False)` (`forward_mha.py:302-310`), so the `dsa_backend.forward_extend` hook guarded by `if save_kv_cache:` cannot populate labels for this path.
   - Round 7 did not add the required tests for “label write still fires during dense prefill” or “first decode after short prefill calls `retrieve_topk`.”

   Required fix: wire the DS label write into the actual MHA_ONE_SHOT KV-write path. After `_set_mla_kv_buffer` writes `kv_a` to `forward_batch.out_cache_loc`, call the active DSA backend’s `_write_token_labels` with the same cache locations and normalized latent key (`kv_a.unsqueeze(1)`), guarded by `hasattr(backend, "_write_token_labels")` so non-DS backends are unaffected. Then add tests that exercise `forward_normal_one_shot_prepare`/`_set_mla_kv_buffer`, not only `_select_topk_indices`, and prove labels are written during short dense prefill.

3. **The original lower-bound plan remains incomplete.**

   Still pending from the plan:
   - AC-1 H200 real `forward_extend` population and AC-8 selector-read smoke.
   - AC-1b chunked-prefill probe.
   - AC-4 Method 1 calibration plus H200 mask generation.
   - AC-5 TP=2 multiprocess all-reduce harness.
   - AC-6 CUDA graph capture and H200 replay.
   - AC-7 production bypass and dense-prefill label-write proof.
   - AC-8 `bench_serving` plus lightweight quality smoke.
   - AC-12 hard NIAH/MMLU quality gate.
   - AC-9 through AC-11 stretch baseline/radix/comparator work.

## Blocking Side Issues

None outside the mainline gaps above. The AC-7 production-state and label-write bugs are mainline blockers for AC-7 itself.

## Queued Side Issues

1. Before AC-6, update `capture_decode_step` to pass `req_to_token`; otherwise graph capture uses the physical-domain fallback.
2. Before AC-8, fix DS observability fields that still publish token selections as page metrics and divide by page counts.
3. Clean stale lifetime/sizing documentation already tracked in the goal tracker.
4. Remove plan markers from production comments while touching this code; the new `# AC-7` comment violates the plan’s implementation-note ban on `AC-` terminology in code.

## Goal Alignment Summary

```text
ACs: 6/15 addressed | Forgotten items: 0 | Unjustified deferrals: 0
```

Verified/met: AC-0, AC-2, AC-3, AC-13. Partial: AC-1, AC-7. Not met: AC-1b, AC-4, AC-5, AC-6, AC-8, AC-9, AC-10, AC-11, AC-12. There are no accepted deferrals; the remaining tasks are active pending work, not waived scope.

## Goal Tracker Update

I updated `.humanize/rlcr/2026-05-27_11-38-21/goal-tracker.md`:
- Added a Round 7 Review plan-evolution row reopening AC-7.
- Moved `task-ac7-bypass` back to Active with `reopened-r7-review`.
- Removed `task-ac7-bypass` from Completed and Verified.
- Removed the duplicate `task-ac4-calibrate` Active row.
- Kept AC-2 in Completed and Verified.

## Required Implementation Plan

1. Finish AC-7 before moving on. Read the MHA decision from the active backend (`get_attn_backend()` / TBO primary), not from `ForwardBatch`. Add the production-context bypass regression, the dense-prefill label-write test, and the first-decode-after-prefill `retrieve_topk` test. Rerun the full unit suite, then move AC-7 to Completed only after review verifies those tests cover production state.
2. Implement AC-4 calibration next: replace K-only statistics with same-forward-pass Q-noPE/K-noPE Method 1 `mean(abs(Q_nope * K_nope))`, keep `qk_nope_head_dim=128`, add fixture tests for Q/K pairing and 512-d index rejection, then run the H200 mask generation and `load_channel_mask` validation.
3. Implement AC-5: add `test/registered/integration/test_double_sparsity_tp_multiprocess.py`, spawn TP=2 processes, all-reduce `[bs, max_seq_len]` logical-domain scores, assert bit-equal logical selections, and include the no-op all-reduce negative.
4. Implement AC-6: thread `req_to_token` through graph capture, preallocate DS output/scratch buffers before capture, add eager-vs-replay and allocation-negative tests, then run the H200 conc=64 replay check.
5. Run the hardware/analyze gates in dependency order: AC-1 H200 population, AC-1b chunked-prefill probe, AC-8 server benchmark and lightweight quality smoke, AC-12 full quality gate, then AC-9 through AC-11 stretch measurements.

NOT COMPLETE
