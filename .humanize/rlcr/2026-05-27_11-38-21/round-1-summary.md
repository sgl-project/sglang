# Round 1 Summary

## Objective

Close AC-0 and AC-13 by repairing the 5 verified gaps reported in the Round-0 Codex review.

## Result

**150 passed, 0 failed** — all success criteria from the Round-1 contract are met.

## Work Completed

### Gap 1: `retrieve_topk` not exported (AC-0)

- `selection_kernel.py`: Added `retrieve_topk = retrieve_topk_via_labels` alias at module bottom.
- `__init__.py`: Added `from .selection_kernel import retrieve_topk` and `"retrieve_topk"` to `__all__`.
- Verification: `python -c "from sglang.srt.layers.attention.double_sparsity import TokenLabelTable, token_label_write, retrieve_topk; print('OK')"` prints OK.

### Gap 2: Selector domain — physical slots instead of logical positions (AC-0, AC-3)

- `selection_kernel.py`: Added `_compute_logical_token_scores()` helper that gathers physical labels per logical position via `req_to_token` gather, returning `[bs, max_seq_len]` fp32 scores with unwritten/OOB positions masked to `-inf`.
- `retrieve_topk_via_labels` detects optional `(req_pool_indices, req_to_token, seq_lens)` triple; when present, runs logical-domain scoring and returns logical positions (0-indexed within each request's sequence); when absent, falls back to original physical-slot path (preserves probe and unit test compatibility).
- `selector.py`: `retrieve_topk` passes the three new optional args through.

### Gap 3: Bind timing — DS bound before KV pool exists (AC-0, AC-1, AC-2)

- `deepseek_v2.py`: `__init__` now stores `self._ds_deferred_bind_args` instead of calling `_bind_double_sparsity_runtime_data` directly. New method `finalize_double_sparsity_bind()` calls it then clears the stored dict.
- `_bind_double_sparsity_runtime_data` removed the `device_buffer_size` fallback; now raises `RuntimeError` if `_ds_req_to_token_pool` is None (fail-fast invariant).
- `model_runner.py`: Added post-`init_memory_pool()` loop that calls `finalize_double_sparsity_bind()` on every module that exposes it.

### Gap 4: DS selector receives latent `q_lora` instead of projected Q-noPE (AC-0, AC-1, AC-8)

- `forward_mla.py`: Added `and not self.use_double_sparsity` to the alt-stream condition so DS always waits for `q_b_proj` to complete on the normal path. In the normal branch, derives `q_nope_for_ds = q[..., :self.qk_nope_head_dim]` after `q_b_proj` and passes it as `q_nope=q_nope_for_ds` to `_select_topk_indices`.
- `deepseek_v2.py:_select_topk_indices`: Added `q_nope: Optional[torch.Tensor] = None` parameter; uses `q_nope` when provided, falls back to `q_lora` otherwise.
- Gate alignment fix: alt-stream branch inner gate updated to `if (self.use_double_sparsity or not self.skip_topk or prev_topk_indices is None):` so both branches carry the identical DS-aware predicate required by `TestSkipTopkGateRespectsDS`.

### Gap 5: AC-13 test failures and stale `nsa_*` names (AC-13)

- Test file: renamed `nsa_prefill_backend`/`nsa_decode_backend` kwargs to `dsa_prefill_backend`/`dsa_decode_backend`.
- `test_nsametadata_has_ds_topk_indices_out_field`: import updated to `DSAMetadata as NSAMetadata`.
- `test_forward_decode_dispatches_to_flashmla_kv`: fixed `nsa_*` metadata fields to `dsa_*`; added `backend.dsa_decode_impl`, `backend.token_to_kv_pool`, and `backend.hisparse_coordinator = None` to mock.
- `test_ds_branch_returns_topk_indices_via_adapter`: added `req_to_token_pool` to `forward_batch` so the logical-domain path is exercised.
- `test_probe_finds_planted_needle`: fixed `max_tokens=16→512` and `needle_position=4→4*64` for token-level label layout.

### Side fix: `channel_mask.py` needle_position inconsistency

Both skip paths now use `needle_page * page_size` for consistent physical-address results.

## Files Changed

- `python/sglang/srt/layers/attention/double_sparsity/__init__.py`
- `python/sglang/srt/layers/attention/double_sparsity/channel_mask.py`
- `python/sglang/srt/layers/attention/double_sparsity/selection_kernel.py`
- `python/sglang/srt/layers/attention/double_sparsity/selector.py`
- `python/sglang/srt/model_executor/model_runner.py`
- `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`
- `python/sglang/srt/models/deepseek_v2.py`
- `test/registered/unit/layers/attention/test_double_sparsity_unit.py`

## Validation

| Command | Result |
|---------|--------|
| `python -c "from sglang.srt.layers.attention.double_sparsity import TokenLabelTable, token_label_write, retrieve_topk; print('OK')"` | OK |
| `PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q` | 150 passed, 0 failed |

## Success Criteria Verification

| Criterion | Result |
|-----------|--------|
| `from ... import TokenLabelTable, token_label_write, retrieve_topk` | PASS |
| Non-contiguous fixture: logical [0,1,2,3] → physical [7,64,200,512] | PASS (test_ds_branch_returns_topk_indices_via_adapter) |
| 150 tests pass | PASS — 150 passed, 0 failed |
| `_bind_double_sparsity_runtime_data` raises if pool absent | PASS — RuntimeError raised, no silent fallback |
| `_select_topk_indices` passes `q[..., :qk_nope_head_dim]` to DS | PASS — q_nope_for_ds derived after q_b_proj |

## Remaining Items

None for this round's scope. AC-0 and AC-13 are closed. Next round targets AC-1 (live token-label cache population from KV-write path).

## BitLesson Delta

Action: none
Lesson ID(s): NONE
Notes: All fixes addressed pre-identified Round-0 gaps. No new generalizable lessons arose.
