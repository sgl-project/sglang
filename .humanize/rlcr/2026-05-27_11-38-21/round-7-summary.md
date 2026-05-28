# Round 7 Summary

## Work Completed

### AC-2 live-wiring regression — `TestAC2LiveWiring`

**`test_production_hook_invalidates_before_retrieve_topk`**
Constructs a `DeepseekV2AttentionMLA` fixture (via `object.__new__`) with `use_double_sparsity=True`, `IS_PLACEHOLDER=False`. Attaches a `TokenLabelTable` with `written[0, 7] = True` (stale). Sets `forward_batch.out_cache_loc = torch.tensor([7])`. Monkey-patches `selector.retrieve_topk` with a `side_effect` spy that captures `written[0, 7]` at call time. Calls `_select_topk_indices` normally. Asserts:
1. The spy was called exactly once (retrieve_topk fired).
2. `written[0, 7]` was `False` when the spy ran — i.e., the invalidation hook (lines 2087-2093 of `deepseek_v2.py`) fired before `retrieve_topk`. Test FAILS if those lines are deleted.

**`test_after_hook_written_is_restored_by_label_write`**
Exercises the full invalidate → label-write lifecycle: invalidates slot 7, asserts `written=False`, writes a new label, asserts `written=True`. Confirms the lifecycle in the call order that production uses.

### AC-7 bypass — `_select_topk_indices` in `deepseek_v2.py`

Added a guard immediately after the DS-path imports in `_select_topk_indices` (lines 2071-2079):

```python
# AC-7: skip sparse selection during short-seq MHA-mode prefill.
_attn_backend = getattr(forward_batch, "attn_backend", None)
if getattr(_attn_backend, "use_mha", False):
    return None
```

When `forward_batch.attn_backend.use_mha` is `True` (short extend below `SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD` on SM90/SM100), selection is skipped entirely. The label write in `dsa_backend._write_token_labels` (line 1510) fires unconditionally — it precedes the `if self.use_mha:` branch at line 1513 — and is unaffected by this change.

### AC-7 tests — `TestAC7MHABypass` (4 tests)

**`test_mha_bypass_returns_none_and_skips_retrieve_topk`**
`attn_backend.use_mha=True` → result is `None`, `retrieve_topk` not called.

**`test_no_bypass_when_use_mha_false`**
`attn_backend.use_mha=False` → `retrieve_topk` called once, result is non-None tensor.

**`test_bypass_when_no_attn_backend`**
`attn_backend=None` → `use_mha` defaults to `False` via `getattr` fallback → `retrieve_topk` called (no bypass).

**`test_mha_bypass_does_not_affect_nsa_path`**
`use_double_sparsity=False` with `attn_backend.use_mha=True` → NSA `indexer` called, MHA flag is irrelevant.

## Files Changed

- `python/sglang/srt/models/deepseek_v2.py` — AC-7 bypass (10 lines added in `_select_topk_indices`)
- `test/registered/unit/layers/attention/test_double_sparsity_unit.py` — `TestAC2LiveWiring` (2 tests) + `TestAC7MHABypass` (4 tests)

## Validation

```
PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q
176 passed, 0 failed
```

Commit: `a81b6532e`

## Remaining Items

- `task-ac1-hwtest`: H200 hardware population test — pending hardware access.
- `task-ac4-calibrate`: Method 1 Q+K joint hooks in calibrate.py.
- `task-ac4-hwrun`: Hardware run on H200 to generate dsv32-fp8-channel-mask.safetensors.
- `task-ac5-tp`: TP=2 multiprocess all-reduce test.
- `task-ac6-cuda-graph`: Decode-path graph capture.
- `task-ac6-hwrun`: Hardware full-graph capture at conc=64.
- `task-ac1b-probe`: Chunked-prefill probe.
- `task-ac8-server`, `task-ac8-quality`, `task-ac12-quality`: Server smoke + quality gates.
- `task-ac9-baseline`, `task-ac10-radix`, `task-ac11-compare`: Stretch comparators.

## BitLesson Delta

Action: none
Lesson ID(s): NONE
Notes: AC-2 live wiring is a standard spy/side-effect pattern (capture assertion inside mock side_effect). AC-7 bypass is a standard guard-before-main-logic pattern. Neither introduces a project-specific lesson beyond what is already captured in the existing test fixtures.
