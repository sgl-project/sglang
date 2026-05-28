# Round 8 Summary

## Work Completed

### AC-7 Fix 1 — Production bypass reads `ForwardContext` (`deepseek_v2.py`)

Replaced the dead `forward_batch.attn_backend` read with a `has_forward_context()` + `get_attn_backend()` guard that mirrors `handle_attention_dsa`:

```python
from sglang.srt.layers.attention.tbo_backend import TboAttnBackend as _TboAttnBackend
from sglang.srt.model_executor.forward_context import (
    get_attn_backend as _get_attn_backend,
    has_forward_context as _has_forward_context,
)
if _has_forward_context():
    _fc_backend = _get_attn_backend()
    if isinstance(_fc_backend, _TboAttnBackend):
        _fc_backend = _fc_backend.primary
    if hasattr(_fc_backend, "use_mha") and _fc_backend.use_mha:
        return None
```

`ForwardBatch` has no `attn_backend` dataclass field — the old code was reading from a field that production never sets. The `has_forward_context()` guard preserves backward compatibility for unit tests that do not publish a `ForwardContext`.

### AC-7 Fix 2 — Label write in MHA_ONE_SHOT path (`forward_mha.py`)

Added after all pool-write branches in `_set_mla_kv_buffer`:

```python
if getattr(self, "use_double_sparsity", False):
    _ds_backend = _resolve_attn_backend(forward_batch)
    if hasattr(_ds_backend, "_write_token_labels"):
        _ds_backend._write_token_labels(
            self.attn_mha, forward_batch.out_cache_loc, kv_a.unsqueeze(1)
        )
```

`dsa_backend.forward_extend` guards `_write_token_labels` behind `save_kv_cache=True`, but `forward_normal_prepare` calls `_set_mla_kv_buffer` then passes `save_kv_cache=False` to the attention kernel — so labels were NEVER written for short dense prefills. This hook closes that gap. `_resolve_attn_backend` already handles `TboAttnBackend` unwrap.

### TestAC7MHABypass — updated to use real `ForwardContext` (6 tests)

**`test_bypass_fires_via_forward_context_use_mha_true`**
Runs under `forward_context(ForwardContext(attn_backend=mock_backend_use_mha_true))`.
`forward_batch` has NO `attn_backend` attribute — the production case.
Asserts: `result=None`, `retrieve_topk.assert_not_called()`.

**`test_no_bypass_when_forward_context_use_mha_false`**
`ForwardContext.use_mha=False` → `retrieve_topk` called, result non-None.

**`test_no_bypass_without_forward_context`**
No `ForwardContext` published → `has_forward_context()=False` → no bypass → `retrieve_topk` called.
Preserves backward compatibility for legacy unit tests.

**`test_mha_bypass_does_not_affect_nsa_path`**
`use_double_sparsity=False` → DS bypass block not entered → NSA indexer called.

**`test_mha_label_write_fires_in_set_mla_kv_buffer`**
Runs under `forward_context(ForwardContext(attn_backend=mock_backend_with_spy))`.
Calls `_set_mla_kv_buffer` directly with `T=3, kv_lora_rank=4` tensors.
Asserts: spy called once, `k.shape == [3, 1, 4]` (= `kv_a.unsqueeze(1)`).
Test FAILS if the `_write_token_labels` call is removed from `_set_mla_kv_buffer`.

**`test_no_label_write_when_not_double_sparsity`**
`use_double_sparsity=False` → `_write_token_labels` not called (negative).

## Files Changed

- `python/sglang/srt/models/deepseek_v2.py` — bypass reads `ForwardContext`
- `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py` — `_write_token_labels` call in `_set_mla_kv_buffer`
- `test/registered/unit/layers/attention/test_double_sparsity_unit.py` — `TestAC7MHABypass` updated (6 tests vs 4)

## Validation

```
PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q
178 passed, 0 failed
```

Commit: `8e2a18f03`

## Remaining Items

- `task-ac1-hwtest`: H200 hardware population test.
- `task-ac4-calibrate` / `task-ac4-hwrun`: Method 1 calibration + mask generation.
- `task-ac5-tp`: TP=2 multiprocess all-reduce test.
- `task-ac6-cuda-graph` / `task-ac6-hwrun`: Graph capture + H200 replay.
- `task-ac1b-probe`, `task-ac8-*`, `task-ac12-quality`: Hardware/analyze gates.
- `task-ac9-baseline`, `task-ac10-radix`, `task-ac11-compare`: Stretch.

## BitLesson Delta

Action: none
Lesson ID(s): NONE
Notes: Both fixes follow patterns already in the codebase: `has_forward_context()` guard (mirrors `handle_attention_dsa`), `_resolve_attn_backend` for TBO unwrap (already in `forward_mha.py`), and conditional `_write_token_labels` hook (same pattern as the three existing call sites in `dsa_backend.py`). No new generalizable lesson.
