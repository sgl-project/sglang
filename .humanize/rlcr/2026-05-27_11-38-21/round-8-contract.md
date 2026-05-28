# Round 8 Contract

## Mainline Objective
Close AC-7 for real: fix the production bypass to read `use_mha` from the active
`ForwardContext` backend (not from `ForwardBatch`), and wire the DS label write into
the MHA_ONE_SHOT KV-write path (`_set_mla_kv_buffer`).

## Target ACs
- AC-7: short-seq MHA bypass (production path + label write coverage)

## Blocking Issues
None outside the mainline gaps.

## Queued (out of scope this round)
- AC-4 calibration
- AC-5 TP test
- AC-6 graph capture
- All hardware/analyze tasks

## Implementation Tasks

### task-ac7-bypass-fix (coding)

**Fix 1 — `_select_topk_indices` production bypass** (`deepseek_v2.py`)
Replace the `getattr(forward_batch, "attn_backend", None)` path with:
```python
from sglang.srt.model_executor.forward_context import (
    get_attn_backend as _get_attn_backend,
    has_forward_context as _has_forward_context,
)
from sglang.srt.layers.attention.tbo_backend import TboAttnBackend as _TboAttnBackend
if _has_forward_context():
    _fc_backend = _get_attn_backend()
    if isinstance(_fc_backend, _TboAttnBackend):
        _fc_backend = _fc_backend.primary
    if hasattr(_fc_backend, "use_mha") and _fc_backend.use_mha:
        return None
```
Mirror exactly how `handle_attention_dsa` reads the decision. Remove `# AC-7` comment prefix.

**Fix 2 — label write in `_set_mla_kv_buffer`** (`forward_mha.py`)
`forward_mha.py` already imports `TboAttnBackend` and `_resolve_attn_backend` handles TBO unwrap.
Add after the KV pool write in EACH branch of `_set_mla_kv_buffer`:
```python
if getattr(self, "use_double_sparsity", False):
    _ds_backend = _resolve_attn_backend(forward_batch)
    if hasattr(_ds_backend, "_write_token_labels"):
        _ds_backend._write_token_labels(
            self.attn_mha, forward_batch.out_cache_loc, kv_a.unsqueeze(1)
        )
```
Actually, simpler to add once after all pool-write branches complete (not inside each if/elif/else).

**Tests** — update `TestAC7MHABypass`:
- Replace `attn_backend=SimpleNamespace(...)` on `forward_batch` with proper `forward_context(ForwardContext(attn_backend=mock_backend))` wrappers
- Add: `test_mha_label_write_fires_in_set_mla_kv_buffer` — under `forward_context`, call `_set_mla_kv_buffer`, assert `_write_token_labels` spy called
- Add: `test_decode_after_mha_prefill_calls_retrieve_topk` — ForwardContext with `use_mha=False`, assert retrieve_topk called

## Success Criteria
- All tests pass (176 + new tests)
- New tests run under real `forward_context(ForwardContext(attn_backend=...))` without synthetic `forward_batch.attn_backend`
- Removing the `has_forward_context()` guard OR the `use_mha` check would cause tests to fail
- Removing the `_write_token_labels` call from `_set_mla_kv_buffer` would cause tests to fail
