## PR Motivation

The MiniMax M2 model implementation contained incorrect function names for retrieving attention tensor parallelism (TP) group information. The code was using:
- `get_attention_tp_group` (incorrect)
- `get_attention_tp_world_size` (incorrect)

However, the correct function names in the distributed module are:
- `get_attn_tp_group` (correct)
- `get_attn_tensor_model_parallel_world_size` (correct)

This naming mismatch caused `AttributeError` when attempting to load or run the MiniMax M2 model, as the incorrectly named functions do not exist in the `sglang.srt.distributed` module.

This PR fixes the function names to align with the actual API in the distributed module, enabling the MiniMax M2 model to work correctly with tensor parallelism.

Fixes #20444

## PR Modifications

### 1. Fixed Function Names in MiniMax M2 Model (`python/sglang/srt/models/minimax_m2.py`)

**Import Statement Changes:**
```python
# Before (incorrect):
from sglang.srt.distributed import (
    get_attention_tp_group,           # ❌ Does not exist
    get_attention_tp_world_size,      # ❌ Does not exist
    ...
)

# After (correct):
from sglang.srt.distributed import (
    get_attn_tp_group,                # ✅ Correct name
    get_attn_tensor_model_parallel_world_size,  # ✅ Correct name
    ...
)
```

**Usage Changes:**
```python
# Before (incorrect):
self.attn_tp_size = get_attention_tp_world_size()
self.attn_tp_group = get_attention_tp_group()

# After (correct):
self.attn_tp_size = get_attn_tensor_model_parallel_world_size()
self.attn_tp_group = get_attn_tp_group()
```

## Affected Code Locations

| Line | Before | After |
|------|--------|-------|
| Import | `get_attention_tp_group` | `get_attn_tp_group` |
| Import | `get_attention_tp_world_size` | `get_attn_tensor_model_parallel_world_size` |
| Usage (line 552) | `get_attention_tp_world_size()` | `get_attn_tensor_model_parallel_world_size()` |
| Usage (line 553) | `get_attention_tp_group()` | `get_attn_tp_group()` |

## Behavior Changes

| Scenario | Before | After |
|----------|--------|-------|
| Loading MiniMax M2 model | `AttributeError` | ✅ Model loads successfully |
| Running MiniMax M2 with TP | Crash | ✅ Works correctly |
| Other models | Unaffected | Unaffected |

## Backward Compatibility

- **No breaking changes**: This is a pure bug fix that corrects function names to match the actual API
- **No API changes**: The corrected function names are the ones that have always been the correct API
- **No user-facing changes**: Users do not need to modify any code or configurations

## Testing

- [x] Verified MiniMax M2 model loads without `AttributeError`
- [x] Verified tensor parallelism works correctly with the corrected function names
- [x] Verified the functions exist in `sglang.srt.distributed` module
- [x] Verified no other models are affected by this change
- [x] Verified fix resolves issue #20444
