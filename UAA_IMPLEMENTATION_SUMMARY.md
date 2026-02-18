# UAA (Ulysses Anything Attention) Implementation Summary

This document summarizes the UAA implementation in SGLang, enabling USP (Unified Sequence Parallelism) to work with arbitrary head counts and sequence lengths.

## Overview

UAA is an experimental feature that allows Ulysses sequence parallelism to work with models that have head counts not evenly divisible by the world size. Based on the [diffusers PR #12996](https://github.com/huggingface/diffusers/pull/12996).

## Changes Made

### 1. Core UAA Functions (`python/sglang/multimodal_gen/runtime/layers/usp.py`)

**Added two new functions:**

- `_usp_split_for_ulysses(x, dim, world_size)`: Splits tensor along dimension using `tensor_split()`, which correctly handles non-divisible sizes (unlike `chunk()`).
- `_usp_gather_for_ulysses(x, dim, world_size)`: Gathers tensors from all ranks along dimension, handling varying sizes per rank.

**Modified existing functions:**

- `_usp_input_all_to_all()`: Added `enable_uaa` parameter. When enabled, uses split+gather pattern instead of all-to-all.
- `_usp_output_all_to_all()`: Added `enable_uaa` parameter for the inverse operation.

### 2. USPAttention Layer (`python/sglang/multimodal_gen/runtime/layers/attention/layer.py`)

**Modified `USPAttention` class:**

- Added `enable_uaa` parameter to `__init__()` with auto-fetching from `server_args`
- Added validation warning when UAA is used with `ring_degree > 1`
- Updated `forward()` to pass `enable_uaa` flag to all-to-all functions
- Added experimental warning for UAA + ring attention combination

### 3. Server Arguments (`python/sglang/multimodal_gen/runtime/server_args.py`)

**Added UAA configuration:**

- New field: `enable_uaa: bool = False` in `ServerArgs` dataclass
- Validation in `_validate_parallelism()` to warn about UAA + ring_degree > 1
- CLI argument: `--enable-uaa` with help text describing experimental nature

### 4. Environment Variables (`python/sglang/multimodal_gen/envs.py`)

**Added environment variable:**

- `SGLANG_ENABLE_UAA`: Boolean environment variable to enable UAA globally

## Usage

### Command Line

```bash
# Enable UAA for arbitrary GPU configurations
sglang generate \
  --model-path FLUX.1-dev \
  --prompt "A beautiful sunset" \
  --num-gpus 3 \
  --ulysses-degree 3 \
  --enable-uaa
```

### Environment Variable

```bash
export SGLANG_ENABLE_UAA=true
sglang generate --model-path FLUX.1-dev --num-gpus 3 --ulysses-degree 3
```

## Key Implementation Details

### Use of `tensor_split()` vs `chunk()`

The implementation uses `torch.tensor_split()` instead of `torch.chunk()` because:
- `chunk()` may return fewer chunks than specified when size isn't divisible
- `tensor_split()` always returns the requested number of chunks with varying sizes

### Split + Gather Pattern

Instead of traditional all-to-all communication, UAA uses:
- **Input**: Split heads locally → Gather sequences from all ranks
- **Output**: Split sequences locally → Gather heads from all ranks

This avoids the need for padding and is more memory-efficient.

### Automatic Configuration

The `USPAttention` layer automatically fetches `enable_uaa` from `server_args` if not explicitly provided, making it transparent to model code.

## Known Limitations

1. **Ring Attention**: UAA with `ring_degree > 1` is experimental and may not work correctly
2. **Performance**: No fast path optimization for evenly divisible cases yet
3. **LRU Caching**: Functions depending on tensor sizes should avoid caching

## Testing Recommendations

1. Test with uneven head counts (e.g., 25 heads on 3 GPUs)
2. Compare output with/without UAA on evenly divisible cases
3. Test USP combination (Ulysses + Ring)
4. Verify no regressions when UAA is disabled

## Future Work

- Add auto-detection (enable when needed)
- Implement fast path for evenly divisible cases
- Add comprehensive tests
- Full support for UAA + ring attention
- Performance benchmarking and optimization
- Async communication overlap

## References

- Diffusers UAA PR: https://github.com/huggingface/diffusers/pull/12996
- SGLang USP code: `/python/sglang/multimodal_gen/runtime/layers/usp.py`
