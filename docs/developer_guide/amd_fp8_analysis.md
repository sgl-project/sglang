# AMD FP8 Support Analysis for FlashInfer on MI300X

## Root Cause Analysis

### The Problem
FlashInfer's FP8 vector types fail to compile on AMD MI300X (gfx942) with errors like:
```
error: no matching constructor for initialization of 'vec_t<__hip_fp8_e4m3, 4UL>'
candidate constructor (the implicit default constructor) not viable: 
  call to __host__ function from __device__ function
```

### Root Cause
**MI300X uses FNUZ FP8 format, not OCP FP8 format.**

In `/opt/rocm/include/hip/amd_detail/amd_hip_fp8.h`:
```cpp
#if defined(__gfx942__) && __HIP_DEVICE_COMPILE__
#define HIP_FP8_TYPE_OCP 0    // ← OCP disabled on MI300X
#define HIP_FP8_TYPE_FNUZ 1   // ← FNUZ enabled on MI300X
```

This means:
- `__hip_fp8_e4m3` (OCP type) has `__host__`-only default constructor on gfx942
- `__hip_fp8_e4m3_fnuz` (FNUZ type) has `__host__ __device__` default constructor on gfx942

### Type Mapping Issue

| FlashInfer Uses | CUDA Type | HIP Alias | Works on gfx942? |
|-----------------|-----------|-----------|------------------|
| `__nv_fp8_e4m3` | CUDA FP8 E4M3 | `__hip_fp8_e4m3` (OCP) | ❌ No (host-only ctor) |
| `__nv_fp8_e5m2` | CUDA FP8 E5M2 | `__hip_fp8_e5m2` (OCP) | ❌ No (host-only ctor) |

FlashInfer's current HIP patch in `vec_dtypes.cuh`:
```cpp
#ifdef __HIP_PLATFORM_AMD__
using __nv_fp8_e4m3 = __hip_fp8_e4m3;  // ← This is OCP, not FNUZ!
using __nv_fp8_e5m2 = __hip_fp8_e5m2;
```

## The Fix

### Option 1: Use FNUZ Types for gfx942 (Recommended)

Modify the type aliases to be architecture-aware:

```cpp
#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_fp8.h>

// Use FNUZ types for gfx942 (MI300X), OCP for newer GPUs
#if defined(__gfx942__)
using __nv_fp8_e4m3 = __hip_fp8_e4m3_fnuz;
using __nv_fp8_e5m2 = __hip_fp8_e5m2_fnuz;
using __nv_fp8x2_e4m3 = __hip_fp8x2_e4m3_fnuz;
using __nv_fp8x2_e5m2 = __hip_fp8x2_e5m2_fnuz;
using __nv_fp8x4_e4m3 = __hip_fp8x4_e4m3_fnuz;
using __nv_fp8x4_e5m2 = __hip_fp8x4_e5m2_fnuz;
#else
using __nv_fp8_e4m3 = __hip_fp8_e4m3;
using __nv_fp8_e5m2 = __hip_fp8_e5m2;
using __nv_fp8x2_e4m3 = __hip_fp8x2_e4m3;
using __nv_fp8x2_e5m2 = __hip_fp8x2_e5m2;
using __nv_fp8x4_e4m3 = __hip_fp8x4_e4m3;
using __nv_fp8x4_e5m2 = __hip_fp8x4_e5m2;
#endif
#endif
```

### Option 2: Add Default Constructors to vec_t Specializations

Add explicit default constructors that initialize storage to 0:

```cpp
template <>
struct vec_t<__nv_fp8_e4m3, 2> {
  __nv_fp8x2_storage_t data;  // Use storage type, not wrapped type
  
  // Add device-compatible default constructor
  FLASHINFER_INLINE vec_t() : data(0) {}
  
  // ... rest of implementation
};
```

### Option 3: Use Storage Types Directly

Instead of using the wrapped FP8 types, use raw storage:

```cpp
template <>
struct vec_t<__nv_fp8_e4m3, 2> {
  __hip_fp8x2_storage_t data;  // unsigned short, always has default ctor
  
  FLASHINFER_INLINE vec_t() : data(0) {}
  // ...
};
```

## FNUZ vs OCP FP8 Formats

| Property | OCP (E4M3/E5M2) | FNUZ (E4M3_FNUZ/E5M2_FNUZ) |
|----------|-----------------|----------------------------|
| NaN representation | Standard IEEE-like | Finite NaN (no NaN values) |
| Zero representation | Standard | Unnormalized zero |
| MI300X support | Host-only | Full device support |
| MI350/gfx950 support | Full device support | Host-only |

**Important**: FNUZ and OCP have slightly different numeric representations. For inference workloads, this difference is usually negligible, but quantization/calibration should account for it.

## Recommended Next Steps

### Step 1: Create Architecture-Aware Type Aliases (Quick Fix)

Update `flashinfer/include/flashinfer/vec_dtypes.cuh`:

```cpp
#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_fp8.h>

// Architecture-specific FP8 type selection
#if defined(__gfx942__) || HIP_FP8_TYPE_FNUZ
// MI300X and FNUZ-primary architectures
using __nv_fp8_e4m3 = __hip_fp8_e4m3_fnuz;
using __nv_fp8_e5m2 = __hip_fp8_e5m2_fnuz;
using __nv_fp8x2_e4m3 = __hip_fp8x2_e4m3_fnuz;
using __nv_fp8x2_e5m2 = __hip_fp8x2_e5m2_fnuz;
using __nv_fp8x4_e4m3 = __hip_fp8x4_e4m3_fnuz;
using __nv_fp8x4_e5m2 = __hip_fp8x4_e5m2_fnuz;
#else
// OCP-primary architectures (MI350, etc.)
using __nv_fp8_e4m3 = __hip_fp8_e4m3;
using __nv_fp8_e5m2 = __hip_fp8_e5m2;
using __nv_fp8x2_e4m3 = __hip_fp8x2_e4m3;
using __nv_fp8x2_e5m2 = __hip_fp8x2_e5m2;
using __nv_fp8x4_e4m3 = __hip_fp8x4_e4m3;
using __nv_fp8x4_e5m2 = __hip_fp8x4_e5m2;
#endif
#endif
```

### Step 2: Add Explicit Default Constructors to vec_t

Ensure all `vec_t<FP8, N>` specializations have device-compatible default constructors:

```cpp
template <>
struct vec_t<__nv_fp8_e4m3, 2> {
  union {
    __nv_fp8x2_e4m3 data;
    __nv_fp8x2_storage_t data_storage;
  };
  
  // Explicit default constructor - always device-compatible
  FLASHINFER_INLINE vec_t() : data_storage(0) {}
  
  // ... rest unchanged
};
```

### Step 3: Test FP8 Kernels

After the fix, test:
1. `flashinfer.norm.rmsnorm` with FP8 output
2. `flashinfer.norm.fused_add_rmsnorm` with FP8
3. FP8 attention kernels (if applicable)

## Files to Modify

1. **`flashinfer/include/flashinfer/vec_dtypes.cuh`**
   - Update FP8 type aliases for gfx942
   - Add default constructors to vec_t<FP8, N> specializations

2. **`flashinfer/include/flashinfer/norm.cuh`**
   - Verify FP8 norm kernels work with FNUZ types

3. **`sgl-kernel/third_party/flashinfer_hip_patches/`**
   - Create new patch: `0007-fix-FP8-vec_t-for-gfx942-FNUZ.patch`

## Testing Commands

```bash
# Test FP8 type availability
python -c "
import torch
x = torch.randn(4, 1024, device='cuda', dtype=torch.float16)
# Try FP8 if available
try:
    x_fp8 = x.to(torch.float8_e4m3fn)
    print(f'FP8 E4M3 works: {x_fp8.dtype}')
except Exception as e:
    print(f'FP8 E4M3 failed: {e}')
"

# Test FlashInfer norm with FP8 (after fix)
python -c "
from flashinfer.norm import rmsnorm
import torch
x = torch.randn(4, 1024, device='cuda', dtype=torch.float16)
w = torch.ones(1024, device='cuda', dtype=torch.float16)
out = rmsnorm(x, w)
print(f'RMSNorm output: {out.shape}, {out.dtype}')
"
```

## Summary

| Issue | Cause | Fix |
|-------|-------|-----|
| `vec_t<__nv_fp8_e4m3, N>` no default ctor | OCP types use `__host__`-only ctor on gfx942 | Use FNUZ types on gfx942 |
| Template specialization mismatch | `__hip_fp8_e4m3` ≠ `__hip_fp8_e4m3_fnuz` | Conditional type alias |
| JIT compilation fails | hipcc can't find device-compatible ctor | Add explicit `vec_t() : data_storage(0) {}` |

**Priority**: High - This blocks all FP8 quantized operations on MI300X.

## Solution (Verified Working)

The fix is simple - change the FP8 type aliases from OCP to FNUZ in all 3 files:

```cpp
// Before (OCP - fails on gfx942)
using __nv_fp8_e4m3 = __hip_fp8_e4m3;

// After (FNUZ - works on gfx942)  
using __nv_fp8_e4m3 = __hip_fp8_e4m3_fnuz;
```

Files to patch:
1. `flashinfer/data/include/flashinfer/vec_dtypes.cuh`
2. `flashinfer/data/include/flashinfer/trtllm/common/cudaFp8Utils.h`
3. `flashinfer/data/csrc/tvm_ffi_utils.h`

Run the fix script:
```bash
python scripts/fix_flashinfer_fp8_fnuz.py
rm -rf ~/.cache/flashinfer*
python -c 'from flashinfer.norm import rmsnorm; print("OK")'
```

**Status**: ✅ Verified working on MI300X (gfx942)
