# FP8 GEMM Forward Path - Quick Reference

## Entry Point
```
Fp8LinearMethod.apply(x, weight, bias)
└─ Lines 442-498 in python/sglang/multimodal_gen/runtime/layers/quantization/fp8.py
```

---

## Main Flow (Block Quantization)

### Step 1: Activation Quantization
```python
q_input, x_scale = sglang_per_token_group_quant_fp8(
    input_2d,
    group_size=block_k,           # e.g., 128
    column_major_scales=True,
    scale_tma_aligned=True,
    scale_ue8m0=True,             # UE8M0 packed format
)
```
📍 **File:** `python/sglang/srt/layers/quantization/fp8_kernel.py:479-541`

**Allocations:**
- `q_input`: (M, K) FP8 quantized activation
- `x_scale`: Packed UE8M0 scales (Int32)

**Kernels Called:**
- `sgl_per_token_group_quant_8bit()` (C++ / JIT)
- `deep_gemm.transform_sf_into_required_layout()` (scale transformation)

---

### Step 2: DeepGEMM Block GEMM
```python
output = w8a8_block_fp8_matmul_deepgemm(
    q_input,      # FP8 (M, K)
    weight,       # FP8 (N, K) - pre-quantized at load time
    x_scale,      # Activation scales
    weight_scale, # Weight scales (per-block)
    block_size,   # [128, 128]
    output_dtype=torch.bfloat16,
)
```
📍 **File:** `python/sglang/srt/layers/quantization/fp8_kernel.py:1091-1106`

**Inside this function:**

#### 2a: Prepare & Allocate Output
```python
M, N, K, C = prepare_block_fp8_matmul_inputs(
    A, B, As, Bs, block_size, torch.bfloat16
)
# M = total tokens (M, K) from input
# N = output features (N, K) from weight
# K = input features
# C = NEW ALLOCATION: (M, N) tensor in BF16
```
📍 **Lines:** 1043-1088 in same file

#### 2b: Call DeepGEMM Kernel
```python
deep_gemm_fp8_fp8_bf16_nt(
    q_input,      # FP8 (M, K)
    x_scale,      # Activation scales
    weight,       # FP8 (N, K)
    weight_scale, # Weight scales
    C,            # Output BF16 (M, N) [pre-allocated]
)
```
📍 **Lines:** 113-120 in same file

This calls:
```python
deep_gemm_wrapper.gemm_nt_f8f8bf16(
    (q_input, x_scale),
    (weight, weight_scale),
    C
)
```
📍 **File:** `python/sglang/srt/layers/deep_gemm_wrapper/entrypoint.py:84-102`

Which finally calls:
```python
deep_gemm.fp8_gemm_nt(...)  # C++ library call
```

---

### Step 3: Add Bias & Reshape
```python
if bias is not None:
    output += bias

return output.view(*output_shape)  # Reshape to batch dims
```

---

## Key Tensor Shapes & Dtypes

| Variable | Shape | Dtype | Where Allocated |
|----------|-------|-------|-----------------|
| `x` (input) | (M, K) | BF16 | Input to forward |
| `q_input` | (M, K) | FP8 | `sglang_per_token_group_quant_fp8()` |
| `x_scale` | (M, K/128) or packed | Int32 or Float32 | `create_per_token_group_quant_fp8_output_scale()` |
| `weight` | (N, K) | FP8 | Pre-quantized at model load |
| `weight_scale` | (N/block_n, K/block_k) | Float32 or Int32 | Pre-loaded at model load |
| `C` (output) | (M, N) | BF16 | `prepare_block_fp8_matmul_inputs()` |

---

## Constraints

✅ **Requirements for DeepGEMM:**
- Output dtype = `torch.bfloat16`
- Weight shape: `N % 64 == 0` and `K % 128 == 0`
- Input must be contiguous

❌ **If constraints not met:**
- Falls back to `triton_w8a8_block_fp8_linear()`

---

## Comparison: BF16 Path

**No quantization needed!**
```python
# Inside UnquantizedLinearMethod.apply()
return F.linear(x, weight, bias)
# Computes: output = input @ weight.T + bias (all BF16)
```
📍 **File:** `python/sglang/multimodal_gen/runtime/layers/linear.py:152-160`

---

## `per_token_group_quant_fp8` Arguments

| Argument | Type | Default | Usage |
|----------|------|---------|-------|
| `x` | Tensor | - | Input to quantize (BF16) |
| `group_size` | int | - | Quantization group size (e.g., 128) |
| `column_major_scales` | bool | False | Scale layout for DeepGEMM |
| `scale_tma_aligned` | bool | False | TMA alignment for NVIDIA hardware |
| `scale_ue8m0` | bool | False | Pack scales in UE8M0 format |

**Returns:**
- `(q_input, x_scale)` where:
  - `q_input`: (M, K) FP8 quantized
  - `x_scale`: scales (packed if UE8M0)

---

## Actual DeepGEMM Kernel Computation

```
Input: (A_fp8, A_scales, B_fp8, B_scales, output_tensor)
Process:
  1. Dequantize A: A_dequant = A_fp8 * A_scales (per-group scaling)
  2. Dequantize B: B_dequant = B_fp8 * B_scales (per-block scaling)
  3. GEMM: C = A_dequant @ B_dequant^T
  4. Store in BF16 to output_tensor
Output: C (M, N) BF16
```

---

## File Hierarchy

```
fp8.py (entry: Fp8LinearMethod.apply)
  └─→ fp8_utils.py:deepgemm_w8a8_block_fp8_linear_with_fallback
       ├─→ fp8_kernel.py:sglang_per_token_group_quant_fp8 (quantize activation)
       │    ├─→ create_per_token_group_quant_fp8_output_scale (alloc scales)
       │    ├─→ sgl_per_token_group_quant_8bit (C++ kernel)
       │    └─→ deep_gemm.transform_sf_into_required_layout (scale xform)
       └─→ fp8_kernel.py:w8a8_block_fp8_matmul_deepgemm
            ├─→ prepare_block_fp8_matmul_inputs (alloc output C)
            └─→ fp8_kernel.py:deep_gemm_fp8_fp8_bf16_nt
                 └─→ entrypoint.py:gemm_nt_f8f8bf16
                      └─→ deep_gemm.fp8_gemm_nt ← C++ library
```

---

## Performance Tuning Notes

- **Block size:** Typically `[128, 128]`, divisible by hardware constraints
- **UE8M0 scales:** Packed format saves bandwidth; slightly slower unpacking
- **Triton fallback:** Used if shape constraints not met; slower than DeepGEMM
- **Quantization kernel:** Triton-based, runs per-token-group max computation

