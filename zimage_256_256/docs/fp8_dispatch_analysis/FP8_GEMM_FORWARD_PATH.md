# FP8 GEMM Forward Path Analysis - SGLang Diffusion

## Overview
This document traces the exact FP8 GEMM forward path in the sglang-diffusion codebase, showing how FP8 linear layers execute and comparing with BF16 paths.

---

## 1. FP8 Linear Layer Forward Entry Point

**File:** `python/sglang/multimodal_gen/runtime/layers/quantization/fp8.py`

### Class: `Fp8LinearMethod`
Located at lines 159-498

The `apply()` method (lines 442-498) is the entry point for FP8 forward:

```python
def apply(
    self,
    layer: torch.nn.Module,
    x: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if self.use_marlin:
        return apply_fp8_marlin_linear(...)
    
    if self.block_quant:
        # BLOCK QUANTIZATION PATH (W8A8 with per-token-group activation quantization)
        if use_intel_amx_backend(layer):
            return torch.ops.sgl_kernel.fp8_scaled_mm_cpu(...)
        
        if isinstance(x, tuple):
            # x = (input, input_scale) already pre-quantized
            return self.w8a8_block_fp8_linear(
                input=x[0],
                weight=layer.weight,
                block_size=self.quant_config.weight_block_size,
                weight_scale=layer.weight_scale_inv,
                input_scale=x[1],
                bias=bias,
            )
        else:
            # x is raw input - quantize inside
            return self.w8a8_block_fp8_linear(
                input=x,
                weight=layer.weight,
                block_size=self.quant_config.weight_block_size,
                weight_scale=layer.weight_scale_inv,
                input_scale=None,
                bias=bias,
            )
    
    # NON-BLOCK QUANTIZATION PATH (per-tensor or per-channel)
    return apply_fp8_linear(
        input=x,
        weight=layer.weight,
        weight_scale=layer.weight_scale,
        input_scale=layer.input_scale,
        bias=bias,
        cutlass_fp8_supported=self.cutlass_fp8_supported,
        use_per_token_if_dynamic=False,
    )
```

---

## 2. Block Quantization Path (Primary Path)

### Handler: `w8a8_block_fp8_linear`

The `self.w8a8_block_fp8_linear` dispatcher is set during `__init__` (line 191):

```python
self.w8a8_block_fp8_linear = dispatch_w8a8_block_fp8_linear()
```

For DeepGEMM-enabled systems, this resolves to:

**Function:** `deepgemm_w8a8_block_fp8_linear_with_fallback()`

**File:** `python/sglang/srt/layers/quantization/fp8_utils.py`
**Lines:** 647-692

```python
def deepgemm_w8a8_block_fp8_linear_with_fallback(
    input: torch.Tensor,
    weight: torch.Tensor,
    block_size: List[int],
    weight_scale: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert input_scale is None
    
    output_dtype = input.dtype
    dtype_supported = output_dtype == torch.bfloat16  # CONSTRAINT: only BF16 output
    
    # DeepGEMM requires: shape[0] % 64 == 0 and shape[1] % 128 == 0
    shape_supported = weight.shape[0] % 64 == 0 and weight.shape[1] % 128 == 0
    
    if not (shape_supported and dtype_supported):
        # FALLBACK TO TRITON
        if weight_scale.dtype == torch.int32:
            weight_scale = _unpack_ue8m0_scale_for_triton(
                weight_scale, weight.shape, block_size
            )
        return triton_w8a8_block_fp8_linear(
            input, weight, block_size, weight_scale, input_scale, bias
        )
    
    # ==================== DEEPGEMM PATH ====================
    
    # Reshape input to 2D
    input_2d = input.view(-1, input.shape[-1])
    output_shape = [*input.shape[:-1], weight.shape[0]]
    
    # STEP 1: QUANTIZE ACTIVATION
    # =================================================
    q_input, x_scale = sglang_per_token_group_quant_fp8(
        input_2d,
        block_size[1],  # group_size = block_k (typically 128)
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,  # Uses UE8M0 packed format
    )
    # Returns:
    #   q_input: FP8 quantized activation, shape (M, K)
    #   x_scale: activation scales in UE8M0 packed format
    
    # STEP 2: DEEPGEMM GEMM
    # =================================================
    output = w8a8_block_fp8_matmul_deepgemm(
        q_input, weight, x_scale, weight_scale, block_size, output_dtype=output_dtype
    )
    
    # STEP 3: ADD BIAS (if provided)
    # =================================================
    if bias is not None:
        output += bias
    
    # Reshape output to match input shape
    return output.to(dtype=output_dtype).view(*output_shape)
```

---

## 3. Activation Quantization Function

**Function:** `sglang_per_token_group_quant_fp8()`

**File:** `python/sglang/srt/layers/quantization/fp8_kernel.py`
**Lines:** 479-541

```python
def sglang_per_token_group_quant_fp8(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    column_major_scales: bool = False,
    scale_tma_aligned: bool = False,
    scale_ue8m0: bool = False,
    fuse_silu_and_mul: bool = False,
    masked_m: Optional[torch.Tensor] = None,
    enable_v2: Optional[bool] = None,
):
    assert (
        x.shape[-1] % group_size == 0
    ), "the last dimension of `x` cannot be divisible by `group_size`"
    assert x.is_contiguous(), "`x` is not contiguous"
    
    out_shape = (*x.shape[:-1], x.shape[-1] // (2 if fuse_silu_and_mul else 1))
    
    # TENSOR ALLOCATION #1: Quantized activation
    x_q = torch.empty(out_shape, device=x.device, dtype=fp8_dtype)
    
    # TENSOR ALLOCATION #2: Activation scales (UE8M0 packed or float32)
    x_s = create_per_token_group_quant_fp8_output_scale(
        x_shape=out_shape,
        device=x.device,
        group_size=group_size,
        column_major_scales=column_major_scales,
        scale_tma_aligned=scale_tma_aligned,
        scale_ue8m0=scale_ue8m0,
    )
    
    if x.shape[0] > 0:
        # Dispatch to appropriate quantization kernel
        if enable_sgl_per_token_group_quant_8bit:
            if enable_v2:
                sgl_per_token_group_quant_8bit(...)  # C++ kernel
            else:
                sgl_per_token_group_quant_8bit_jit(...)  # JIT kernel
        else:
            # Triton kernels (fallback)
            sgl_per_token_group_quant_fp8(...)
    
    # Handle UE8M0 scale transformation for DeepGEMM
    if scale_ue8m0:
        from deep_gemm import transform_sf_into_required_layout
        x_s = transform_sf_into_required_layout(
            x_s,
            num_groups=None,
            mn=x_q.shape[0],
            k=x_q.shape[1],
            recipe=(1, group_size, group_size),
            is_sfa=True,
        )
    
    return x_q, x_s
```

### Scale Output Tensor Details

**Function:** `create_per_token_group_quant_fp8_output_scale()`
**Lines:** 435-476

For UE8M0 mode (DeepGEMM):
```python
if scale_ue8m0:
    # UE8M0 packed format: 4 uint8 values packed into int32
    *x_batch, x_q_mn, x_q_k = x_shape
    x_s_mn, x_s_k = x_q_mn, x_q_k // 128
    aligned_mn = ceil_align(x_s_mn, 4)
    aligned_k = ceil_align(x_s_k, 4)
    
    # Shape: (M_groups, K_groups/4) with dtype int32
    return torch.empty(
        (*x_batch, aligned_k // 4, aligned_mn),
        device=device,
        dtype=torch.int,
    ).transpose(-1, -2)[..., :x_s_mn, :]
```

---

## 4. DeepGEMM GEMM Execution

**Function:** `w8a8_block_fp8_matmul_deepgemm()`

**File:** `python/sglang/srt/layers/quantization/fp8_kernel.py`
**Lines:** 1091-1106

```python
def w8a8_block_fp8_matmul_deepgemm(
    A: torch.Tensor,      # Quantized activation (M, K) in FP8
    B: torch.Tensor,      # Quantized weight (N, K) in FP8
    As: torch.Tensor,     # Activation scales (per-token-group)
    Bs: torch.Tensor,     # Weight scales (per-block)
    block_size: List[int],
    output_dtype: torch.dtype,
) -> torch.Tensor:
    # Prepare inputs and allocate output
    M, N, K, C = prepare_block_fp8_matmul_inputs(A, B, As, Bs, block_size, output_dtype)
    # M = total tokens (flattened batch)
    # N = output feature dimension
    # K = input feature dimension
    # C = output tensor, shape (M, N) with dtype=BF16
    
    # DeepGEMM only supports BF16 output
    assert C.dtype == torch.bfloat16 and deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
    
    # ==================== DEEPGEMM KERNEL CALL ====================
    deep_gemm_fp8_fp8_bf16_nt(A, As, B, Bs, C)
    
    return C
```

### Output Tensor Allocation

**Function:** `prepare_block_fp8_matmul_inputs()`
**Lines:** 1043-1088

```python
def prepare_block_fp8_matmul_inputs(
    A: torch.Tensor,      # Input (M, K) FP8
    B: torch.Tensor,      # Weight (N, K) FP8
    As: torch.Tensor,     # Activation scales
    Bs: torch.Tensor,     # Weight scales
    block_size: List[int],
    output_dtype: torch.dtype = torch.float16,
) -> Tuple[int, int, int, torch.Tensor]:
    
    assert len(block_size) == 2
    block_n, block_k = block_size[0], block_size[1]
    
    assert A.shape[-1] == B.shape[-1]
    assert A.is_contiguous()
    
    M = A.numel() // A.shape[-1]  # Total batch size
    
    assert B.ndim == 2
    assert B.is_contiguous()
    assert Bs.ndim == 2
    N, K = B.shape  # N = output features, K = input features
    
    # ==================== OUTPUT TENSOR ALLOCATION ====================
    C_shape = A.shape[:-1] + (N,)
    C = A.new_empty(C_shape, dtype=output_dtype)  # ALLOCATE: (M, N) in BF16
    
    return M, N, K, C
```

---

## 5. DeepGEMM Wrapper Kernel

**Function:** `deep_gemm_fp8_fp8_bf16_nt()`

**File:** `python/sglang/srt/layers/quantization/fp8_kernel.py`
**Lines:** 113-120

```python
def deep_gemm_fp8_fp8_bf16_nt(
    A: torch.Tensor,      # Quantized activation (M, K) FP8
    As: torch.Tensor,     # Activation scales
    B: torch.Tensor,      # Quantized weight (N, K) FP8
    Bs: torch.Tensor,     # Weight scales (per-block)
    C: torch.Tensor,      # Output (M, N) BF16 [ALLOCATED BY CALLER]
) -> None:
    # Delegates to DeepGEMM C++ library
    deep_gemm_wrapper.gemm_nt_f8f8bf16((A, As), (B, Bs), C)
```

### DeepGEMM Wrapper Call

**File:** `python/sglang/srt/layers/deep_gemm_wrapper/entrypoint.py`
**Lines:** 84-102

```python
def gemm_nt_f8f8bf16(
    lhs: Tuple[torch.Tensor, torch.Tensor],  # (A, As) activation + scales
    rhs: Tuple[torch.Tensor, torch.Tensor],  # (B, Bs) weight + scales
    out: torch.Tensor,                        # C output tensor
):
    m, k = lhs[0].shape  # (M, K)
    n, _ = rhs[0].shape  # (N, K)
    num_groups = 1
    kernel_type = compile_utils.DeepGemmKernelType.GEMM_NT_F8F8BF16
    
    _sanity_check_input(lhs)
    _sanity_check_input(rhs)
    
    with compile_utils.deep_gemm_execution_hook(m, n, k, num_groups, kernel_type):
        # ==================== ACTUAL KERNEL CALL ====================
        deep_gemm.fp8_gemm_nt(
            lhs,  # (A_fp8, A_scales)
            rhs,  # (B_fp8, B_scales)
            out,  # C_bf16 (pre-allocated)
        )
```

**Signature (from deep_gemm library):**
```
deep_gemm.fp8_gemm_nt(
    lhs: Tuple[Tensor, Tensor],  # (A_fp8, A_scales)
    rhs: Tuple[Tensor, Tensor],  # (B_fp8, B_scales)
    out: Tensor,                   # C_bf16
)
# Computes: C = (A * A_scales) @ (B * B_scales)^T in BF16 arithmetic
```

---

## 6. Comparison: BF16 Linear Path (No Quantization)

**Class:** `UnquantizedLinearMethod`

**File:** `python/sglang/multimodal_gen/runtime/layers/linear.py`
**Lines:** 127-160

```python
class UnquantizedLinearMethod(LinearMethodBase):
    """Linear method without quantization."""
    
    def apply(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        # Simple PyTorch linear operation
        output = (
            F.linear(x, layer.weight, bias)
            if current_platform.is_amp_supported() or bias is None
            else F.linear(x, layer.weight, bias.to(x.dtype))
        )
        return output
```

**PyTorch Function:** `F.linear()`
- Signature: `F.linear(input, weight, bias=None)`
- Computes: `output = input @ weight.T + bias`
- No quantization, no scale computations
- All operations in native dtype (BF16)

---

## 7. Data Flow Summary

### FP8 Path (Block Quantization)

```
Input (BF16, shape M×K)
        ↓
[sglang_per_token_group_quant_fp8]
        ↓
        ├─→ Allocate: q_input (FP8, M×K)
        ├─→ Allocate: x_scale (UE8M0 packed)
        ├─→ Quantization kernel
        │   (per-token-group scales)
        └─→ Transform scales to DeepGEMM layout
        ↓
(q_input: FP8, x_scale: UE8M0)
        ↓
[w8a8_block_fp8_matmul_deepgemm]
        ↓
        ├─→ prepare_block_fp8_matmul_inputs
        │   ├─→ Allocate: C (BF16, M×N)
        │   └─→ Return: M, N, K, C
        ↓
[deep_gemm_fp8_fp8_bf16_nt]
        ↓
[deep_gemm.fp8_gemm_nt]  ← C++ Library Call
        ↓
Output (BF16, M×N)
        ├─→ Add bias (if provided)
        └─→ Reshape to original batch shape
```

### BF16 Path (No Quantization)

```
Input (BF16, M×K)
        ↓
[F.linear(x, weight, bias)]
        ↓
Output (BF16, M×N)
```

---

## 8. Tensor Allocations in FP8 Forward Path

| Tensor | Shape | Dtype | Location | Purpose |
|--------|-------|-------|----------|---------|
| `q_input` | (M, K) | FP8 | GPU | Quantized activation |
| `x_scale` | Varies | Int32/Float32 | GPU | Activation scales (UE8M0 if DeepGEMM) |
| `C` | (M, N) | BF16 | GPU | Output tensor (allocated in `prepare_block_fp8_matmul_inputs`) |
| `output` | (M, N) | BF16 | GPU | Same as C, returned after bias add |

---

## 9. Key Functions Reference

| Function | File | Lines | Purpose |
|----------|------|-------|---------|
| `Fp8LinearMethod.apply()` | `fp8.py` | 442-498 | Entry point for FP8 forward |
| `deepgemm_w8a8_block_fp8_linear_with_fallback()` | `fp8_utils.py` | 647-692 | Block quant dispatcher |
| `sglang_per_token_group_quant_fp8()` | `fp8_kernel.py` | 479-541 | Activation quantization |
| `create_per_token_group_quant_fp8_output_scale()` | `fp8_kernel.py` | 435-476 | Scale allocation logic |
| `w8a8_block_fp8_matmul_deepgemm()` | `fp8_kernel.py` | 1091-1106 | DeepGEMM wrapper |
| `prepare_block_fp8_matmul_inputs()` | `fp8_kernel.py` | 1043-1088 | Output tensor allocation |
| `deep_gemm_fp8_fp8_bf16_nt()` | `fp8_kernel.py` | 113-120 | DeepGEMM kernel call |
| `gemm_nt_f8f8bf16()` | `entrypoint.py` | 84-102 | DeepGEMM execution hook |
| `UnquantizedLinearMethod.apply()` | `linear.py` | 152-160 | BF16 reference path |

---

## 10. Key Parameters

### Block Quantization Configuration
- **block_size:** `[block_n, block_k]` - typically `[128, 128]` or similar
- **group_size:** Same as `block_k` (typically 128)
- **Output dtype:** Must be `torch.bfloat16` for DeepGEMM

### Constraints
- Input must be contiguous
- Weight shape must be divisible: `shape[0] % 64 == 0 and shape[1] % 128 == 0`
- Output always BF16 for DeepGEMM path

### Scale Formats
- **Regular scales:** Float32, shape varies by quantization strategy
- **UE8M0 packed scales:** Int32, packed 4 uint8 values per int32

---

## 11. Quantization Kernel Details

### Per-Token-Group Quantization

Inside the Triton kernel `_per_token_group_quant_8bit_colmajor()` (lines 167-212):

```python
# For each token group g_id:
y = load_group(x, g_id)  # Load group_size elements
_absmax = max(abs(y))
y_s = _absmax / fp8_max   # Scale = max value / max representable
y_q = clamp(y / y_s, fp8_min, fp8_max).to(fp8)  # Quantize

# Store quantized values and scale
store(x_q, y_q)
store(x_s, y_s)
```

For UE8M0 format, scale is further transformed:
```python
if SCALE_UE8M0:
    y_s = exp2(ceil(log2(abs(y_s))))  # Round to power-of-2
```

---

## 12. Command Flow Example

```python
# User code (diffusion model forward)
x = model(input)  # BF16 input

# Inside linear layer forward
output, bias = linear_layer(x)

# Inside Fp8LinearMethod.apply()
# 1. Check: block_quant=True, input_scale=None
# 2. Call: w8a8_block_fp8_linear(x, weight, block_size, weight_scale_inv)

# Inside deepgemm_w8a8_block_fp8_linear_with_fallback()
# 3. Check: output_dtype=BF16 ✓, shape constraints ✓
# 4. Call: sglang_per_token_group_quant_fp8(
#       x, block_size[1], 
#       column_major_scales=True,
#       scale_tma_aligned=True,
#       scale_ue8m0=True,
#    )
# 5. Dispatch: sgl_per_token_group_quant_8bit (C++ kernel or JIT)
# 6. Transform scales: deep_gemm.transform_sf_into_required_layout()

# Inside w8a8_block_fp8_matmul_deepgemm()
# 7. Call: prepare_block_fp8_matmul_inputs() → allocates C (BF16)
# 8. Call: deep_gemm_fp8_fp8_bf16_nt(q_input, x_scale, weight, weight_scale, C)

# Inside gemm_nt_f8f8bf16()
# 9. Execution hook setup
# 10. Call: deep_gemm.fp8_gemm_nt((q_input, x_scale), (weight, weight_scale), C)

# Return
# 11. Add bias: C += bias
# 12. Reshape: C.view(*output_shape)
# 13. Return C (BF16)
```

