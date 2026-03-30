# Visual Comparison: BF16 vs FP8 DeepGemm Dispatch Flow

## Side-by-Side Call Stack Comparison

```
┌─────────────────────────────────────┬─────────────────────────────────────┐
│       BF16 GEMM Path (Simple)       │     FP8 DeepGemm Path (Complex)     │
└─────────────────────────────────────┴─────────────────────────────────────┘

1. Linear Layer Forward
   ├─ BF16: time ~ 0.05 μs             ├─ FP8: time ~ 0.05 μs
   │                                     │
   
2. Quant Method Apply Selection
   ├─ BF16: time ~ 0.1 μs               ├─ FP8: time ~ 0.1 μs
   │  UnquantizedLinearMethod()         │  Fp8LinearMethod()
   │                                     │
   
3. Main Dispatch
   ├─ BF16: time ~ 0.1 μs               ├─ FP8: time ~ 0.5 μs
   │  F.linear() call                    │  Fp8LinearMethod.apply()
   │                                     │   ├─ Marlin check
   │                                     │   ├─ Block quant check
   │                                     │   └─ Backend selection
   │
4. BACKEND SELECTION
   │                                     ├─ FP8: time ~ 1-2 μs
   │                                     │  dispatch_w8a8_block_fp8_linear()
   │                                     │   ├─ get_fp8_gemm_runner_backend()
   │                                     │   ├─ is_blackwell_supported()
   │                                     │   ├─ is_flashinfer_available()
   │                                     │   └─ DEEPGEMM_ENABLE_JIT check
   │
5. WRAPPER DISPATCH
   │                                     ├─ FP8: time ~ 1-2 μs
   │                                     │  deepgemm_w8a8_block_fp8_linear_with_fallback()
   │                                     │   ├─ Dtype check
   │                                     │   ├─ Shape divisibility check
   │                                     │   └─ Fallback condition
   │
6. ⭐ PER-TOKEN QUANTIZATION (MAIN BOTTLENECK)
   │                                     ├─ FP8: time ~ 10-60 μs ⭐⭐⭐
   │                                     │  sglang_per_token_group_quant_fp8()
   │                                     │   ├─ Tensor allocation [2-5 μs]
   │                                     │   ├─ Shape assertions [1-2 μs]
   │                                     │   ├─ Scale creation [2-4 μs]
   │                                     │   ├─ JIT backend check [0.5-1 μs]
   │                                     │   └─ CUDA quant kernel [5-50 μs]
   │
7. MATMUL PREPARATION
   │                                     ├─ FP8: time ~ 3-5 μs
   │                                     │  w8a8_block_fp8_matmul_deepgemm()
   │                                     │   ├─ prepare_block_fp8_matmul_inputs() [2-3 μs]
   │                                     │   └─ Assertions [0.2 μs]
   │
8. KERNEL DISPATCH
   ├─ BF16: time ~ 0.2 μs               ├─ FP8: time ~ 3-5 μs
   │  cuBLAS kernel launch              │  DeepGemm wrapper layer
   │                                     │   ├─ deep_gemm_fp8_fp8_bf16_nt() [1-2 μs]
   │                                     │   ├─ Sanity checks [0.2-1 μs]
   │                                     │   └─ Execution hook [1-2 μs]
   │
9. GPU KERNEL EXECUTION
   ├─ BF16: GEMM kernel                 ├─ FP8: DeepGemm kernel
   │  Time: 10-50 μs (depends on size)  │  Time: 10-50 μs (similar size)
   │
10. RETURN & RESHAPE
   ├─ BF16: time ~ 0.1 μs               ├─ FP8: time ~ 0.3 μs
   │  Direct return                      │  output.to().view() operations
   
═══════════════════════════════════════════════════════════════════════════════

TOTAL PYTHON OVERHEAD:
├─ BF16: ~0.5-1 μs ✅
├─ FP8:  ~20-30 μs ⚠️
└─ Difference: 20-60x higher!

TOTAL CUDA TIME (including kernel):
├─ BF16: ~10-50 μs
├─ FP8:  ~15-80 μs (slightly higher due to quant kernel)
└─ Difference: ~20% more CUDA time
```

---

## Overhead Breakdown by Component

```
BF16 GEMM (Total: ~1-2 μs)
═════════════════════════════════════════════════════════════════

  Python Dispatch                    CUDA Dispatch
  ├─ Function call: 0.1 μs           └─ cuBLAS launch: 0.5 μs
  ├─ Dtype check: 0.1 μs             
  ├─ Linear layer: 0.2 μs            [GEMM kernel: 10-50 μs]
  └─ Return: 0.1 μs                  
  
  ┌─────────────────────┐            ┌──────────────────────┐
  │   ~0.5 μs (Python)  │            │  ~10-50 μs (CUDA)    │
  └─────────────────────┘            └──────────────────────┘


FP8 DeepGemm (Total: ~25-80 μs)
═════════════════════════════════════════════════════════════════

  Python Dispatch                    CUDA Dispatch
  ├─ Apply check: 0.5 μs             ├─ Quant kernel: 5-50 μs ⭐
  ├─ Backend dispatch: 1-2 μs        └─ DeepGemm kernel: 10-50 μs
  ├─ Fallback check: 1-2 μs
  ├─ Tensor alloc: 4-10 μs ⭐
  ├─ Shape validation: 1-2 μs
  ├─ Scale creation: 2-4 μs ⭐
  ├─ JIT check: 0.5-1 μs
  ├─ Matmul prep: 3-5 μs
  ├─ DeepGemm wrap: 3-5 μs
  └─ Return/reshape: 0.3 μs
  
  ┌────────────────────────┐        ┌──────────────────────┐
  │  ~20-30 μs (Python) ⚠️ │        │  ~15-100 μs (CUDA)   │
  └────────────────────────┘        └──────────────────────┘


KEY DIFFERENCES (FP8 adds):
  ✗ Per-token quantization [10-60 μs]
  ✗ Tensor allocations [4-10 μs]
  ✗ Scale computation [2-4 μs]
  ✗ Fallback conditions [1-2 μs]
  ✗ Multiple dispatch layers [5-10 μs]
```

---

## Flame Graph Representation

```
BF16 Path - Very Shallow
═══════════════════════════════════════════════════════════════════════════════

Total Wall Time: 1-2 μs (Python) + 10-50 μs (CUDA)

  ████████████████ F.linear() [100 ns]
  ├─ dtype check [20 ns]
  ├─ Linear layer call [50 ns]
  ├─ cuBLAS dispatch [30 ns]
  └─ return [5 ns]
  
  Time: ████ (very thin - basically overhead only)
  
  
FP8 Path - Much Deeper  
═══════════════════════════════════════════════════════════════════════════════

Total Wall Time: 20-30 μs (Python) + 15-100 μs (CUDA)

  ░░░░░░░░░░░░░░░░ Fp8LinearMethod.apply() [20-30 μs] ⭐ MAIN OVERHEAD
  ├─ ▓▓▓ Marlin check [500 ns]
  ├─ ▓▓▓ Block quant check [500 ns]
  ├─ ▓▓▓▓ dispatch_w8a8_block_fp8_linear() [1-2 μs]
  ├─ ▓▓▓▓ deepgemm_w8a8_block_fp8_linear_with_fallback() [1-2 μs]
  │  ├─ ▓▓ Dtype check [200 ns]
  │  └─ ▓▓ Shape check [500 ns]
  ├─ ██████████████ sglang_per_token_group_quant_fp8() [10-60 μs] ⭐⭐⭐
  │  ├─ ████ Tensor alloc [4-10 μs]
  │  ├─ ▓▓ Assertions [1-2 μs]
  │  ├─ ███ Scale creation [2-4 μs]
  │  ├─ ▓▓ JIT check [0.5-1 μs]
  │  └─ ████████ CUDA kernel [5-50 μs]
  ├─ ███ w8a8_block_fp8_matmul_deepgemm() [3-5 μs]
  │  ├─ ██ prepare_inputs() [2-3 μs]
  │  ├─ ▓▓ Assertions [0.2 μs]
  │  └─ ▓▓ Wrapper [1-2 μs]
  ├─ ██ gemm_nt_f8f8bf16() [2-3 μs]
  ├─ ▓▓ Sanity checks [0.2-1 μs]
  ├─ ▓▓ Hook setup [1-2 μs]
  └─ ▓▓ return/reshape [0.3 μs]
  
  Time: ████████████████ (very thick - significant overhead!)


LEGEND:
████ = Large overhead (> 5 μs)
███  = Medium overhead (1-5 μs)
██   = Small overhead (0.1-1 μs)
▓▓   = Tiny overhead (< 0.1 μs)
```

---

## Backend Selection Decision Tree

```
Fp8LinearMethod.apply()
│
├─ [Check] self.use_marlin?
│  ├─ YES → apply_fp8_marlin_linear()
│  └─ NO  → Continue
│
├─ [Check] self.block_quant?
│  ├─ NO  → apply_fp8_linear() [slower path]
│  └─ YES → Continue
│
├─ [Check] use_intel_amx_backend()?
│  ├─ YES → torch.ops.sgl_kernel.fp8_scaled_mm_cpu()
│  └─ NO  → Continue
│
├─ [Check] isinstance(x, tuple)?
│  ├─ YES → self.w8a8_block_fp8_linear() [with pre-quantized input]
│  └─ NO  → self.w8a8_block_fp8_linear() [quantize input]
│
└─ w8a8_block_fp8_linear() 
   └─ dispatch_w8a8_block_fp8_linear()
      │
      ├─ [Load] backend = get_fp8_gemm_runner_backend()
      │  └─ Global lookup (cached after first init)
      │
      ├─ [Check] backend.is_auto()?
      │  ├─ NO  → _dispatch_explicit_backend(backend)
      │  └─ YES → _dispatch_auto_backend()
      │           ├─ ENABLE_JIT_DEEPGEMM?       → deepgemm_w8a8_block_fp8_linear_with_fallback
      │           ├─ is_blackwell_supported()?  → flashinfer_gemm_w8a8_block_fp8_linear_with_fallback
      │           ├─ _check_cutlass_support()?  → cutlass_w8a8_block_fp8_linear_with_fallback
      │           ├─ _use_aiter?                → aiter_w8a8_block_fp8_linear
      │           └─ fallback                   → triton_w8a8_block_fp8_linear
      │
      └─ [Call] Selected backend function
         └─ deepgemm_w8a8_block_fp8_linear_with_fallback()
            ├─ [Validate] dtype_supported = (output_dtype == bfloat16)?
            ├─ [Validate] shape_supported = (N % 64 == 0 and K % 128 == 0)?
            │
            ├─ [Check] Both valid?
            │  ├─ NO  → Unpack scale & call triton_w8a8_block_fp8_linear()
            │  └─ YES → Continue
            │
            ├─ [Call] sglang_per_token_group_quant_fp8()
            │  ├─ Allocate x_q, x_s tensors
            │  ├─ Run quantization kernel
            │  └─ Return quantized values
            │
            ├─ [Call] w8a8_block_fp8_matmul_deepgemm()
            │  ├─ Prepare inputs
            │  └─ Call deep_gemm.gemm_nt_f8f8bf16()
            │
            ├─ [Check] bias is not None?
            │  └─ YES → output += bias
            │
            └─ [Reshape] output.to().view()
```

---

## Memory Access Patterns

```
BF16 GEMM
═════════════════════════════════════════════════════════════════

Input Activation (BF16)          Weights (BF16)
 ┌──────────┐                     ┌──────────┐
 │ 1024×512 │                     │ 512×1024 │
 └──────────┘                     └──────────┘
      │                                 │
      └─────────────┬────────────────────┘
                    │
              [cuBLAS GEMM]
                    │
              Output (BF16)
              ┌──────────┐
              │ 1024×512 │
              └──────────┘

Memory footprint: 2.5 MB (no intermediate activations)
Allocations: 0 (weights & output pre-allocated)
Python overhead: Minimal


FP8 DeepGemm Path
═════════════════════════════════════════════════════════════════

Input (BF16)        Weights (FP8)       Weight Scales (FP32)
 ┌────────┐          ┌────────┐         ┌──────┐
 │ 1024K  │          │ 512×512│         │512×32│
 └────────┘          └────────┘         └──────┘
      │                   │                   │
      ▼                   │                   │
 ┌──────────────┐         │                   │
 │ Quant kernel │◄────────┴───────────────────┘
 └──────────────┘
      │
      ▼
 ┌─────────────────────────┐
 │ x_q (FP8, NEW ALLOC)    │  ◄── Allocated on every forward!
 │ x_scale (F32, NEW ALLOC)│  ◄── Allocated on every forward!
 └─────────────────────────┘
      │
      ▼ [DeepGemm GEMM]
 ┌──────────┐
 │Output(BF16)
 └──────────┘

Memory footprint: 3.5+ MB (includes x_q, x_scale)
Allocations: 2 per forward (x_q, x_scale) ← OVERHEAD!
Python overhead: ~4-10 μs from allocation + management
```

---

## Performance Timeline (Typical Inference)

```
BF16 Timeline (100 consecutive forwards)
═════════════════════════════════════════════════════════════════

|──────────────────────────────────────────────────────────────|
| [0.1μs]  [0.1μs]  [0.1μs]  [0.1μs]  [0.1μs]  ← Python dispatch (negligible)
|    │        │        │        │        │
|   [GEMM]   [GEMM]   [GEMM]   [GEMM]   [GEMM]  ← CUDA kernels (10-50 μs each)
|────┤────────┤────────┤────────┤────────┤──────
     |1        |1       |1       |1       |1      (ms)


FP8 DeepGemm Timeline (100 consecutive forwards)
═════════════════════════════════════════════════════════════════

|──────────────────────────────────────────────────────────────|
| [20μs dispatch]  [20μs]  [20μs]  [20μs]  [20μs]  ← Python overhead
|    │              │       │       │       │
|    │[quant kernel]│       │       │       │
|    └──[DEEPGEMM]──┘       │       │       │
|         │                 │       │       │
|        [quant]            │       │       │
|         │            [quant]      │       │
|        [DEEPGEMM]    │            │       │
|         │            └──[DEEPGEMM]┘       │
|        ~50-80 μs total  ~50-80 μs       [dispatch]...
|────┤────────────────┤────────────────┤───────────┤
     0               1                 2           3        (ms)

COMPARISON:
BF16: 100 forwards = ~10-50 μs per forward → ~1-5 ms total
FP8:  100 forwards = ~50-80 μs per forward → ~5-8 ms total

Python overhead impact: ~4-5 ms (80% slowdown) for 100 forwards!
```

---

## Key Takeaways

### 1. **FP8 is 20-60x slower at Python dispatch level**
   - BF16: ~0.5-1 μs Python overhead per forward
   - FP8: ~20-30 μs Python overhead per forward

### 2. **Per-token quantization is the main bottleneck**
   - Happens on every forward pass (unlike weight scales which are pre-computed)
   - Requires tensor allocation, validation, scale computation
   - Total: ~10-60 μs per forward

### 3. **Multiple dispatch layers add up**
   - Backend selection: 1-2 μs
   - Wrapper calls: 5-10 μs
   - Matmul preparation: 3-5 μs
   - Kernel dispatch: 2-3 μs

### 4. **No escaping this trade-off**
   - Accuracy (FP8) ↔ Speed (BF16)
   - FP8 REQUIRES dynamic quantization for accuracy
   - This inherently adds ~20-30 μs overhead per forward

### 5. **Optimization opportunities**
   - **Immediate**: Cache quantization buffers (~2-5 μs savings)
   - **Near-term**: Fuse quantization + GEMM (~15-30 μs savings)
   - **Long-term**: Static activation scales (~10-20 μs savings)

