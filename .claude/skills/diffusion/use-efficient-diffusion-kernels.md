---
name: use-efficient-diffusion-kernels
description: Guidance for using SGLang Diffusion fused kernels and fast CUDA paths. Use when mapping fusion patterns in diffusion inference, choosing fused ops or attention backends, handling RoPE/QK norm performance pitfalls, or integrating new diffusion models with kernel-aware constraints.
---

# Use Efficient Diffusion Kernels

**Overview**
This skill focuses on SGLang Diffusion (`sglang.multimodal_gen`) kernel fusion patterns and fast CUDA paths. Prefer existing fused ops (Triton, CuTe DSL, sgl-kernel). Make constraints and fallbacks explicit.

**Key Files**
- `python/sglang/multimodal_gen/runtime/layers/layernorm.py`
- `python/sglang/multimodal_gen/runtime/layers/elementwise.py`
- `python/sglang/multimodal_gen/runtime/layers/rotary_embedding/utils.py`
- `python/sglang/jit_kernel/diffusion/triton/scale_shift.py`
- `python/sglang/jit_kernel/diffusion/triton/norm.py`
- `python/sglang/jit_kernel/diffusion/triton/rmsnorm_onepass.py`
- `python/sglang/jit_kernel/diffusion/triton/rotary.py`
- `python/sglang/jit_kernel/diffusion/cutedsl/scale_residual_norm_scale_shift.py`
- `python/sglang/jit_kernel/norm.py`
- `python/sglang/multimodal_gen/runtime/platforms/cuda.py`
- `python/sglang/multimodal_gen/runtime/layers/attention/selector.py`
- `docs/diffusion/performance/attention_backends.md`

**Core Fusion Patterns**

1. Scale/Shift elementwise fusion (AdaLN modulation)
- Kernels: `fuse_scale_shift_kernel`, `fuse_scale_shift_gate_select01_kernel`
- Locations: `elementwise.py`, `layernorm.py`, `qwen_image.py`, `triton/scale_shift.py`
- Use cases: `x * (1 + scale) + shift` and `a * (k + b) + c`
- Constraints: `x` must be CUDA and contiguous. `scale/shift` support 0D/1D/2D/3D/4D broadcast. 4D `[B, F, 1, C]` requires `L % F == 0`.
- NPU fallback: `scale_shift.py` swaps to `npu_fallback` native path.

2. Norm + Scale/Shift fusion (CuTe DSL)
- Kernels: `fused_norm_scale_shift`, `fused_scale_residual_norm_scale_shift`
- Locations: `layernorm.py`, `cutedsl/scale_residual_norm_scale_shift.py`
- Use cases:
  - `y = norm(x) * (1 + scale) + shift`
  - `y = norm(residual + gate * x) * (1 + scale) + shift`
- Constraints: `D % 256 == 0` and `D <= 8192`. `x/residual/gate/scale/shift` must pass shape and stride validation. Dtypes limited to fp16/bf16/fp32.
- Behavior: CuTe DSL compilation cached by `(dtype, ndim, D, norm_type)`. `None` tensors replaced by scalar placeholders. If constraints fail, `layernorm.py` warns and falls back to native PyTorch.

3. Triton LayerNorm/RMSNorm fusion
- Kernels: `rms_norm_fn`, `layer_norm_fn`, `norm_infer`
- Locations: `triton/norm.py`, `layernorm.py`
- Use cases: fp32 RMSNorm with residual/dropout/rowscale/x1 branches, and inference-friendly `norm_infer`.
- Constraints: last dim must be contiguous, and `N * element_size < 64KB`.

4. Triton one-pass RMSNorm (small hidden size fast path)
- Kernel: `triton_one_pass_rms_norm`
- Locations: `triton/rmsnorm_onepass.py`, `layernorm.py`
- Use case: `hidden_size <= 128` in `RMSNorm.forward_cuda`.

5. Triton RoPE fusion
- Kernel: `apply_rotary_embedding`
- Locations: `triton/rotary.py`, `rotary_embedding/utils.py`
- Use case: GPT-J style RoPE when not Neox.
- Constraints: `head_size` must be even.
- NPU fallback: `npu_fallback.apply_rotary_embedding_native`.

**Faster CUDA Kernel Usage Points**

1. sgl-kernel RMSNorm and fused add RMSNorm
- Location: `layernorm.py`
- Behavior: CUDA uses `sgl_kernel.fused_add_rmsnorm` and `sgl_kernel.rmsnorm`. `hidden_size <= 128` uses Triton one-pass. ROCm falls back to native.

2. Attention backend selection (FlashAttention, Sage, SDPA)
- Locations: `platforms/cuda.py`, `attention/selector.py`, `docs/diffusion/performance/attention_backends.md`
- Behavior: CUDA prefers FlashAttention (FA3/FA4) when supported, otherwise Torch SDPA. Force via `--attention-backend` or `global_force_attn_backend`.

3. FlashInfer RoPE (Q/K inplace)
- Location: `rotary_embedding/utils.py`
- Behavior: `flashinfer.rope.apply_rope_with_cos_sin_cache_inplace` when available, otherwise Triton RoPE fallback.

**QK Norm Optimization**

- Entry point: `apply_qk_norm` in `layernorm.py`.
- Fast path: JIT fused inplace QK norm from `python/sglang/jit_kernel/norm.py` via `fused_inplace_qknorm`.
- Preconditions for fused path:
  - CUDA only.
  - `allow_inplace=True` and `q_eps == k_eps`.
  - `can_use_fused_inplace_qknorm(head_dim, dtype)` returns true.
  - Supported head dims: `64, 128, 256, 512, 1024`.
- Behavior: Fused path operates on `q` and `k` in place after reshaping to `[B, -1, head_dim]`. If preconditions fail, fall back to per-tensor RMSNorm.

**Common Entry Points in Diffusion Models**
- AdaLN modulation: `LayerNormScaleShift`, `RMSNormScaleShift`, `ScaleResidual*` in `layernorm.py`.
- Qwen-Image gating: `fuse_scale_shift_gate_select01_kernel` in `qwen_image.py`.
- QK norm: `apply_qk_norm` used in `flux.py`, `flux_2.py`, `qwen_image.py`, `zimage.py`, `wanvideo.py`, `ltx_2.py`, `hunyuanvideo.py`.
- RoPE: `_apply_rotary_emb` prefers Triton; Q/K RoPE prefers FlashInfer when present.

**Constraints and Fallbacks**
- `scale_shift` Triton requires CUDA + contiguous `x`. NPU swaps to native.
- CuTe DSL fused norms require `D % 256 == 0` and `D <= 8192`.
- Triton norm kernels error on feature size >= 64KB.
- FlashAttention requires fp16/bf16 and SM80+; otherwise SDPA.

**Integration Checklist for New Models**

1. Reuse `LayerNormScaleShift` or `ScaleResidual*` modules instead of re-implementing fusion logic.
2. Keep tensors contiguous and satisfy D alignment (`% 256`) and size (`<= 8192`) for CuTe fused paths.
3. Use `fuse_scale_shift_kernel` for AdaLN modulation and keep a PyTorch fallback.
4. Use `apply_qk_norm` and ensure head_dim is in the supported list for fused QK norm.
5. If using FlashInfer RoPE, avoid `pack qkv` and ensure Q/K are contiguous.
6. For attention, follow `selector.py` priority; override with CLI only if needed.

**When Extending or Modifying Kernels**
- Add `torch.library.custom_op` and `register_fake` for compile and meta support.
- Keep CuTe compile cache keys aligned to `(dtype, ndim, D)`.
- Avoid implicit broadcasts that force hidden `contiguous()` copies.
- Preserve NPU and ROCm fallback paths.
