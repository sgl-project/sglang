# SGLang Diffusion Fast Paths

Use this guide when mapping a diffusion bottleneck to an existing fused path or
distributed overlap pattern in `sglang.multimodal_gen`. Prefer reuse and
configuration first before handing the problem to a specialized kernel-optimization skill.

**Key Files**
- `python/sglang/multimodal_gen/runtime/layers/layernorm.py`
- `python/sglang/multimodal_gen/runtime/layers/elementwise.py`
- `python/sglang/multimodal_gen/runtime/layers/rotary_embedding/utils.py`
- `python/sglang/jit_kernel/diffusion/triton/scale_shift.py`
- `python/sglang/jit_kernel/diffusion/group_norm_silu.py`
- `python/sglang/jit_kernel/diffusion/triton/group_norm_silu.py`
- `python/sglang/jit_kernel/diffusion/triton/norm.py`
- `python/sglang/jit_kernel/diffusion/triton/rmsnorm_onepass.py`
- `python/sglang/jit_kernel/diffusion/triton/rotary.py`
- `python/sglang/jit_kernel/diffusion/cutedsl/scale_residual_norm_scale_shift.py`
- `python/sglang/jit_kernel/tests/diffusion/test_group_norm_silu.py`
- `python/sglang/jit_kernel/benchmark/diffusion/bench_group_norm_silu.py`
- `python/sglang/jit_kernel/norm.py`
- `python/sglang/multimodal_gen/runtime/platforms/cuda.py`
- `python/sglang/multimodal_gen/runtime/layers/attention/selector.py`
- `docs/diffusion/performance/attention_backends.md` (repo root)

**Core Fusion Patterns**

1. Scale/Shift elementwise fusion (AdaLN modulation)
- Kernels: `fuse_scale_shift_kernel`, `fuse_scale_shift_gate_select01_kernel`
- Locations: `elementwise.py`, `layernorm.py`, `qwen_image.py`, `triton/scale_shift.py`
- Use cases: `x * (1 + scale) + shift` and `a * (k + b) + c`
- Constraints: `x` must be CUDA and contiguous. `scale/shift` support 0D/1D/2D/3D/4D broadcast. 4D `[B, F, 1, C]` requires `L % F == 0`.
- NPU fallback: `scale_shift.py` swaps to `npu_fallback` native path.
- Validation: `python/sglang/jit_kernel/tests/diffusion/test_qwen_image_modulation.py`.

2. Norm + Scale/Shift fusion (CuTe DSL)
- Kernels: `fused_norm_scale_shift`, `fused_scale_residual_norm_scale_shift`
- Locations: `layernorm.py`, `cutedsl/scale_residual_norm_scale_shift.py`
- Use cases:
  - `y = norm(x) * (1 + scale) + shift`
  - `y = norm(residual + gate * x) * (1 + scale) + shift`
- Constraints: `D % 256 == 0` and `D <= 8192`. `x/residual/gate/scale/shift` must pass shape and stride validation. Dtypes limited to fp16/bf16/fp32.
- Behavior: CuTe DSL compilation cached by `(dtype, ndim, D, norm_type)`. `None` tensors replaced by scalar placeholders. If constraints fail, `layernorm.py` warns and falls back to native PyTorch.

3. Z-Image fused tanh/gate modulation
- Kernels: `fused_norm_tanh_mul_add`, `fused_norm_tanh_mul_add_norm_scale`
- Locations: `layernorm.py`, `cutedsl/norm_tanh_mul_add_norm_scale.py`, `zimage.py`
- Use cases:
  - `y = tanh(gate) * norm(x) + shift`
  - `y, y2 = tanh(gate) * norm(x) + shift`, then `y2 = norm(y) * (1 + scale)`
- Constraints: same CuTe DSL envelope as the norm+scale/shift family in practice: contiguous last dim, fp16/bf16/fp32, and `D % 256 == 0`, `D <= 8192`.
- Validation: `python/sglang/jit_kernel/tests/diffusion/test_norm_tanh_mul_add_norm_scale.py`
- Behavior: this is already a mainline fast path, so if Z-Image traces show the unfused chain, treat it as a missing or regressed existing optimization before proposing a new kernel.

4. Triton LayerNorm/RMSNorm fusion
- Kernels: `rms_norm_fn`, `layer_norm_fn`, `norm_infer`
- Locations: `triton/norm.py`, `layernorm.py`
- Use cases: fp32 RMSNorm with residual/dropout/rowscale/x1 branches, and inference-friendly `norm_infer`.
- Constraints: last dim must be contiguous, and `N * element_size < 64KB`.
- Validation: `python/sglang/jit_kernel/tests/test_rmsnorm.py`.

5. Triton one-pass RMSNorm (small hidden size fast path)
- Kernel: `triton_one_pass_rms_norm`
- Locations: `triton/rmsnorm_onepass.py`, `layernorm.py`
- Use case: `hidden_size <= 128` in `RMSNorm.forward_cuda`.
- `torch.compile` note: keep this path behind the custom-op wrapper in `rmsnorm_onepass.py`; direct `wrap_triton` can recompile on dynamic row counts.

6. Triton RoPE fusion
- Kernel: `apply_rotary_embedding`
- Locations: `triton/rotary.py`, `rotary_embedding/utils.py`
- Use case: GPT-J style RoPE when not Neox.
- Constraints: `head_size` must be even.
- NPU fallback: `npu_fallback.apply_rotary_embedding_native`.
- Validation: `python/sglang/jit_kernel/tests/test_rope.py`.

7. HunyuanVideo VAE GroupNorm + SiLU fusion
- Kernel: `triton_group_norm_silu`
- Locations: `diffusion/group_norm_silu.py`, `triton/group_norm_silu.py`, `runtime/models/vaes/hunyuanvae.py`
- Use case: `activation(group_norm(x))` when the activation is non-inplace `nn.SiLU` and the GroupNorm is affine.
- Enablement: mainline uses `apply_group_norm_silu(...)` in HunyuanVideo VAE paths by default; there is no env toggle. The wrapper dispatches to Triton only when guards pass.
- Constraints: CUDA inference path only; no grad, `x.requires_grad == False`, `nn.GroupNorm`, `nn.SiLU(inplace=False)`, affine norm with weight and bias. Unsupported cases fall back to native `activation(norm(x))`.
- Validation: `python/sglang/jit_kernel/tests/diffusion/test_group_norm_silu.py`.
- Microbench: `python/sglang/jit_kernel/benchmark/diffusion/bench_group_norm_silu.py`.

**Faster CUDA Kernel Usage Points**

1. sgl-kernel RMSNorm and fused add RMSNorm
- Location: `layernorm.py`
- Behavior:
- Standard `bf16`/`fp16` CUDA paths use `sgl_kernel.fused_add_rmsnorm` and `sgl_kernel.rmsnorm`.
- The Z-Image `fp32` `32x2560` path under `torch.compile` avoids `wrap_triton` and uses the native fp32 path.
- `hidden_size <= 128` uses Triton one-pass.
- ROCm falls back to native.

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
- Validation: `python/sglang/jit_kernel/tests/test_qknorm.py` and `python/sglang/jit_kernel/tests/test_qknorm_across_heads.py`.

**QK Norm + RoPE Optimization**

- Entry point: `apply_qk_norm_rope` in `layernorm.py`.
- Fast path: JIT fused inplace QK norm + RoPE from `python/sglang/jit_kernel/diffusion/qknorm_rope.py` via `fused_inplace_qknorm_rope`.
- Toggle: `SGLANG_ENABLE_FUSED_QKNORM_ROPE=1` keeps the fused path enabled by default.
- Preconditions for fused path:
  - CUDA only.
  - `allow_inplace=True` and `q_eps == k_eps`.
  - `q` / `k` are contiguous 4D tensors with the same shape.
  - `q.dtype` is `fp16` or `bf16`, and norm weights match tensor dtype.
  - `can_use_fused_inplace_qknorm_rope(head_dim, rope_dim, is_neox, dtype)` returns true.
  - Supported head dims: `64, 128, 256`.
- Behavior: `apply_qk_norm_rope` prefers the fused JIT kernel when all guards pass; otherwise it falls back to `apply_qk_norm(...)` plus `apply_flashinfer_rope_qk_inplace(...)`.
- Validation: `python/sglang/jit_kernel/tests/diffusion/test_qknorm_rope.py`.
- Watchlist: PR #24025 adds LTX2-specific QK norm fusion work. Until merged, treat LTX2 traces that miss the generic fused path as an enablement/shape-guard issue first, not proof that the PR path exists locally.

**Nunchaku Fused GELU MLP**

- Entry point: `_fused_gelu_mlp` in `runtime/models/dits/flux.py`.
- Fast path: Nunchaku checkpoints can fuse `fc1 GEMM + GELU + shift + re-quant + fc2.lora_down` before the second GEMM instead of materializing a standalone GELU activation.
- Scope: this is a model-specific fast path for Nunchaku-quantized FLUX-family checkpoints.
- Workflow rule: if a Nunchaku trace shows split `fc1 -> gelu -> quant -> fc2.lora_down`, treat it as a missing existing fast path before proposing a new fusion.

**NVFP4 / Nunchaku Packed QKV**

- Entry points: `runtime/models/dits/flux.py`, `runtime/models/dits/flux_2.py`, and the FLUX config remapping in `configs/models/dits/flux.py`.
- Fast path: quantized FLUX-family checkpoints can store attention projections in packed QKV form, and SGLang intentionally switches to `MergedColumnParallelLinear` paths such as `to_qkv`, `to_added_qkv`, and `to_qkv_mlp_proj` instead of separate `to_q`, `to_k`, `to_v`.
- FLUX.2 NVFP4 note: `flux_2.py` explicitly enables fused packed QKV when `quant_config` is `ModelOptFp4Config`, because the NVFP4 checkpoint stores image-attention QKV packed on disk.
- Nunchaku note: raw and converted Nunchaku checkpoint names are remapped onto fused `to_qkv` / `to_added_qkv` names in `configs/models/dits/flux.py`; correctness on NVFP4-style checkpoints also depends on quant metadata such as `wtscale` and attention `wcscales`.
- Workflow rule: if an NVFP4 or Nunchaku trace shows split `to_q -> to_k -> to_v` where packed QKV is expected, treat it as a missing quantized fast path or checkpoint-format mismatch before proposing a new attention fusion.

**Common Entry Points in Diffusion Models**
- AdaLN modulation: `LayerNormScaleShift`, `RMSNormScaleShift`, `ScaleResidual*` in `layernorm.py`.
- Qwen-Image gating: `fuse_scale_shift_gate_select01_kernel` in `qwen_image.py`.
- Z-Image residual-form modulation: `fused_norm_tanh_mul_add` and `fused_norm_tanh_mul_add_norm_scale` in `zimage.py`.
- HunyuanVideo VAE GroupNorm+SiLU: `apply_group_norm_silu` in `hunyuanvae.py`; default-eligible when wrapper guards pass.
- QK norm: `apply_qk_norm` used in `flux.py`, `flux_2.py`, `qwen_image.py`, `zimage.py`, `wanvideo.py`, `ltx_2.py`, `hunyuanvideo.py`.
- QK norm + RoPE: `apply_qk_norm_rope` in `layernorm.py`; use this path when the model wants fused attention prep instead of separate QK norm and RoPE calls.
- Nunchaku fused GELU MLP: `_fused_gelu_mlp` in `flux.py` for quantized FLUX-family checkpoints.
- NVFP4 / packed QKV attention: `to_qkv`, `to_added_qkv`, and `to_qkv_mlp_proj` in FLUX-family quantized paths.
- RoPE: `_apply_rotary_emb` prefers Triton; Q/K RoPE prefers FlashInfer when present.

**Existing Overlap / Communication Families**

- Ulysses / USP attention: treat `all_to_all`, `ring_attn`, and head / sequence reshards as an existing distributed attention family, not a new overlap idea.
- Turbo-layer async all-to-all: `all_to_all_single(..., async_op=True)` plus staged waits already form an existing overlap family in `turbo_layer.py`.
- TorchInductor compute / communication reorder: `torch._inductor.config.reorder_for_compute_comm_overlap = True` can already partially overlap compiled denoise traces.
- Dual-stream diffusion models: `use_dual_stream = True` in models such as `hunyuan3d.py` is an existing overlap family.
- Workflow rule: if a hotspot is communication-heavy, rule out these in-repo overlap families before proposing a brand new overlap design.

**Open PR Watchlist**

As of 2026-05-02, these SGLang PRs were still open. Use them as upstream
direction and prior art, not as current-main behavior. Re-check the PR state
before relying on any file path or flag.

- Norm, modulation, and packed projection fusions:
  - #24025 LTX2 QK norm fusion.
  - #24059 Helios fused norm modulation.
  - #24117 Z-Image packed QKV.
  - #19488 Wan cross-block elementwise fusion.
  - #19249 Z-Image `scale residual norm scale shift` plus `add gate norm` fusion.
  - #18897 dual norm fusion for FLUX-family paths (draft).
  - #20429 Qwen-Image layernorm and `fuse_scale_shift_gate_select01` work.
  - #20530 MOVA fused RMSNorm + interleaved RoPE.
- VAE and decode-side acceleration:
  - #22531 LTX2 parallel VAE support and #20927 batched tiled VAE decode (draft).
- Attention, communication, and runtime scheduling:
  - #22805 FLUX.2 packed QKV for all-to-all.
  - #21742 hybrid attention schedule.
  - #24053 USP attention with replicated prefixes.
  - #18764 dynamic batching v0.
  - #24200 disaggregated diffusion v2.
- Cache and CUDA graph:
  - #21613 TeaCache refactor.
  - #24227 WanVideo TeaCache skipping fix.
  - #20447 TeaCache support for GLM-Image, Qwen-Image, and related models.
  - #19516 Qwen-Image CUDA Graph.
  - #21912 Z-Image Turbo FP8 full quantization and CUDA Graph.

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
- If none of the families above match, package the evidence from the benchmark/profile skill and hand the kernel work to a specialized optimization skill such as `sglang-diffusion-ako4all-kernel`.
