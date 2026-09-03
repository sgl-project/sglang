# SGLang Diffusion Fast Paths

Use this guide when mapping a diffusion bottleneck to an existing fused path or
distributed overlap pattern in `sglang.multimodal_gen`. Prefer reuse and
configuration first before handing the problem to a kernel, Nsight, or
framework-specific optimization workflow.

**Key Files**
- `python/sglang/multimodal_gen/runtime/layers/layernorm.py`
- `python/sglang/multimodal_gen/runtime/layers/elementwise.py`
- `python/sglang/multimodal_gen/runtime/layers/fused_scale_shift_gate.py`
- `python/sglang/multimodal_gen/runtime/layers/rotary_embedding/utils.py`
- `python/sglang/kernels/ops/diffusion/modulate/scale_shift_triton.py`
- `python/sglang/kernels/ops/diffusion/modulate/modulate_scale_shift_jit.py`
- `python/sglang/kernels/ops/diffusion/sites/fused_ln_modulate_site.py`
- `python/sglang/kernels/ops/diffusion/sites/quality_gate.py`
- `python/sglang/kernels/ops/diffusion/sites/bitexact_gate.py`
- `python/sglang/kernels/ops/diffusion/norm/group_norm_silu.py`
- `python/sglang/kernels/ops/diffusion/norm/group_norm_silu_triton.py`
- `python/sglang/kernels/ops/diffusion/norm/group_norm_silu_twopass_triton.py`
- `python/sglang/kernels/ops/diffusion/norm/norm_triton.py`
- `python/sglang/kernels/ops/diffusion/norm/rmsnorm_onepass_triton.py`
- `python/sglang/kernels/ops/diffusion/norm/layernorm_modulate_triton.py`
- `python/sglang/kernels/ops/diffusion/norm/native_bf16_rmsnorm_triton.py`
- `python/sglang/kernels/ops/diffusion/norm/zimage_qk_rmsnorm_triton.py`
- `python/sglang/kernels/ops/diffusion/rope/rotary_triton.py`
- `python/sglang/kernels/ops/diffusion/rope/helios_qk_rope_jit.py`
- `python/sglang/kernels/ops/diffusion/rope/ltx2_rotary_triton.py`
- `python/sglang/kernels/ops/diffusion/rope/ltx2_qknorm_split_rope_jit.py`
- `python/sglang/kernels/ops/diffusion/sites/ltx2_rmsnorm_modulate_site.py`
- `python/sglang/kernels/ops/diffusion/modulate/indexed_modulation_triton.py`
- `python/sglang/kernels/ops/diffusion/layout/ulysses_qkv_triton.py`
- `python/sglang/kernels/ops/diffusion/layout/usp_relayout_jit.py`
- `python/sglang/multimodal_gen/runtime/layers/usp.py`
- `python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py`
- `python/sglang/multimodal_gen/runtime/models/dits/longcat_image.py`
- `python/sglang/multimodal_gen/runtime/models/dits/sana_video.py`
- `python/sglang/multimodal_gen/runtime/models/dits/lingbot_video_moe.py`
- `python/sglang/multimodal_gen/runtime/models/decoders/ltx_2_5_diffusion_decoder.py`
- `python/sglang/multimodal_gen/runtime/layers/moe.py`
- `python/sglang/srt/layers/moe/topk.py`
- `python/sglang/kernels/ops/diffusion/modulate/residual_gate_add_jit.py`
- `python/sglang/kernels/jit/csrc/diffusion/residual_gate_add.cuh`
- `python/sglang/kernels/ops/diffusion/layout/varlen_pack_pad_triton.py`
- `python/sglang/kernels/ops/diffusion/layout/wan_causal_cache_triton.py`
- `python/sglang/kernels/ops/diffusion/norm/scale_residual_norm_cutedsl.py`
- `python/sglang/multimodal_gen/runtime/models/vaes/fast_path_gate.py`
- `python/sglang/multimodal_gen/runtime/models/vaes/flux2_vae_cuda_opt.py`
- `python/sglang/multimodal_gen/runtime/models/vaes/wan_vae_cuda_opt.py`
- `python/sglang/multimodal_gen/runtime/breakable_cuda_graph/runner.py`
- `test/registered/kernels/ops/diffusion/test_qwen_image_modulation.py`
- `test/registered/kernels/ops/diffusion/test_group_norm_silu.py`
- `test/registered/kernels/ops/diffusion/test_residual_gate_add.py`
- `test/registered/kernels/ops/diffusion/test_varlen_pack_pad.py`
- `test/registered/kernels/ops/diffusion/test_varlen_uspattn_equivalence.py`
- `test/registered/kernels/ops/diffusion/test_native_bf16_rmsnorm.py`
- `test/registered/kernels/ops/diffusion/test_flux_ln_modulate.py`
- `test/registered/kernels/ops/diffusion/test_glm_image_ln_modulate.py`
- `test/registered/kernels/ops/diffusion/test_sana_ln_modulate.py`
- `test/registered/kernels/ops/diffusion/test_quality_gate.py`
- `test/registered/kernels/ops/diffusion/test_ltx2_rms_norm_modulate.py`
- `test/registered/kernels/ops/diffusion/test_bitexact_gate.py`
- `test/registered/kernels/ops/diffusion/test_wan_causal_cache.py`
- `test/registered/kernels/ops/diffusion/test_stage_profiler_sync.py`
- `test/registered/kernels/benchmark/diffusion/bench_qwen_image_modulation.py`
- `test/registered/kernels/benchmark/diffusion/bench_group_norm_silu.py`
- `test/registered/kernels/benchmark/diffusion/bench_residual_gate_add.py`
- `python/sglang/kernels/ops/layernorm/norm.py`
- `python/sglang/multimodal_gen/runtime/platforms/cuda.py`
- `python/sglang/multimodal_gen/runtime/layers/attention/selector.py`
- `docs/docs/sglang-diffusion/attention_backends.mdx` (repo root)

**Core Fusion Patterns**

1. Scale/Shift elementwise and gate fusion (AdaLN modulation)
- Kernels: `fuse_scale_shift_kernel`, `fuse_layernorm_scale_shift_gate_select01_kernel`, `fuse_residual_layernorm_scale_shift_gate_select01_kernel`
- Locations: `elementwise.py`, `layernorm.py`, `fused_scale_shift_gate.py`, `qwen_image.py`, `triton/scale_shift.py`
- Use cases: `x * (1 + scale) + shift`, `a * (k + b) + c`, and Qwen-style `(layernorm/residual layernorm) + scale/shift + gate select`.
- Constraints: `x` must be CUDA and contiguous. `scale/shift` support 0D/1D/2D/3D/4D broadcast. 4D `[B, F, 1, C]` requires `L % F == 0`.
- Causal-video cold start: the 4D path uses a static capped power-of-two
  column tile rather than Triton autotuning. Do not reintroduce request-time
  autotuning here: LingBot-World calls this path once per transformer block,
  and tuning overhead can dominate its first denoise step.
- NPU fallback: `scale_shift.py` swaps to `npu_fallback` native path.
- Validation: `test/registered/kernels/ops/diffusion/test_qwen_image_modulation.py`.

2. Norm + Scale/Shift fusion (CuTe DSL)
- Kernels: `fused_norm_scale_shift`, `fused_scale_residual_norm_scale_shift`
- Locations: `layernorm.py`, `cutedsl/scale_residual_norm_scale_shift.py`
- Use cases:
  - `y = norm(x) * (1 + scale) + shift`
  - `y = norm(residual + gate * x) * (1 + scale) + shift`
- Constraints: `D % 256 == 0` and `D <= 8192`. `x/residual/gate/scale/shift` must pass shape and stride validation. Dtypes limited to fp16/bf16/fp32.
- Behavior: CuTe DSL compilation cached by `(dtype, ndim, D, norm_type)`. `None` tensors replaced by scalar placeholders. If constraints fail, `layernorm.py` warns and falls back to native PyTorch.

3. Bit-exact adaLN modulation and LayerNorm + modulation
- Kernels: `modulate_scale_shift`, `fused_layernorm_modulate`, and
  `fused_qk_head_layernorm`.
- Locations: `modulate_scale_shift.py`, `triton/layernorm_modulate.py`,
  `runtime/models/dits/flux.py`, `glm_image.py`, and `sana.py`.
- Use cases:
  - `x * (1 + scale[:, None]) + shift[:, None]` as one JIT CUDA launch.
  - BF16 `LayerNorm(x) * (1 + scale) + shift` as one Triton launch that
    reproduces the active aten BF16 reduction and rounding order.
  - GLM-Image per-head Q/K LayerNorm with the same aten-compatible reduction.
- Constraints: the JIT modulation path requires aligned contiguous CUDA
  fp16/bf16 BLC inputs with `[B, D]` scale/shift. The Triton LayerNorm path is
  BF16-specific and only claims bit-exactness for its guarded aten dispatch;
  FLUX/GLM/SANA run a live eager equality check and fail closed on mismatch.
- Validation: `test_flux_ln_modulate.py`, `test_glm_image_ln_modulate.py`,
  `test_sana_ln_modulate.py`, `test_modulate_scale_shift.py`, and
  `test_fused_ln_modulate.py`.
- Workflow rule: if these models show separate norm and modulation kernels,
  check dtype, alignment, shape, BCG/compile context, and the one-time equality
  self-test before proposing another fusion.

4. Request-scoped fusion gates at `quality=extra-high` or `quality=high`
- Locations: `quality_gate.py`, `fused_ln_modulate.py`, `denoising.py`,
  `decoding.py`, `fast_path_gate.py`, `flux2_vae_cuda_opt.py`, and
  `wan_vae_cuda_opt.py`.
- Behavior: `quality="lossless"` is the default exact reference path.
  `quality="extra-high"` and `quality="high"` mount the same validated but
  non-bit-exact DiT fusions and decode-scoped VAE rewrites. `high` is
  cumulative and may additionally enable model-owned approximate paths.
  Mounting is all-or-nothing per
  transformer/fusion family; VAE gates reset after every decode.
- Current families include FLUX affine-folded LN+modulate / fused GELU sites,
  Wan cublasLt/NVFP4 GELU, Qwen added-QKV, GLM/Qwen/Hunyuan/LTX fused GELU,
  LTX RMSNorm+modulate, Hunyuan QK RMSNorm, Ideogram gated RMSNorm,
  LingBot RMSNorm, SANA-Video linear attention, generic KL VAE
  decoder rewrites used by FLUX.1/FLUX.2/Z-Image/SD3, and Wan VAE
  RMSNorm+SiLU.
- Do not confuse request `--quality` with `--output-quality`, which controls
  output-file compression rather than model math.
- Validation: `test_quality_gate.py`, `test_fused_ln_modulate.py`,
  `test_flux2_vae_fastpath.py`, `test_wan_vae_fastpath.py`, and
  `test_vae_fast_path_gate.py`.

5. Z-Image bf16-native RMSNorm modulation (Triton)
- Kernels: `rmsnorm_scale`, `rmsnorm_tanh_residual`
- Locations: `triton/native_bf16_rmsnorm.py`, with wrappers in `zimage.py` and
  `fused_gate_rmsnorm.py`. Note: `triton/zimage_native_norm.py` is QK-only.
- Use cases:
  - `y = rmsnorm(x) * scale`
  - `y = residual + tanh(gate) * rmsnorm(x)`
- Constraints: CUDA bf16 tensors, contiguous weights, flattenable row strides,
  compatible modulation row counts, and `D <= 8192`.
- Validation: `test/registered/kernels/ops/diffusion/test_native_bf16_rmsnorm.py`
- Behavior: the kernels preserve Z-Image's native bf16 arithmetic. They return
  `None` when an eligibility guard fails, and the runtime wrapper executes the
  native PyTorch formula.

6. Triton LayerNorm/RMSNorm fusion
- Kernels: `rms_norm_fn`, `layer_norm_fn`, `norm_infer`
- Locations: `triton/norm.py`, `layernorm.py`
- Use cases: fp32 RMSNorm with residual/dropout/rowscale/x1 branches, and inference-friendly `norm_infer`.
- Constraints: last dim must be contiguous, and `N * element_size < 64KB`.
- Validation: `test/registered/kernels/ops/layernorm/test_rmsnorm.py`.

7. Triton one-pass RMSNorm (small hidden size fast path)
- Kernel: `triton_one_pass_rms_norm`
- Locations: `triton/rmsnorm_onepass.py`, `layernorm.py`
- Use case: `hidden_size <= 128` in `RMSNorm.forward_cuda`.
- `torch.compile` note: keep this path behind the custom-op wrapper in `rmsnorm_onepass.py`; direct `wrap_triton` can recompile on dynamic row counts.

8. Triton RoPE fusion
- Kernel: `apply_rotary_embedding`
- Locations: `triton/rotary.py`, `rotary_embedding/utils.py`
- Use case: GPT-J style RoPE when not Neox.
- Constraints: `head_size` must be even.
- NPU fallback: `npu_fallback.apply_rotary_embedding_native`.
- Validation: `test/registered/kernels/ops/attention/test_rope.py`.

9. LTX2 split RoPE fusion
- Kernel: `apply_ltx2_split_rotary_emb`
- Locations: `triton/ltx2_rotary.py`, `runtime/models/dits/ltx_2.py`
- Use case: LTX-2 split rotary embedding over `[B, S, num_heads * head_dim]` with separate `cos` and `sin` tensors.
- Constraints: `cos` and `sin` shapes must match `[B, H, S, head_dim / 2]`, and `inner_dim == H * head_dim`.
- Workflow rule: if LTX-2 traces show a large split-RoPE PyTorch chain, check whether the LTX2-specific Triton path was disabled by shape or dtype before proposing a new RoPE kernel.

10. Shared residual-gate add fusion (LTX2, LongCat-Image, SANA, and SANA-Video)
- Kernel: `diffusion_residual_gate_add`
- Locations: `kernels/ops/diffusion/modulate/residual_gate_add_jit.py`, `kernels/jit/csrc/diffusion/residual_gate_add.cuh`, `runtime/models/dits/ltx_2.py`, `runtime/models/dits/longcat_image.py`, `runtime/models/dits/sana.py`, and `runtime/models/dits/sana_video.py`.
- Use case: `residual + update * gate` in LTX2 attention/MLP residuals, LongCat-Image joint- and single-stream transformer residuals, and SANA/SANA-Video transformer blocks.
- Constraints: inputs must be same-device CUDA tensors with one dtype (`fp16`, `bf16`, or `fp32`) and `update.shape == residual.shape`. The ordinary path accepts contiguous inputs and a full or row-broadcast gate. The SANA-Video path also accepts a transposed-dense 3D residual (`stride == (tokens * hidden, 1, tokens)`), a contiguous update, and a contiguous `[1, 1, hidden]` gate; it preserves the residual stride in its output.
- Behavior: model code calls `residual_gate_add(...)` directly. The CUDA custom op is used while guards pass. On a runtime exception outside `torch.compile`, it logs once, disables the fast path for that device/dtype, and falls back to `residual + update * gate`.
- Validation: `test/registered/kernels/ops/diffusion/test_modulate.py`, `python/sglang/multimodal_gen/test/unit/test_longcat_image_residual_gate.py`.
- Microbench: `test/registered/kernels/benchmark/diffusion/bench_residual_gate_add.py`.
- Workflow rule: if LTX2, LongCat-Image, or SANA traces show repeated elementwise `mul` + `add` ladders around attention or MLP residuals, inspect input strides and check whether this existing CUDA path was disabled by shape, dtype, layout, or a prior runtime failure before proposing another elementwise fusion. For a transposed residual plus contiguous update, do not force `.contiguous()`; the tiled path is designed to fuse the mixed-layout access.

11. MiniMax-H3 indexed AdaLN modulation and gated residual fusion
- Kernels: `indexed_scale_shift_bf16_`, `indexed_gate_bf16_`
- Locations: `triton/indexed_modulation.py`, `runtime/models/dits/minimax_h3.py`
- Use cases: H3's packed video/audio/text rows select per-token modulation with `combined_indices`; the Triton paths replace `index_select` plus scale/shift or gated residual chains in place.
- Constraints: CUDA BF16 H3 tensors, BF16 modulation tensors, contiguous disposable inputs; the gated path also requires contiguous `other`. Unsupported shapes/dtypes retain the eager formula.
- Numerical contract: the kernels explicitly reproduce H3's eager BF16 rounding boundaries. Do not replace them with a mathematically equivalent contraction without the H3 consistency check.
- Workflow rule: if H3 traces show `index_select` plus elementwise ladders around every block, check dtype, contiguity, and input-reuse eligibility before designing another modulation kernel.

12. MiniMax-H3 packed Ulysses QKV and output relayout
- Kernels: `pack_qkv_destination_major`, `usp_merge_heads`
- Locations: `triton/ulysses_qkv.py`, `usp_relayout.py`, `runtime/layers/usp.py`, `runtime/models/dits/minimax_h3.py`
- Use cases: one destination-major QKV pack plus one collective replaces three separately prepared Ulysses input exchanges; the output JIT kernel replaces `permute(...).contiguous()` when merging gathered heads.
- Constraints: packed QKV fast packing requires CUDA fp16/bf16 Q/K/V with matching dtypes, contiguous head dimension, and eager execution. `usp_merge_heads` requires a nonempty contiguous 5D CUDA fp16/bf16/fp32 tensor and is disabled inside `torch.compile`.
- Related transport: 2-rank, peer-accessible CUDA groups can use the existing IPC A2A transport; larger or unsupported groups fall back to the normal collective path.
- Workflow rule: if an H3 Ulysses trace has three Q/K/V preparation ladders or a large output `permute + contiguous`, first prove why these existing guards missed.

13. HunyuanVideo / LTX upsampler GroupNorm + SiLU fusion
- Kernel: `triton_group_norm_silu`
- Locations: `diffusion/group_norm_silu.py`, `triton/group_norm_silu.py`, `runtime/models/vaes/hunyuanvae.py`, `runtime/models/upsampler/latent_upsampler.py`
- Use case: `activation(group_norm(x))` when the activation is non-inplace `nn.SiLU` and the GroupNorm is affine.
- Enablement: mainline uses `apply_group_norm_silu(...)` in HunyuanVideo VAE paths and LTX latent upsampler paths by default; there is no env toggle. The wrapper dispatches to Triton only when guards pass.
- Constraints: CUDA inference path only; no grad, `x.requires_grad == False`, `nn.GroupNorm`, `nn.SiLU(inplace=False)`, affine norm with weight and bias. Unsupported cases fall back to native `activation(norm(x))`.
- Validation: `test/registered/kernels/ops/diffusion/test_group_norm_silu.py`.
- Microbench: `test/registered/kernels/benchmark/diffusion/bench_group_norm_silu.py`.

14. Wan causal-VAE data-movement fusion
- Kernels: `cat_pad_channels_last_3d` and `dup_up3d_add`.
- Locations: `triton/wan_causal_cache.py` and
  `runtime/models/vaes/wanvae.py`.
- Use cases: build causal Conv3d input plus the next compact feature cache in
  one channels-last-3D pass, and fuse `main + DupUp3D(src)` without
  materializing `repeat_interleave + permute().contiguous()` intermediates.
- Numerical contract: these are bit-exact data-movement / same-order-add
  replacements and run independently of the request-gated Wan RMSNorm+SiLU
  path. Unsupported layouts or padding fall back to the aten chain.
- Validation: `test/registered/kernels/ops/diffusion/test_wan_causal_cache.py`.

15. Helios paired transposed RoPE
- Kernel: `fused_inplace_helios_qk_rope`.
- Locations: `rope/helios_qk_rope_jit.py`,
  `csrc/diffusion/helios_qk_rope.cuh`, and
  `runtime/models/dits/helios.py`.
- Use case: apply Helios' transposed fp32 frequency table to already-normalized
  contiguous Q/K together, in place, instead of launching the eager
  unflatten/chunk/multiply/add/stack chain twice per attention block.
- Constraints: CUDA fp16/bf16 Q/K with matching contiguous `[B, S, H, D]`
  layouts, contiguous fp32 frequencies shaped `[B, S, 2 * D]`, even `D`, and
  pair-aligned Q/K pointers. Tensor-parallel RMSNorm keeps the eager path.
  Current real-model validation covers one H100; it is not a multi-GPU scaling
  claim.
- Numerical contract: explicit round-to-nearest fp32 operations reproduce the
  eager elementwise rounding boundaries before the result is cast back to the
  activation dtype. Correctness tests require `torch.equal`, including the
  production `[8640, 40, 128]` shape.
- Validation: `test/registered/kernels/ops/diffusion/test_helios_qk_rope.py`.
- Microbench:
  `test/registered/kernels/benchmark/diffusion/bench_helios_qk_rope.py`.
- Workflow rule: if a Helios trace still shows two transposed-RoPE elementwise
  ladders per block, check TP mode, dtype, shape, contiguity, and pointer
  alignment before proposing another RoPE kernel.

**Faster CUDA Kernel Usage Points**

1. sgl-kernel RMSNorm and fused add RMSNorm
- Location: `layernorm.py`
- Behavior:
- Standard `bf16`/`fp16` CUDA paths use `sgl_kernel.fused_add_rmsnorm` and `sgl_kernel.rmsnorm`.
- Z-Image keeps bf16 arithmetic and uses its dedicated Triton native-norm
  kernels when their guards pass.
- `hidden_size <= 128` uses Triton one-pass.
- ROCm falls back to native.

2. Attention backend selection (FlashAttention, Sage, SDPA)
- Locations: `platforms/cuda.py`, `attention/selector.py`, `docs/docs/sglang-diffusion/attention_backends.mdx`
- Behavior: CUDA prefers FlashAttention (FA3/FA4) when supported, otherwise Torch SDPA. Force via `--attention-backend` or `global_force_attn_backend`.

3. FlashInfer RoPE (Q/K inplace)
- Location: `rotary_embedding/utils.py`
- Behavior: `flashinfer.rope.apply_rope_with_cos_sin_cache_inplace` when available, otherwise Triton RoPE fallback.

4. Varlen USP attention pack/scatter
- Locations: `runtime/layers/attention/layer.py`, `triton/varlen_pack_pad.py`
- Behavior: masked `USPAttention.forward` can gather dense Q/K/V into packed `[total_valid, H, D]` rows with `fused_pack_qkv`, run varlen attention, then scatter back with `fused_scatter_to_padded`.
- Validation: `test/registered/kernels/ops/diffusion/test_varlen_pack_pad.py` and `test/registered/kernels/ops/diffusion/test_varlen_uspattn_equivalence.py`.
- Workflow rule: if a masked attention trace spends time in Python/advanced indexing pack or scatter, first check whether this fused varlen path should have engaged.

**QK Norm Optimization**

- Entry point: `apply_qk_norm` in `layernorm.py`.
- Fast path: JIT fused inplace QK norm from `python/sglang/kernels/ops/layernorm/norm.py` via `fused_inplace_qknorm`.
- Preconditions for fused path:
  - CUDA only.
  - `allow_inplace=True` and `q_eps == k_eps`.
  - `can_use_fused_inplace_qknorm(head_dim, dtype)` returns true.
  - Supported head dims: `64, 128, 256, 512, 1024`.
- Behavior: Fused path operates on `q` and `k` in place after reshaping to `[B, -1, head_dim]`. If preconditions fail, fall back to per-tensor RMSNorm.
- Validation: `test/registered/kernels/ops/layernorm/test_qknorm.py` and `test/registered/kernels/ops/layernorm/test_qknorm_across_heads.py`.

**QK Norm + RoPE Optimization**

- Entry point: `apply_qk_norm_rope` in `layernorm.py`.
- Fast path: JIT fused inplace QK norm + RoPE from `python/sglang/kernels/ops/diffusion/rope/qknorm_rope_jit.py` via `fused_inplace_qknorm_rope`.
- Toggle: `SGLANG_ENABLE_FUSED_QKNORM_ROPE=1` keeps the fused path enabled by default.
- Preconditions for fused path:
  - CUDA only.
  - `allow_inplace=True` and `q_eps == k_eps`.
  - `q` / `k` are contiguous 4D tensors with the same shape.
  - `q.dtype` is `fp16` or `bf16`, and norm weights match tensor dtype.
  - `can_use_fused_inplace_qknorm_rope(head_dim, rope_dim, is_neox, dtype)` returns true.
  - Supported head dims: `64, 128, 256`.
- Behavior: `apply_qk_norm_rope` prefers the fused JIT kernel when all guards pass; otherwise it falls back to `apply_qk_norm(...)` plus `apply_flashinfer_rope_qk_inplace(...)`.
- Validation: `test/registered/kernels/ops/diffusion/test_qknorm_rope.py`.
- MiniMax-H3: the H3 DiT calls `fused_inplace_qknorm_rope` directly for BF16
  head dim 128 with 96 rotary dims, NeoX layout, and
  `round_norm_before_rope=True`. This flag is part of H3's eager numerical
  contract. Compiled execution deliberately falls back to separate eager
  operations.
- Workflow rule: treat LTX2 traces that miss the generic fused path as an enablement/shape-guard issue first, and check the separate LTX2 split-RoPE path before proposing new attention-prep kernels.

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

**SANA Packed Projection GEMMs**

- Entry point: `runtime/models/dits/sana.py`.
- Fast path: SANA self-attention uses one `MergedColumnParallelLinear` `to_qkv` GEMM for Q/K/V, and SANA cross-attention uses one `MergedColumnParallelLinear` `to_kv` GEMM for encoder K/V.
- Scope: this is a mainline SANA model fast path. Query projection in cross-attention remains separate because it uses denoising hidden states, while K/V share step-invariant encoder hidden states.
- Workflow rule: if a SANA trace shows separate self-attention `to_q`, `to_k`, `to_v` GEMMs, or separate cross-attention `to_k` and `to_v` GEMMs, treat that as a regressed existing packed-projection path before proposing a new GEMM fusion.

**Request-Scoped DiT Fusions with Breakable CUDA Graphs**

- DiT sites at `quality=extra-high` or `quality=high` are mounted at a request boundary. BCG warmup uses
  the model's lossless sampling default unless a quality-aware graph variant
  was captured explicitly.
- A graph captured before the request-quality mount retains the lossless module
  branches. Replaying it after the mount silently bypasses the requested fused
  kernels even when the tensor signature matches.
- Workflow rule: an extra-high/high+BCG cell is valid only when the model has no
  request-scoped DiT quality sites, or when logs prove those sites were mounted
  before the matching graph capture. A mount after `[Diffusion BCG] captured`
  invalidates the row; do not use its latency or output as request-quality
  evidence.

**Recent Model Audit Boundaries**

- LongCat-Image supports breakable CUDA graph at fixed, captured resolutions.
  Its DiT always receives a 512-token prompt body, so different raw prompt
  lengths reuse the same graph signature without padding. A model-specific
  pass-through padder prevents the generic buckets from expanding this fixed
  shape into unused graph signatures.
  The model still has split image/text QKV projections and performs
  joint-stream `cat`/split inside each single block. Do not misclassify those
  as a missed existing packed path; they are model-local structural
  opportunities that need their own weight-loader and parity coverage.
- SANA-Video already packs self QKV and cross KV. For fixed 832x480 serving,
  its default 300-token prompt shape can reuse one breakable CUDA graph without
  generic text-bucket padding. An H200 81-frame, 8-step run measured
  920.6--925.3 ms/step eager versus 797.8--798.9 ms/step with BCG, with
  bit-exact final videos; reserved peak memory increased by about 3.4 GB.
  Its conv/modulation formulas mirror SANA, but it does not yet call SANA's
  bit-exact bias-SiLU, bias-GLU, residual-gate, LayerNorm-modulation, or
  one-time contiguous-layout helpers. Reuse or extract those helpers before
  authoring a video-only kernel.
- LingBot Video MoE's router implements sigmoid+bias grouped top-k in
  `multimodal_gen/runtime/layers/moe.py`. Check parameter and output-order
  compatibility with `srt/layers/moe/topk.py::biased_grouped_topk` before
  writing a new router kernel. Current main mounts fused Triton RMSNorm row
  kernels by weight dtype and hidden size for `quality=extra-high` and
  `quality=high`; check the quality-site guards before treating an expanded
  `pow/mean/rsqrt` chain as a new opportunity.
- LTX-2.5 reuses the mature LTX-2 DiT paths. Treat the optional diffusion
  decoder separately: confirm NATTEN `na3d` is active, then inspect its
  per-block 3D RoPE construction and split QKV/SwiGLU projections.
- Cosmos3 Edge inherits the existing Cosmos3 attention-prep fusions. Profile
  the dense squared-ReLU MLP before proposing another Cosmos kernel, and do not
  repeat the closed experimental Cosmos BCG direction without solving its
  model-state lifecycle problem.

**Common Entry Points in Diffusion Models**
- AdaLN modulation: `LayerNormScaleShift`, `RMSNormScaleShift`, `ScaleResidual*` in `layernorm.py`.
- Bit-exact adaLN modulation / LayerNorm folding: `modulate_scale_shift` and
  `fused_layernorm_modulate` through `flux.py`, `glm_image.py`, and `sana.py`.
- Request-scoped extra-high/high acceleration: `QualityGatedFusion` in
  `quality_gate.py`, `_maybe_toggle_quality_fusions` in `denoising.py`, and
  `use_vae_fast_path` in `decoding.py`.
- Bit-exact first-sight verify/disable: `BitExactFusionGate` in
  `bitexact_gate.py`, used by FLUX / GLM / Sana / Ernie fused norm sites.
- Qwen-Image gating: `fuse_layernorm_scale_shift_gate_select01_kernel` and `fuse_residual_layernorm_scale_shift_gate_select01_kernel` through `fused_scale_shift_gate.py` and `qwen_image.py`.
- Z-Image native norm modulation: `rmsnorm_scale` and `rmsnorm_tanh_residual`
  in `triton/native_bf16_rmsnorm.py`, with wrappers in `zimage.py` /
  `fused_gate_rmsnorm.py`. `zimage_native_norm.py` is QK-only.
- HunyuanVideo VAE and LTX upsampler GroupNorm+SiLU: `apply_group_norm_silu` in `hunyuanvae.py` and `latent_upsampler.py`; default-eligible when wrapper guards pass.
- MiniMax-H3 indexed modulation: `_modulate_scale_shift` and `_modulate_gate` in `minimax_h3.py`, backed by `triton/indexed_modulation.py`.
- MiniMax-H3 Ulysses relayout: `_usp_input_all_to_all_packed_qkv` and `usp_merge_heads` through `runtime/layers/usp.py`.
- QK norm: `apply_qk_norm` used in `flux.py`, `flux_2.py`, `qwen_image.py`, `zimage.py`, `wanvideo.py`, `ltx_2.py`, `hunyuanvideo.py`.
- QK norm + RoPE: `apply_qk_norm_rope` in `layernorm.py`; use this path when the model wants fused attention prep instead of separate QK norm and RoPE calls.
- LTX2 split RoPE: `apply_ltx2_split_rotary_emb` in `ltx_2.py`.
- LTX2 RMSNorm+modulate and FFN GELU epilogue under `quality="extra-high"` and `quality="high"`:
  `mark_ltx2_rms_norm_modulate_site` / `fused_ltx2_rms_norm_modulate` in
  `kernels/ops/diffusion/sites/ltx2_rmsnorm_modulate_site.py` (mount-based
  `QualityGatedFusion`, not a first-sight `BitExactFusionGate` — the fused
  kernel is <=1 ULP off aten, so it is request-gated instead of verified),
  wired at the six `LTX2TransformerBlock` adaLN sites in `ltx_2.py`.
- Shared residual-gate add: `ltx_2.py`, `sana.py`, and `sana_video.py` call `residual_gate_add` from
  `kernels/ops/diffusion/modulate/residual_gate_add_jit.py` directly for attention,
  cross-attention, and MLP residual updates; SANA-Video's transposed residual
  uses the mixed-layout tiled kernel without an intermediate contiguous copy.
- Wan causal VAE: `cat_pad_channels_last_3d` and `dup_up3d_add` in
  `wanvae.py`, backed by `triton/wan_causal_cache.py`.
- Varlen USP attention: `fused_pack_qkv` and `fused_scatter_to_padded` in `attention/layer.py`.
- SANA packed projections: `to_qkv` and `to_kv` in `sana.py`.
- Nunchaku fused GELU MLP: `_fused_gelu_mlp` in `flux.py` for quantized FLUX-family checkpoints.
- NVFP4 / packed QKV attention: `to_qkv`, `to_added_qkv`, and `to_qkv_mlp_proj` in FLUX-family quantized paths.
- RoPE: `_apply_rotary_emb` prefers Triton; Q/K RoPE prefers FlashInfer when present.

**Existing Overlap / Communication Families**

- Ulysses / USP attention: treat `all_to_all`, `ring_attn`, and head / sequence reshards as an existing distributed attention family, not a new overlap idea.
- Cross-node SP: current server args support `--nnodes`, `--node-rank`, and
  `--dist-init-addr`. Prefer node-local Ulysses multiplied by cross-node Ring;
  keep encoders replicated and verify each model's Ring admission before
  treating cross-node transport as a new framework gap.
- MiniMax-H3 TP AdaLN: the DiT stacks every block's TP-local AdaLN projection and performs one batched all-gather before the block loop when `_can_batch_block_adaln()` passes. One all-gather per block indicates that this existing batching path missed.
- MiniMax-H3 final projections: H3 removes dead text/padding rows before the final TP column gathers and combines video/audio for the SP row gather. Preserve that ordering when optimizing output communication.
- Turbo-layer async all-to-all: `all_to_all_single(..., async_op=True)` plus staged waits already form an existing overlap family in `turbo_layer.py`.
- TorchInductor compute / communication reorder: `torch._inductor.config.reorder_for_compute_comm_overlap = True` can already partially overlap compiled denoise traces.
- Breakable CUDA graph: `runtime/breakable_cuda_graph/runner.py` captures
  fixed-resolution DiT segments around eager attention/collectives for
  supported pipelines. It is mutually exclusive with `torch.compile` and
  Cache-DiT. The model's default resolution is captured automatically; put
  every additional served resolution in `--warmup-resolutions`, and use
  `--bcg-text-buckets` for prompt signatures. Check this path before proposing
  a second graph-capture mechanism for launch-bound traces.
- A valid BCG benchmark must show `[Diffusion BCG] captured` and no support
  disable, capture failure, `serving signature MISSED`, or eager-fallback
  marker. Width and height are not the whole signature: public
  `--warmup-resolutions` does not override a video model's synthetic warmup
  frame count, so a short profiling request can capture the default temporal
  shape and then miss during serving. Reject that timing instead of labeling
  it BCG.
- LongCat-Image uses this generic runner directly: one 1024x1024 capture covers
  short and long prompts because text conditioning is fixed at 512 tokens.
  Keep eager as the baseline because the gain is hardware-dependent; an H200
  50-step, three-prompt run measured 177.0--177.3 ms/step eager versus
  173.1--173.3 ms/step with BCG, with bit-exact final images.
- SANA-Video uses this runner directly at declared 832x480 resolutions. Its
  default text pipeline always emits 300 prompt slots, so one graph covers
  different raw prompt lengths without padding cross-attention to 512 slots.
- Dual-stream diffusion models: `use_dual_stream = True` in models such as `hunyuan3d.py` is an existing overlap family.
- Workflow rule: if a hotspot is communication-heavy, rule out these in-repo overlap families before proposing a brand new overlap design.

**Historical PR Watchlist**

These SGLang PRs are useful as upstream direction and prior art, not as
current-main behavior. Re-check the PR state and the active source tree before
relying on any file path, flag, or claim about whether the work has merged.

- Norm, modulation, and packed projection fusions:
  - #24025 LTX2 QK norm fusion.
  - #24059 Helios fused norm modulation.
  - #24117 Z-Image packed QKV.
  - #19488 Wan cross-block elementwise fusion.
  - #19249 Z-Image `scale residual norm scale shift` plus `add gate norm` fusion.
  - #18897 dual norm fusion for FLUX-family paths (draft).
  - #20429 Qwen-Image layernorm and `fuse_scale_shift_gate_select01` work.
  - #20530 MOVA fused RMSNorm + interleaved RoPE.
  - #29361 LTX2 residual-gate CUDA fast path for `residual + update * gate`.
  - #34172 LTX2 quality-high fusion; #34305/#34314 Ideogram eager fusions.
  - #34584 Wan TI2V modulation/RoPE; #34616 FLUX2; #34617 Hunyuan;
    #34619 GLM; #34620 ERNIE; #34928 SANA; #34932 Cosmos3; #35728
    SANA-Video linear attention.
  - SANA-Video shared-kernel reuse and LingBot request-gated RMSNorm are now
    current-main fast paths; verify the source tree before treating their
    historical PRs as open work.
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
  - #34174 automatic default-resolution BCG warmup; #34210 Z-Image BCG
    correctness; #34929 LTX2.3 BCG; #35724 LongCat-Image BCG; #35729
    SANA-Video fixed-300-token BCG. #34618 is a closed Cosmos BCG experiment,
    not a reusable mainline fast path.

**Constraints and Fallbacks**
- `scale_shift` Triton requires CUDA + contiguous `x`. NPU swaps to native.
- Bit-exact BF16 LayerNorm+modulate requires the guarded aten-compatible shape
  and a successful live equality check; request-gated affine folding is a
  separate non-bit-exact path.
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
- If none of the families above match, package the evidence from the benchmark/profile skill and hand the kernel work to the appropriate kernel, Nsight, or framework-specific optimization workflow.
