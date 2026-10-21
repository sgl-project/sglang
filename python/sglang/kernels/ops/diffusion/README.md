# `sglang.kernels.ops.diffusion`

Fused kernels for diffusion (multimodal-generation) models — DiT transformer
blocks, VAE encoders/decoders, and the sequence-parallel plumbing around them.

Unlike the LLM operator groups, almost nothing here is a general-purpose
operator. Each kernel replaces a **specific eager op chain in a specific
model**, and its value comes as much from *which rounding boundaries it
reproduces* as from its bandwidth. Multi-step denoising amplifies a per-step
rounding difference into visible quality loss, so "close enough" is a
different product from "bit-exact", and the two are gated differently.

## Import surface

```python
from sglang.kernels.ops.diffusion import fused_rmsnorm_scale_shift_bitexact
```

**Import from the package, never from a submodule.** The internal layout is
free to move; the facade is not. Callers should use the facade, with
a small allowlist for tests that deliberately exercise one backend.

Resolution is lazy (PEP 562): the backends have disjoint heavy dependencies
(Triton, CUTLASS/CuTe-DSL, and FlyDSL on ROCm), so an eager
re-export would make all of them import-time requirements everywhere.

## Layout

Ordinary implementations use one subpackage per **operator domain**; the
compiler is a **filename suffix** (`_triton`, `_jit`, `_cutedsl`, `_flydsl`, or
`_bitexact` where that says more). Implementations with Kernel Design Agents
provenance live under `sglang.kernels.kda_kernels`; this facade remains their
only supported runtime import surface.

```
norm/        RMSNorm / LayerNorm / GroupNorm and their fused epilogues
modulate/    adaLN modulate, gating, timestep conditioning
rope/        rotary embeddings and the QK-norm chains fused into them
activation/  SiLU / GLU / GELU fusions
quantization/ FP8 rowwise producers fused into the GEMM input path
attention/   sparse linear attention, gated delta-net
routing/     diffusion-model MoE routing and expert selection
layout/      pure data movement: USP/Ulysses relayout, varlen pack, causal pad
common/      platform predicates, non-Triton fallbacks, cached Triton launches (numerics: kernels/numerics.py)
sites/       request-scoped mount policy — NOT kernels (see below)
ext/         JIT C++/CUDA extensions (Hunyuan3D raster/inpaint) — NOT kernels
../../kda_kernels/  agent-generated implementations and their JIT CUDA sources
```

## Numerical contracts and quality policy

**Bit-exact (`torch.equal` vs the eager chain) → mounted unconditionally.**
These kernels reproduce every aten rounding boundary, sometimes down to the
reduction tree: `../../kda_kernels/layernorm_modulate_triton.py` replicates torch 2.11's
`vectorized_layer_norm_kernel` (128-thread Welford, `_rcp4` guarded
reciprocal, `shfl.down` fold order, `div.rn` + `MUFU.RSQ`), and
`norm/rmsnorm_scale_shift_bitexact.py` replicates flashinfer's CuTe-DSL
`RMSNormKernel` fragment order and `shfl.bfly` fold. They still verify
themselves against the live eager chain on first sight via
`sites/bitexact_gate.py` and fall back permanently on mismatch — the
dispatch they replicate can change under them.

**Not bit-exact → quality-gated.** Mounted onto marked `nn.Module` sites at
batch boundaries, all-or-nothing per transformer (`sites/quality_gate.py`),
from the lowest tier that may run each one: `lossless` for a fusion that keeps
the reference math and every operand's precision and only moves the rounding,
`high` for one that quantizes or lowers a precision. `high` is cumulative and
may also enable model-owned approximate paths such as Cache-DiT or a lower-precision
decode. A plain fp32 single-pass norm fusion looks harmless and is not: on
ERNIE-Image it moved the 50-step trajectory to PSNR 18.83 dB, which is what
motivated the bit-exact rewrite.

**Model/checkpoint-native.** Generic close-contract kernels, sparse operators,
and FP8/NVFP4 producers can belong to the selected model or deployment path.
The request `quality` tier neither selects nor disables those independent
choices.

SANA-Video's quality-gated linear-attention site keeps BF16 inputs for the
first GEMM while requesting FP32 accumulation/output, then runs the second
GEMM in FP32. The default path still promotes Q/K/V before both GEMMs.

## Entry-point protocol

Keep backend selection separate from input validation.

- Select a backend using properties that distinguish supported implementations:
  device family, dtype, layout, alignment, architecture, and numerical contract.
  A shape restriction belongs here only when it is a real specialization, such
  as a fixed head dimension or an unsupported broadcast mode.
  Use the input's device when selecting CPU versus CUDA: a CUDA-capable
  process can still receive CPU tensors. A CPU fallback must not disable a
  process-wide GPU fusion gate.
- Validate the selected implementation's inputs before launching. For JIT CUDA,
  use `TensorMatcher` and `CHECK_HOST` in the C++ launcher; keep dtype checks in
  the cached module factory. For Triton, validate in the Python launch wrapper.
  Mismatched shapes or devices are errors, not reasons to silently try another
  backend. Preserve tensor dimensions until they have been validated.
- Add a `can_use_*` predicate only when a caller needs a backend choice. A direct
  kernel entry point does not need a second copy of the same predicate. Keep
  fallback selection in one wrapper when several callers share it.
- Let errors from a selected implementation propagate. Do not catch every
  exception and permanently disable a kernel for later requests. First-use
  numerical verification in `sites/` is a separate policy: retain its reference
  comparison when exactness depends on a library's reduction or rounding order.

For example, Helios selects JIT CUDA RoPE on NVIDIA GPUs unless it uses TP
RMSNorm. The launcher validates the original batched Q/K and frequencies,
including shape relationships, dtype, device, layout and pair alignment.
There is no Python `can_use_helios_qk_rope` scan before the call.

A numerical specialization query should take configuration values, not a list
of tensors. For example, `can_use_fused_rmsnorm_scale_shift(dtype, hidden)`
selects BF16 with width 2048 or 4096 because those are the FlashInfer reduction
trees the kernel reproduces. Its launcher validates weight, scale and shift.

Existing `try_*` entry points return `None` for an unsupported specialization.
Keep that convention explicit at their call sites; do not add it to direct
kernel entry points.

## Selection matrix

Several norms look interchangeable and are not. Start here.

### Norm + scale/shift (adaLN)

| Entry point | Backend | Contract | Applies to |
|---|---|---|---|
| `rmsnorm_cast_before_weight_modulate_bitexact` | Triton | ATen-ordered FP32 reduction and exact BF16 modulation | MOSS-Image BSC, width 3072 |
| `fuse_scale_shift_kernel` | Triton | close | contiguous BLC; scalar/row/token modulation plus causal-video `[B, F, 1, C]`, using a static capped power-of-two tile to avoid request-time autotuning |
| `fused_rmsnorm_scale_shift_bitexact` | Triton | bit-exact vs flashinfer CuTe RMSNorm + aten modulate | bf16, contiguous rows, `H == 64 * threads_per_row` |
| `fused_scale_residual_rmsnorm_scale_shift_bitexact` | Triton | bit-exact, incl. the preceding residual-gate add | as above |
| `fused_layernorm_modulate` | Triton | bit-exact vs aten `vectorized_layer_norm` | bf16, `N % 4 == 0`, 16B-aligned |
| `fused_norm_scale_shift` / `fused_scale_residual_norm_scale_shift` | CuTe-DSL | fp32 statistics, close | fp16/bf16/fp32, LN or RMS, many broadcast modes |
| `flydsl_norm_scale_shift` / `flydsl_fused_residual_norm_scale_shift` | FlyDSL | close | **ROCm gfx950 only** |
| `try_fused_scale_residual_norm_scale_shift_nvfp4` | JIT CUDA | matches the selected NVFP4 producer contract | Qwen residual LayerNorm/modulation + FC1 NVFP4 quantization |
| `fuse_layernorm_scale_shift_gate_select01_kernel` | Triton | close | per-token select between two modulation rows (Qwen-Image) |
| `indexed_scale_shift_bf16_` / `indexed_scale_shift_fp32_` | Triton | bit-exact vs `x * (1 + scale[idx]) + shift[idx]`, in place | bf16 rows, per-row index into `[n, D]` shift/scale tables; fp32 tables are serialized INT8 MiniMax-H3 |
| `norm_infer` / `rms_norm_fn` | Triton (+torch/NPU/MPS fallbacks) | close | the generic entry point; use when nothing above fits |

### Norm variants

| Entry point | Backend | Contract | Applies to |
|---|---|---|---|
| `rmsnorm_cast_before_weight_bitexact` | Triton / aten | FP32 reduction, BF16 round before learned scale | MOSS-Image token and head RMSNorm |
| `triton_group_norm_silu` / `apply_group_norm_silu` | Triton | close | NCHW-contiguous, any channels-per-group, always applies SiLU |
| `group_norm_silu_4d` / `group_norm_silu_rows` | Triton | close | **channels_last only**; power-of-two `C <= 2048`; optional SiLU. This is what lets a VAE decoder run channels_last end-to-end with no `nchwToNhwc` |
| `wan_rmsnorm_silu` | Triton | close | dense `channels_last_3d` 5D (`stride(C) == 1`), Wan VAE channel-first RMSNorm + SiLU; `[ROWS, C]` tiles (~3 TB/s at C = 96); optional `conv_bias` folds the preceding conv's bias with aten `add_` arithmetic |
| `channel_rmsnorm_finish_silu` | JIT CUDA | bit-exact: keeps the aten fp32 `norm(dim=1)` reduction and fuses the pointwise finish (`max(norm, 1e-12)`, divide, scale, gamma, `+0.0`) with SiLU | Qwen-Image 2.1 VAE, contiguous NCHW/NCDHW bf16, spatial `% 8 == 0` |
| `channel_rmsnorm_silu_nhwc` | JIT CUDA | close (own fp32 reduction over C in a lane group sized from C; optional fused conv bias with aten rounding) | Qwen-Image 2.1 VAE under `quality=extra-high`/`high`; dense channels_last, even `C <= 2048` |
| `rmsnorm_scale` / `rmsnorm_tanh_residual` | Triton | bf16-native statistics | Z-Image (matches its own reference exactly), Ideogram 4 (gated) |
| `zimage_qk_rmsnorm_native` | Triton | bit-exact | Z-Image per-head QK RMSNorm |
| `fused_qk_head_layernorm` | Triton | bit-exact | per-head LN on q/k, `dim_head % 4 == 0`, `<= 128` |
| `triton_one_pass_rms_norm` | Triton | close | standalone RMSNorm, one pass |

### Residual gating

| Entry point | Backend | Contract | Applies to |
|---|---|---|---|
| `bias_residual_add` | JIT CUDA | bit-exact `(y + bias[c]) + h` with aten's per-op bf16 rounding, for a conv that skipped its bias pass | channels_last-dense (`C % 8 == 0`) or NCHW/NCDHW contiguous (spatial `% 8 == 0`) bf16; `h` shares `y`'s layout (Qwen-Image 2.1 VAE residual blocks) |
| `residual_gate_add` | JIT CUDA (contiguous), Triton (transposed) | bit-exact `residual + update * gate` | contiguous tensors, or a transposed-dense `[B, tokens, hidden]` residual/output with contiguous update and row-broadcast gate (SANA-Video) |
| `residual_gate_fp32` | Triton | separate FP32 multiply/add, then cast to input dtype | contiguous `[B,S,D]`, strided `[B or 1,1,D]` gate; Kandinsky6 |
| `indexed_gate_bf16` / `indexed_gate_fp32` (`_` in place) | Triton | bit-exact vs `x + gate[idx] * other`: bf16 rounds the product, fp32 keeps a separate FP32 multiply/add | bf16 rows, per-row index into a `[n, D]` gate table (MiniMax-H3) |

These contracts differ: `residual_gate_fp32` does not round the product to
BF16/FP16, and disables FMA to preserve the eager FP32 rounding boundary.

The transposed-dense path tiles along the physical stride-1 token dimension
for coalesced residual reads and output writes. The contiguous layouts use
JIT CUDA. Do not insert a `.contiguous()` to reach that path: it adds a full
tensor copy at each residual site.

### Activations

| Entry point | Backend | Contract | Applies to |
|---|---|---|---|
| `fused_situ_softcap_mul_bitexact` | Triton | bit-exact vs MOSSImage's nine-op BF16 SiTU chain | CUDA BF16 `[B, S, D]`, dense rows or strided halves of a packed FFN projection; first-use verification in the model wrapper |

### RoPE / QK-norm

| Entry point | Backend | Contract |
|---|---|---|
| `fused_inplace_qknorm_rope` | JIT CUDA | one bf16 rounding step vs split baseline; `round_norm_before_rope=True` makes it exact; supports compact and full-width NeoX/interleaved caches |
| `fused_qknorm_rope_pack_kv` | JIT CUDA | as above, also packs prefix K/V |
| `fused_qknorm_rope_out_of_place` | JIT CUDA | as above, bit-equal to the in-place kernel; reads strided q/k and writes contiguous copies, inputs untouched (VDN-H3 keeps the raw q/k for its linear branch) |
| `try_fused_flux2_qkv_epilogue` | KDA (JIT CUDA) | bit-exact vs the selected BF16 chain | FLUX.2 QK RMSNorm + RoPE + joint QKV packing |
| `try_fused_qwen_qkv_epilogue` | JIT CUDA | bit-exact vs the selected BF16 chain | Qwen-Image QK RMSNorm + RoPE + joint QKV writes; SM90+ |
| `rmsnorm_cast_before_weight_rope_bitexact` | Triton | ATen-ordered RMSNorm plus exact BF16 rotate-half RoPE, MOSS-Image width 128 |
| `fused_rope_rotate_half_bitexact` | Triton | bit-exact (elementwise only) |
| `fused_complex_rope` | Triton | preserves CUDA complex64 multiply rounding for contiguous BSHD inputs; Qwen-Image 2.1 verifies its first call against eager |
| `rmsnorm_preserve_reduction` | Triton + aten | preserves the FP32 mean reduction and cast-before-weight rounding; fuses only pointwise work for FP16/BF16 inputs and materializes strided inputs as contiguous; Qwen-Image 2.1 verifies its first call |
| `qknorm_complex_rope_cuda` / `qknorm_complex_rope_pack_` | JIT CUDA | bit-exact vs `RMSNorm(cast_x_before_out_mul=True)` + complex64 RoPE for head_dim 128; both accept token-strided views of a packed QKV buffer; the pack variant normalizes Q and K in place behind a prefix and copies prefix K/V (and optionally V) in the same launch; Qwen-Image 2.1 verifies its first call |
| `fused_interleaved_rope_fp64` | JIT CUDA | bit-exact vs paired SANA-Video fp64 RoPE |
| `fused_inplace_helios_qk_rope` | JIT CUDA | bit-exact paired in-place RoPE for Helios' transposed frequency layout |
| `ltx2_qknorm_split_rope_cuda` | KDA (JIT CUDA) | close; **validated on B200** |
| `fused_ltx25_decoder_rope` | JIT CUDA | bit-exact paired 3D RoPE from cached compact axis tables |
| `apply_rotary_embedding` | Triton (+fallbacks) | close; the generic entry point |
| `hunyuan_qkv_rope_pack` | Triton | bit-exact; packs QKV and applies RoPE in one pass |

### MiniMax-H3 / VDN-H3 linear branch

| Entry point | Backend | Contract |
|---|---|---|
| `vdn_frame_stats_prep`, `vdn_gather_linear_state` | Triton | bit-exact (same products, fp32 gather) |
| `vdn_temporal_conv_act`, `vdn_silu_l2norm`, `vdn_linear_epilogue` | Triton | one rounding at the store, within one bf16 ulp of the eager chain; the model's own inference kernels, mounted unconditionally by the VDN-H3 branch |
| `vdn_delta_factors` | JIT CUDA | `(alpha * inv(I + A), B @ inv(I + A))` in one launch; same fp32 accuracy class as the cholesky + solve_triangular chain (cond-dominated); head_dim 128 |

### MXFP8 producers (online `mxfp8`, cuBLASLt block-scaled GEMM on SM100)

Defined in `sglang.kernels.ops.quantization.mxfp8_swizzled_triton` (`silu_mul_mxfp8` in
`sglang.kernels.ops.activation.swiglu_mxfp8_swizzled`, `indexed_scale_shift_mxfp8_` in
`modulate/indexed_scale_shift_mxfp8_triton.py`) and re-exported here.

| Entry point | Backend | Contract |
|---|---|---|
| `mxfp8_quantize_swizzled` | Triton | bit-exact vs `flashinfer.mxfp8_quantize(x, True)`: e4m3 payload + block-32 E8M0 scales in the `SWIZZLE_32_4_4` layout; weights at load and any bf16 GEMM input |
| `silu_mul_mxfp8` | Triton | bit-exact vs eager bf16 `silu(gate) * up` followed by the quantizer above; the fc2 input |
| `indexed_scale_shift_mxfp8_` | Triton | bit-exact vs `indexed_scale_shift_bf16_` followed by the quantizer above, optionally keeping the bf16 rows in place; the qkv / fc1 inputs |

### MoE routing

| Entry point | Backend | Contract | Applies to |
|---|---|---|---|
| `group_limited_topk` | Triton | selected expert-id set matches the guarded CUDA `torch.topk(..., sorted=False)` chain; output order is unspecified | LingBot Video's default-on sigmoid+bias group-limited routing; contiguous fp32 `[tokens, experts]`, at least two power-of-two experts per group |

### Data movement and quantized layout producers

`usp_merge_heads`, `pack_qkv_destination_major`, `fused_pack_qkv`,
`fused_pack_segmented_qkv`, `fused_scatter_to_padded`,
`fused_causal_conv3d_cat_pad_cuda`,
`cat_pad_channels_last_3d`, `dup_up3d_add`, `nearest_upsample_nhwc`,
`conv_bias_epilogue`, `fused_temb_table_slices`,
and `ltx2_ada_values9` are bit-exact data movement or same-order arithmetic.
`fused_layernorm_modulate_fp8_quant_raw` folds FLUX.2 LayerNorm, adaLN
modulation, and static FP8 quantization. `try_flux2_token_cat_fp8` and
`try_flux2_token_cat_nvfp4` fuse branch concatenation directly into the
quantized representation selected by the FLUX.2 checkpoint path.

`conv_bias_epilogue` exists because PyTorch's cuDNN conv path adds the bias as
a separate broadcast `add_` after `cudnn_convolution`, and on channels_last
outputs aten does not vectorise that add (1.5 TB/s vs 4.1 TB/s here on H200;
~10% of a Wan 2.1 decode). Run the conv without bias and apply the bias in one
pass with aten's exact arithmetic (fp32 add, one rounding), optionally fused
with the residual add that follows (`x.dtype(x.dtype(conv + b) + h)`, the two
roundings of the eager chain). The Wan decoder defers `conv1`'s bias into the
following norm (`WanRMS_norm.forward(x, conv_bias=...)`, absorbed by the fused
kernel under the gate) and fuses `conv2`'s bias with the block's residual add.

`nearest_upsample_nhwc` replaces `nn.Upsample(nearest / nearest-exact,
integer factor)` on a dense channels_last input with a Triton gather: same
values and layout as aten, but aten's own NHWC nearest kernel is several times
slower than its NCHW sibling, which is what the Wan-family VAE decoders hit
once they run channels_last end-to-end.

`fused_temb_table_slices` is worth knowing about: the eager
`(table + temb.float()).chunk(6, dim=2)` materializes ~8 GB of fp32 at
704p/121f *and* hands six strided slices downstream, whose `.contiguous()`
calls copy each one again.

### BF16 activation and normalization contracts

MOSS-Image's `fused_situ_softcap_mul_bitexact` uses exact BF16 lookup tables for inputs with
at least 65536 elements. The two factors are evaluated with the live aten CUDA
formula for every BF16 encoding, preserving each rounding boundary; the fused
kernel then looks up both factors and multiplies them. Tables use 256 KiB per
device and remain alive for captured graphs. Initialization finishes before
publication to other streams. Small inputs and captures without initialized
tables use the direct fused formula. This is an exact path, subject to the first-use bit-exact check.

MOSS-Image uses `rmsnorm_cast_before_weight_bitexact` for contiguous CUDA BF16 token and
per-head tensors. Widths 3072 and 128 with at least 16 rows and fewer than
`2**29` elements use one kernel that reproduces ATen's vectorized FP32 mean
reduction order; other shapes retain ATen's mean with fused pointwise
operations around it (three kernels).
Both round the normalized value to BF16 **before** multiplying by the learned
weight. The model verifies each width/reduction regime on first sight and
falls back on unsupported layouts or a mismatch. Unverified regimes use the
native path during compilation or CUDA graph capture.

MOSS-Image also uses `rmsnorm_cast_before_weight_modulate_bitexact` for width-3072 BSC
activations and `rmsnorm_cast_before_weight_rope_bitexact` for width-128 BSHD heads.
Both retain the same ordered reduction and BF16 round-before-gamma contract,
then preserve each modulation or rotate-half rounding boundary. Their guards
require contiguous CUDA BF16 tensors, at least 16 normalization rows, and
fewer than `2**29` elements. Modulation tensors may have a strided batch axis,
as produced by `unbind`, with contiguous channels. Model wrappers verify against the existing chain
before enabling either fusion and retain the separate operations on fallback.

## What is not a kernel

`sites/` rewrites `nn.Module` trees (mark / mount / unmount) and `ext/` builds
C++/CUDA extensions that have no backend dimension and no numerical contract.
They live here because they are diffusion-specific and share this package's
build machinery, but they are deliberately in their own directories: nothing
in `sites/` or `ext/` belongs in an operator domain, and `sites/` is the one
place allowed to reference `multimodal_gen` types (lazily, inside functions) —
inspecting model modules is its whole job.

## Adding a kernel

1. Put ordinary implementations in their operator domain. Put a kernel
   generated by the KDA workflow in `sglang.kernels.kda_kernels`, together
   with its source revision and any JIT CUDA source files.
2. Export it from `__init__.py` (`_EXPORTS`) and register a `KernelSpec`
   (`_SPECS`).
3. Keep selection and validation separate as described above. Reuse an existing
   wrapper when the fallback policy is shared.
4. State the numerical contract in the module docstring, including which
   shapes it was verified on.
5. If it is not bit-exact, gate it through `sites/` and declare its tier in
   `_QUALITY_FUSION_HANDLERS`: `lossless` when it keeps the reference math and
   every operand's precision, `high` when it does not. Never `exact`, which
   is reserved for paths that reproduce the reference's rounding. A
   `lossless` claim has to pass
   `multimodal_gen/test/quality_tier_admission.py`.
6. Test it in the domain suite (`test/registered/kernels/ops/diffusion/`), and
   the model wiring in `test_model_fast_paths.py`.
