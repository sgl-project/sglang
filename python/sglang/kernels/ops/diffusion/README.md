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
free to move; the facade is not. `test_import_surface.py` enforces this, with
a small allowlist for tests that deliberately exercise one backend.

Resolution is lazy (PEP 562): the backends have disjoint heavy dependencies
(Triton, CUTLASS/CuTe-DSL, and FlyDSL on ROCm), so an eager
re-export would make all of them import-time requirements everywhere.

## Layout

One subpackage per **operator domain**; the backend is a **filename suffix**
(`_triton`, `_jit`, `_cutedsl`, `_flydsl`, or `_bitexact` where that says more).
This matches `ops/attention` and `ops/gemm`, and it keeps every implementation
of one logical op in one directory.

```
norm/        RMSNorm / LayerNorm / GroupNorm and their fused epilogues
modulate/    adaLN modulate, gating, timestep conditioning
rope/        rotary embeddings and the QK-norm chains fused into them
activation/  SiLU / GLU / GELU fusions
attention/   sparse linear attention, gated delta-net
layout/      pure data movement: USP/Ulysses relayout, varlen pack, causal pad
common/      numerics primitives, platform predicates, non-Triton fallbacks
sites/       request-scoped mount policy — NOT kernels (see below)
ext/         JIT C++/CUDA extensions (Hunyuan3D raster/inpaint) — NOT kernels
```

## The two numerical contracts

**Bit-exact (`torch.equal` vs the eager chain) → mounted unconditionally.**
These kernels reproduce every aten rounding boundary, sometimes down to the
reduction tree: `norm/layernorm_modulate_triton.py` replicates torch 2.11's
`vectorized_layer_norm_kernel` (128-thread Welford, `_rcp4` guarded
reciprocal, `shfl.down` fold order, `div.rn` + `MUFU.RSQ`), and
`norm/rmsnorm_scale_shift_bitexact.py` replicates flashinfer's CuTe-DSL
`RMSNormKernel` fragment order and `shfl.bfly` fold. They still verify
themselves against the live eager chain on first sight via
`sites/bitexact_gate.py` and fall back permanently on mismatch — the
dispatch they replicate can change under them.

**Not bit-exact → quality-gated.** Mounted onto marked `nn.Module` sites only
for `quality="high"` requests, at batch boundaries, all-or-nothing per
transformer (`sites/quality_gate.py`). A plain fp32 single-pass norm fusion
looks harmless and is not: on ERNIE-Image it moved the 50-step trajectory to
PSNR 18.83 dB at `quality=high`, which is what motivated the bit-exact
rewrite.

SANA-Video's quality-gated linear-attention site keeps BF16 inputs for the
first GEMM while requesting FP32 accumulation/output, then runs the second
GEMM in FP32. The default path still promotes Q/K/V before both GEMMs.

## Entry-point protocol

Every public kernel is a **predicate + kernel** pair:

```python
if can_use_<op>(...):
    out = <op>(...)
else:
    out = <reference chain>
```

The kernel raises on an unsupported input. It does not return `None` — a
silent `None` is too easy to forget to check, and the failure mode is a
wrong-looking image rather than an exception.

## Selection matrix

Several norms look interchangeable and are not. Start here.

### Norm + scale/shift (adaLN)

| Entry point | Backend | Contract | Applies to |
|---|---|---|---|
| `fused_rmsnorm_scale_shift_bitexact` | Triton | bit-exact vs flashinfer CuTe RMSNorm + aten modulate | bf16, contiguous rows, `H == 64 * threads_per_row` |
| `fused_scale_residual_rmsnorm_scale_shift_bitexact` | Triton | bit-exact, incl. the preceding residual-gate add | as above |
| `fused_layernorm_modulate` | Triton | bit-exact vs aten `vectorized_layer_norm` | bf16, `N % 4 == 0`, 16B-aligned |
| `fused_norm_scale_shift` / `fused_scale_residual_norm_scale_shift` | CuTe-DSL | fp32 statistics, close | fp16/bf16/fp32, LN or RMS, many broadcast modes |
| `flydsl_norm_scale_shift` / `flydsl_fused_residual_norm_scale_shift` | FlyDSL | close | **ROCm gfx950 only** |
| `fuse_layernorm_scale_shift_gate_select01_kernel` | Triton | close | per-token select between two modulation rows (Qwen-Image) |
| `norm_infer` / `rms_norm_fn` | Triton (+torch/NPU/MPS fallbacks) | close | the generic entry point; use when nothing above fits |

### Norm variants

| Entry point | Backend | Contract | Applies to |
|---|---|---|---|
| `triton_group_norm_silu` / `apply_group_norm_silu` | Triton | close | NCHW-contiguous, any channels-per-group, always applies SiLU |
| `group_norm_silu_4d` / `group_norm_silu_rows` | Triton | close | **channels_last only**; power-of-two `C <= 2048`; optional SiLU. This is what lets a VAE decoder run channels_last end-to-end with no `nchwToNhwc` |
| `wan_rmsnorm_silu` | Triton | close | dense `channels_last_3d` 5D (`stride(C) == 1`), Wan VAE channel-first RMSNorm + SiLU |
| `rmsnorm_scale` / `rmsnorm_tanh_residual` | Triton | bf16-native statistics | Z-Image (matches its own reference exactly), Ideogram 4 (gated) |
| `zimage_qk_rmsnorm_native` | Triton | bit-exact | Z-Image per-head QK RMSNorm |
| `fused_qk_head_layernorm` | Triton | bit-exact | per-head LN on q/k, `dim_head % 4 == 0`, `<= 128` |
| `triton_one_pass_rms_norm` | Triton | close | standalone RMSNorm, one pass |

### RoPE / QK-norm

| Entry point | Backend | Contract |
|---|---|---|
| `fused_inplace_qknorm_rope` | JIT CUDA | one bf16 rounding step vs split baseline; `round_norm_before_rope=True` makes it exact; supports compact and full-width NeoX/interleaved caches |
| `fused_qknorm_rope_pack_kv` | JIT CUDA | as above, also packs prefix K/V |
| `fused_rope_rotate_half_bitexact` | Triton | bit-exact (elementwise only) |
| `fused_interleaved_rope_fp64` | JIT CUDA | bit-exact vs paired SANA-Video fp64 RoPE |
| `ltx2_qknorm_split_rope_cuda` | JIT CUDA | close; **validated on B200** |
| `fused_ltx25_decoder_rope` | JIT CUDA | bit-exact paired 3D RoPE from cached compact axis tables |
| `apply_rotary_embedding` | Triton (+fallbacks) | close; the generic entry point |
| `hunyuan_qkv_rope_pack` | Triton | bit-exact; packs QKV and applies RoPE in one pass |

### Data movement (all bit-exact by construction)

`usp_merge_heads`, `pack_qkv_destination_major`, `fused_pack_qkv`,
`fused_scatter_to_padded`, `fused_causal_conv3d_cat_pad_cuda`,
`cat_pad_channels_last_3d`, `dup_up3d_add`, `fused_temb_table_slices`,
`ltx2_ada_values9`.

`fused_temb_table_slices` is worth knowing about: the eager
`(table + temb.float()).chunk(6, dim=2)` materializes ~8 GB of fp32 at
704p/121f *and* hands six strided slices downstream, whose `.contiguous()`
calls copy each one again.

## What is not a kernel

`sites/` rewrites `nn.Module` trees (mark / mount / unmount) and `ext/` builds
C++/CUDA extensions that have no backend dimension and no numerical contract.
They live here because they are diffusion-specific and share this package's
build machinery, but they are deliberately in their own directories: nothing
in `sites/` or `ext/` belongs in an operator domain, and `sites/` is the one
place allowed to reference `multimodal_gen` types (lazily, inside functions) —
inspecting model modules is its whole job.

## Adding a kernel

1. Put it in the operator domain it belongs to, with a backend suffix.
2. Export it from `__init__.py` (`_EXPORTS`) and register a `KernelSpec`
   (`_SPECS`) — `test_import_surface.py` checks both resolve.
3. Give it a `can_use_*` predicate; raise, don't return `None`.
4. State the numerical contract in the module docstring, including which
   shapes it was verified on.
5. If it is not bit-exact, gate it through `sites/`. Do not mount it by
   default.
6. Test it in the domain suite (`test/registered/kernels/ops/diffusion/`), and
   the model wiring in `test_model_fast_paths.py`.
