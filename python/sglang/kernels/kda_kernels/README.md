# Kernel Design Agent kernels

This directory is the implementation home for kernels produced or extended by
the Humanize2 / Kernel Design Agents workflow. `KernelBackend.KDA` records that
provenance; it does not identify the implementation language. A KDA kernel may
use CUDA, Triton, or CuTe DSL.

Runtime code must continue to import the stable operator facade under
`sglang.kernels.ops`. The facade owns registration and fallback policy, while
this directory owns generated implementation modules and their CUDA sources.
Importing `sglang.kernels` therefore remains metadata-only and does not eagerly
load Triton, CUTLASS, or compile a JIT extension.

| Kernel family | Implementation | Provenance |
|---|---|---|
| Qwen3.x ModelOpt NVFP4 GEMM on SM120 | `qwen3x_nvfp4_gemm_sm120.py` | [sgl-project/sglang#36865](https://github.com/sgl-project/sglang/pull/36865), merge commit `c593527f33` |
| ModelOpt static per-tensor FP8 small-batch dispatch on SM12x | `sm120_fp8.py`, `sm120_fp8_skinny_gemm_sm120.py`, `csrc/gemm/sm120_fp8_skinny_gemm.cuh` | [sgl-project/sglang#38082](https://github.com/sgl-project/sglang/pull/38082) |
| Qwen-Image norm / residual-norm scale-shift | `norm_scale_shift_jit.py` | [sgl-project/sglang#27392](https://github.com/sgl-project/sglang/pull/27392), merge commit `26e1d4d847` |
| Cosmos3 causal Conv3D cat-pad | `causal_conv3d_cat_pad_jit.py` | [sgl-project/sglang#29281](https://github.com/sgl-project/sglang/pull/29281), merge commit `5996b54bd3` |
| Diffusion residual-gate add | `residual_gate_add_jit.py` | [sgl-project/sglang#29361](https://github.com/sgl-project/sglang/pull/29361), merge commit `495f13fa12` |
| LTX2 QK-norm split-RoPE | `ltx2_qknorm_split_rope_jit.py` | [sgl-project/sglang#29708](https://github.com/sgl-project/sglang/pull/29708), merge commit `fcb9f229b3` |
| FLUX.2 FP8 producer and QKV packing fusions | `layernorm_modulate_triton.py`, `flux2_qkv_epilogue_jit.py`, `flux2_token_cat_fp8_triton.py` | [sgl-project/sglang#37162](https://github.com/sgl-project/sglang/pull/37162), merge commit `1c3ad92438` |

For JIT kernels, the Python entry module and the corresponding source under
`csrc/` move together. The shared `sglang.kernels.jit` loader remains build
infrastructure rather than an ownership directory.
