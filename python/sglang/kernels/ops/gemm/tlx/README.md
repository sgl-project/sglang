# Vendored gfx950 TLX GEMM core

Copied from [facebookexperimental/triton](https://github.com/facebookexperimental/triton)
(MIT licence) at commit `2e8256a9b`:

| File here | Upstream path | Local edits |
|---|---|---|
| `mm_gfx950.py` | `third_party/tlx/ops/kernels/mm/gfx950.py` | `InvalidInput` from `._compat`, `gemm_fixed_split_k` from `._invariance` |
| `_invariance.py` | `third_party/tlx/ops/_invariance.py` | `InvalidInput` from `._compat` |

`__init__.py`, `_runtime.py` and `_compat.py` are local. Nothing else is
changed, so re-syncing is a copy plus the import edits above; the pre-commit
config excludes the two copied files so formatting never diverges from upstream.

The entry point is registered for inventory as `gemm.tlx_linear`.

## Runtime

The kernel needs `triton.language.extra.tlx`. On stock ROCm Triton, install the
uTLX plugin built for that Triton release; `_runtime.py` imports `utlx_plugin` on
first use. Verified on MI355X with torch 2.14.1+rocm7.2, which pins
`triton-rocm 3.8.0`:

    pip install --no-deps triton-utlx==3.8.0.post16

`--no-deps` because that wheel declares `triton~=3.8.0` (the CUDA package name).
`triton-utlx-rocm==3.8.0.post1` imports but fails at the first kernel compile:
its `libutlx.so` needs `mlir::argSort`, which `triton-rocm 3.8.0` does not export.

`mm_gfx950.py` reads two fbtriton-only instrumentation knobs
(`triton.knobs.runtime.sanitize_overflow`,
`triton.knobs.compilation.fpsan_homomorphic_casts`) to decide whether its
compiled-cache fast launch is safe. `_compat.add_missing_triton_knobs()` defines
them as `False` on stock Triton, which has no such instrumentation.

## Integration

`SGLANG_USE_TLX_GEMM=1` routes BF16/FP16 unquantized linears without bias on
gfx950 to `linear` (`srt/layers/quantization/unquant.py`), ahead of every other
dense GEMM path. Bias, quantized weights, non-contiguous inputs, deterministic
mode, and bf16 single-row inputs stay on the existing paths. Startup fails if no
TLX (uTLX or fbtriton) is installed.

The bf16 single-row fallback exists because `mm`'s packed GEMV uses
`tlx.amd_packed_dot2`, which uTLX does not provide (checked through
`triton-utlx 3.9.0.post4`). Projections a model runs on FP8 weight copies (the ROCm Muse
Spark QKV / O / unpooler paths) do not go through `unquant.py` and are unaffected.

## Tests

`python/test/unit/tlx_ops/test_mm_gfx950.py` in fbtriton is the authoritative
correctness suite.
