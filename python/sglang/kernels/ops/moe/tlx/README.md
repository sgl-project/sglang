# Vendored gfx950 TLX MoE

Copied from [facebookexperimental/triton](https://github.com/facebookexperimental/triton)
(MIT licence) at commit `2e8256a9b`:

| File here | Upstream path | Local edits |
|---|---|---|
| `moe_gfx950.py` | `third_party/tlx/ops/kernels/moe/gfx950.py` | `_register_matmul_core` imported from `sglang.kernels.ops.gemm.tlx.mm_gfx950` |

`__init__.py` is local. The kernel is built on the vendored `tlx.ops.mm` GEMM
core in `sglang/kernels/ops/gemm/tlx/`, whose README covers the uTLX runtime
pairing and the stock-Triton shims. Nothing else is changed, so re-syncing is a
copy plus the import edit above; the pre-commit config excludes the copied file
so formatting never diverges from upstream.

The entry point is registered for inventory as `moe.tlx_moe`.

## Integration

`MUSE_MOE_BACKEND=tlx` routes BF16 ROCm Muse Spark experts to `moe`
(`srt/rocm_muse_spark/model.py`). MXFP4 experts, shapes outside the kernel's
limits (see `moe_supported`), and a post-expert norm without a weight fall back
to FlyDSL with a one-time warning. The first TLX MoE call raises if no TLX
(uTLX or fbtriton) is installed. Only gfx950 is supported, and nothing checks
the architecture.

## Tests

`python/test/unit/tlx_ops/test_moe_gfx950.py` in fbtriton is the authoritative
correctness suite.
