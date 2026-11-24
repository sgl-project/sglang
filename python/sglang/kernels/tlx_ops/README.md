# Vendored fbtriton `tlx.ops`

Every file in this directory except this README and the empty `__init__.py` is
copied from `third_party/tlx/ops/` in
[facebookexperimental/triton](https://github.com/facebookexperimental/triton)
(MIT licence) at the commit listed below, at the same relative path. SGLang's own
code for these kernels lives outside this directory: the wrappers and dispatch in
the operator groups (`kernels/ops/<group>/tlx.py`, or beside the SGLang kernels
of the same contract, as `gemm/router_gemm_fp32.py:router_gemm_tlx`) and the
stock-Triton runtime support in `kernels/jit/utils/tlx_runtime.py`.

| File | Upstream commit | Local edits |
|---|---|---|
| `_catalog.py` | `2e8256a9b` | none (supplies `InvalidInput` and `UnsupportedOp`; its op table is unused here) |
| `_invariance.py` | `2e8256a9b` | none |
| `kernels/__init__.py`, `kernels/mm/__init__.py`, `kernels/moe/__init__.py` | `2e8256a9b` | none |
| `kernels/mm/gfx950.py` | `2e8256a9b` | none |
| `kernels/moe/gfx950.py` | `2e8256a9b` | `_register_matmul_core` imported from `sglang.kernels.tlx_ops.kernels.mm.gfx950` instead of `triton.tlx.ops.kernels.mm.gfx950` |
| `kernels/fused/__init__.py` | `87abb3f5a` | none |
| `kernels/fused/norm_residual_norm/__init__.py`, `gfx950.py` | `87abb3f5a` | none |
| `kernels/fused/qk_norm_rope_kv_cache/__init__.py`, `gfx950.py` | `87abb3f5a` | none |
| `kernels/norm/__init__.py`, `gfx950.py` | `87abb3f5a` | none |
| `kernels/router_gemm/__init__.py`, `gfx950.py` | `87abb3f5a` | none |

Upstream's `kernels/moe/routing_gfx950.py` (`tlx.ops.moe_routing`) is not
vendored: with triton-utlx 3.9.0.post4 its `tl.where` on a `tlx.require_layout`
tensor and a plain one fails `arith.select` verification on stock Triton.

To re-sync, copy those files from the new upstream commit, re-apply the one
import edit, and update the commits above. The pre-commit config excludes this
directory so formatting never diverges from upstream.
