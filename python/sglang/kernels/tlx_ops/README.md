# Vendored fbtriton `tlx.ops`

Every file in this directory except this README and the empty `__init__.py` is
copied from `third_party/tlx/ops/` in
[facebookexperimental/triton](https://github.com/facebookexperimental/triton)
(MIT licence) at commit `2e8256a9b`, at the same relative path. SGLang's own code
for these kernels lives outside this directory: the wrappers and dispatch in the
operator groups (`kernels/ops/<group>/tlx.py`) and the stock-Triton runtime
support in `kernels/jit/utils/tlx_runtime.py`.

| File | Local edits |
|---|---|
| `_catalog.py` | none (supplies `InvalidInput`; its op table is unused here) |
| `_invariance.py` | none |
| `kernels/__init__.py`, `kernels/mm/__init__.py`, `kernels/moe/__init__.py` | none |
| `kernels/mm/gfx950.py` | none |
| `kernels/moe/gfx950.py` | `_register_matmul_core` imported from `sglang.kernels.tlx_ops.kernels.mm.gfx950` instead of `triton.tlx.ops.kernels.mm.gfx950` |

To re-sync, copy those files from the new upstream commit, re-apply the one
import edit, and update the commit above. The pre-commit config excludes this
directory so formatting never diverges from upstream.
