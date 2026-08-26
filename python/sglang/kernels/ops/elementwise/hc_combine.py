from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_hc_combine_module(hc_count: int, hidden_size: int, dtype: torch.dtype) -> Module:
    """Compile and cache the JIT HC combine module for a given shape/dtype."""
    # Checks on the compile key live here, not in `hc_combine`: `cache_once`
    # keys on (hc_count, hidden_size, dtype), so this runs once per
    # specialisation instead of once per call.
    if dtype not in (torch.bfloat16, torch.float16):
        raise RuntimeError(
            f"Unsupported dtype {dtype}. Supported: bfloat16, float16"
        )
    if hidden_size <= 0 or hidden_size % 8 != 0:
        raise RuntimeError(
            f"Unsupported hidden_size {hidden_size}. Must be a multiple of 8."
        )
    if hc_count <= 0 or (hc_count * hidden_size) % 2048 != 0:
        raise RuntimeError(
            f"Unsupported hc_count * hidden_size {hc_count * hidden_size}. "
            "Must be a multiple of 2048."
        )
    args = make_cpp_args(hc_count, hidden_size, is_arch_support_pdl(), dtype)
    return load_jit(
        "hc_combine",
        *args,
        cuda_files=["elementwise/hc_combine.cuh"],
        cuda_wrappers=[
            ("hc_combine", f"HcCombineKernel<{args}>::run"),
            ("hc_combine_split", f"HcCombineSplitKernel<{args}>::run"),
        ],
    )


def hc_combine(
    block_output: torch.Tensor,
    residual: torch.Tensor,
    normed_residual: torch.Tensor,
    inject_weight: torch.Tensor,
    hc_count: int,
    hidden_size: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Fused HyperConnection (gated residual) combine.

        a[m, c] = 2 * sigmoid(dot(normed_residual[m], inject_weight[c]) / hc_count)
        out[m, c*H + i] = residual[m, c*H + i] + a[m, c] * block_output[m, i]

    Mirrors ``GatedResidual._combine_compute`` in
    ``sglang.srt.layers.hyperconnection``. All math is accumulated in fp32.

    Supported dtypes: torch.bfloat16, torch.float16.

    Parameters
    ----------
    block_output    : CUDA tensor [..., hidden_size]
    residual        : CUDA tensor [..., hc_count * hidden_size]
    normed_residual : CUDA tensor, same shape/dtype as residual
    inject_weight   : CUDA tensor [hc_count, hc_count * hidden_size]
    hc_count        : number of hyper-connection branches
    hidden_size     : per-branch hidden size
    out             : optional pre-allocated output tensor (same shape/dtype as residual)

    Returns
    -------
    Combined tensor, same shape/dtype as residual.
    """
    y = block_output.reshape(-1, hidden_size)
    r = residual.reshape(-1, hc_count * hidden_size)
    n = normed_residual.reshape(-1, hc_count * hidden_size)
    if out is None:
        out = torch.empty_like(r)
    else:
        out = out.reshape(-1, hc_count * hidden_size)

    module = _jit_hc_combine_module(hc_count, hidden_size, residual.dtype)
    module.hc_combine(y, r, n, inject_weight, out)
    return out.reshape(residual.shape)


_SPLIT = 8
_MAX_ROWS = 32
_partials_cache = {}


def _get_partials(hc_count: int, device: torch.device, rows: int) -> torch.Tensor:
    key = (device, hc_count)
    buf = _partials_cache.get(key)
    if buf is None or buf.shape[0] < rows:
        # The gate kernel writes one slot per (row, split, hc); a buffer shorter
        # than `rows` is written past its end rather than truncated.
        buf = torch.empty(
            (max(rows, _MAX_ROWS), _SPLIT, hc_count),
            dtype=torch.float32,
            device=device,
        )
        _partials_cache[key] = buf
    return buf


def hc_combine_split(
    block_output: torch.Tensor,
    residual: torch.Tensor,
    normed_residual: torch.Tensor,
    inject_weight: torch.Tensor,
    hc_count: int,
    hidden_size: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    y = block_output.reshape(-1, hidden_size)
    r = residual.reshape(-1, hc_count * hidden_size)
    n = normed_residual.reshape(-1, hc_count * hidden_size)
    if out is None:
        out = torch.empty_like(r)
    else:
        out = out.reshape(-1, hc_count * hidden_size)
    rows = r.shape[0]
    partials = _get_partials(hc_count, r.device, rows)[:rows]
    module = _jit_hc_combine_module(hc_count, hidden_size, residual.dtype)
    module.hc_combine_split(y, r, n, inject_weight, out, partials)
    return out.reshape(residual.shape)
