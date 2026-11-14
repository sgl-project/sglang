"""gfx950 BF16/FP16 GEMM from fbtriton ``tlx.ops.mm``, vendored in ``sglang.kernels.tlx_ops``.

``linear`` runs ``x @ weight.T`` for an ``[N, K]`` weight and ``x`` of any leading
dims and strides, any M/N/K. Rejected (see ``mm_supported``): other or mixed dtypes,
empty inputs, and BF16 single rows. Bias is not part of the contract, and the caller
gates the architecture. Runtime requirements are in
``sglang/kernels/jit/utils/tlx_runtime.py``.
"""

import torch


def mm_supported(x: torch.Tensor, weight: torch.Tensor) -> bool:
    """Whether ``linear(x, weight)`` handles these operands; host-only, never reads device data."""
    if x.dtype not in (torch.bfloat16, torch.float16) or weight.dtype != x.dtype:
        return False
    # mm sends bf16 single rows to a packed GEMV built on tlx.amd_packed_dot2, which uTLX lacks.
    return x.numel() > 0 and not (
        x.dtype == torch.bfloat16 and x.numel() == x.shape[-1]
    )


def linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``x @ weight.T`` for a ``[N, K]`` weight, over any leading dims of ``x``."""
    from sglang.kernels.jit.utils.tlx_runtime import add_missing_triton_knobs
    from sglang.kernels.tlx_ops.kernels.mm.gfx950 import mm

    add_missing_triton_knobs()
    out = mm(x.reshape(-1, x.shape[-1]), weight.t())
    return out.view(*x.shape[:-1], weight.shape[0])
