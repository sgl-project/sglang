"""Platform dispatch for the LPLB kernel backends.

CUDA uses the fused JIT kernels in :mod:`sglang.kernels.ops.lplb.cuda_solver`
(Hopper+); Ascend NPU uses the registered Triton-Ascend operators in
``sgl_kernel_npu.operators.moe.lplb``, adapted by
:mod:`sglang.srt.hardware_backend.npu.lplb.npu_solver`.  Both backends expose
the same ``prep_lp_inputs`` / ``solve_ipm`` / ``extract_log2phy_prob`` /
``dispatch_probability`` surface, so callers select one here and stay free of
platform branches.
"""

from __future__ import annotations

from typing import Callable


def get_lplb_backend():
    """Return the LPLB kernel module for the current device."""
    from sglang.srt.utils import is_npu

    if is_npu():
        from sglang.srt.hardware_backend.npu.lplb import npu_solver

        return npu_solver

    from sglang.kernels.ops.lplb import cuda_solver

    return cuda_solver


def get_lplb_warmup() -> Callable:
    """Return the platform's ``warmup(nc, nv, num_iters, device)`` callable."""
    from sglang.srt.utils import is_npu

    if is_npu():
        from sglang.srt.hardware_backend.npu.lplb import npu_solver

        return npu_solver.warmup

    from sglang.kernels.ops.lplb.torch_solver import warmup

    return warmup
