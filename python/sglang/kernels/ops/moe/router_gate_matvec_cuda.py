from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# Decode-sized router-gate logits (see csrc/moe/router_gate_matvec.cuh): one
# block per kRows expert rows with every weight and activation load in flight
# before the first FMA, fp32 accumulation in a fixed order. kRows = 4
# (128 blocks) measured best on H20.

_MAX_M = 8
_THREADS = 256
_ROWS = 4


def _max_m_bucket(m: int) -> int:
    return 4 if m <= 4 else 8


@cache_once
def _jit_router_gate_matvec_module(
    w_dtype: torch.dtype, max_m: int, rows: int
) -> Module:
    args = make_cpp_args(w_dtype, max_m, rows)
    return load_jit(
        "router_gate_matvec",
        *args,
        cuda_files=["moe/router_gate_matvec.cuh"],
        cuda_wrappers=[("run", f"sglang::RouterGateMatvecKernel<{args}>::run")],
        extra_cuda_cflags=["-O3"],
    )


def use_router_gate_matvec_cuda(
    x: torch.Tensor, weight: torch.Tensor, rows: int = _ROWS
) -> bool:
    if not (
        x.is_cuda and x.dtype == torch.bfloat16 and x.dim() == 2 and x.is_contiguous()
    ):
        return False
    if weight.dtype not in (torch.float32, torch.bfloat16):
        return False
    if not weight.is_contiguous():
        return False
    m, k = x.shape
    e = weight.shape[0]
    chunk = _THREADS * (8 if weight.dtype == torch.bfloat16 else 4)
    return (
        1 <= m <= _MAX_M
        and k % chunk == 0
        and e % rows == 0
        and torch.cuda.get_device_capability(x.device)[0] >= 9
    )


def router_gate_matvec_cuda(
    x: torch.Tensor, weight: torch.Tensor, rows: int = _ROWS
) -> torch.Tensor:
    """out[M, E] fp32 = x[M, K] (bf16) @ weight[E, K]^T (fp32 or bf16)."""
    m, k = x.shape
    e = weight.shape[0]
    out = torch.empty((m, e), dtype=torch.float32, device=x.device)
    module = _jit_router_gate_matvec_module(weight.dtype, _max_m_bucket(m), rows)
    module.run(x, weight, out)
    return out
