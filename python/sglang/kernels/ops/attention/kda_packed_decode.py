"""CUDA KDA packed-decode kernel (batched decode fast path).

Row-streaming port of the triton fused_recurrent_kda_packed_decode_kernel:
the triton kernel keeps a [BV, K] fp32 state tile in one warp's registers and
tops out at ~5 TB/s; this kernel streams the state one 512B row at a time and
reaches the in-place read+write bandwidth of the part (~9.6 TB/s probe).
Outputs match the triton kernel to ULPs (warp-shuffle reduction order), not
bits. Unsupported inputs fall back to triton through a covered() check.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_WARPS: int = 8
# The row-streaming layout needs enough (batch x head) CTAs to fill the GPU;
# below this the triton kernel's launch cost is already the floor.
_MIN_BATCH: int = 8


@cache_once
def _jit_kda_packed_decode_module() -> Module:
    args = make_cpp_args(_WARPS, is_arch_support_pdl())
    return load_jit(
        "kda_packed_decode_" + str(_WARPS),
        *args,
        cuda_files=["attention/kda_packed_decode.cuh"],
        cuda_wrappers=[("run", f"KdaPackedDecodeKernel<{args}>::run")],
        extra_cuda_cflags=["-O3"],
    )


def covered(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    initial_state: torch.Tensor,
    out: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    num_q_heads: int,
) -> bool:
    B = mixed_qkv.shape[0]
    HV, V, K = initial_state.shape[-3:]
    return (
        B >= _MIN_BATCH
        and K == 128
        and V == 128
        and HV % max(num_q_heads, 1) == 0
        # Per-K gate layout. A per-head scalar gate ([B, HV] / [HV], the GDN
        # shape) is a different kernel, not a slower input for this one.
        and a.dim() == 2
        and a.shape[1] == HV * K
        and dt_bias.numel() == HV * K
        and mixed_qkv.dtype == torch.bfloat16
        and a.dtype == torch.bfloat16
        and b.dtype == torch.bfloat16
        and A_log.dtype == torch.float32
        and dt_bias.dtype == torch.float32
        and initial_state.dtype == torch.float32
        and out.dtype == torch.bfloat16
        and ssm_state_indices.dtype == torch.int32
        and mixed_qkv.stride(-1) == 1
        and a.stride(-1) == 1
        and b.stride(-1) == 1
        and initial_state.stride(-1) == 1
        and initial_state.stride(-2) == K
        and initial_state.stride(-3) == V * K
        and out.is_contiguous()
        and ssm_state_indices.is_contiguous()
    )


def kda_packed_decode(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    out: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    num_q_heads: int,
    lower_bound: Optional[float] = None,
) -> None:
    """In-place KDA decode step: updates `initial_state` rows selected by
    `ssm_state_indices` and writes attention output into `out` ([B, 1, HV, V]).
    Caller must have checked covered(); q/k l2-norm is always applied
    (matches the production dispatch)."""
    B = mixed_qkv.shape[0]
    HV, V, _ = initial_state.shape[-3:]
    state = initial_state.view(-1, *initial_state.shape[-3:])
    _jit_kda_packed_decode_module().run(
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        out.view(B, HV, V),
        state,
        ssm_state_indices,
        float(scale),
        float(lower_bound) if lower_bound is not None else 0.0,
        lower_bound is not None,
        int(num_q_heads),
    )
