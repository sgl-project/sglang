"""Kimi-K3 TP8 collectives fused with an attention-residual read on gfx950 (see
csrc/kimi_k3/attn_res/read_hip.cuh), over communication/k3_ll_comm_hip's
scratch: the all-gather of the next stream, whose every rank finishes one
[T, H / 8] column slice in-kernel from the column-parallel latent-MoE tail's
split-K partials; and the attention output's all-reduce with the residual
add, ahead of the FFN-side read.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.kernels.ops.communication.k3_ll_comm_hip import (
    MAX_ROWS,
    WORLD_SIZE,
    LLComm,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

HIDDEN_SIZE: int = 7168
SLICE_SIZE: int = HIDDEN_SIZE // WORLD_SIZE
MAX_NVB: int = 8
# Must match kTailK / kTailSplits in the kernel.
TAIL_LATENT: int = 3584
TAIL_SPLITS: int = 7


@cache_once
def _module() -> Module:
    from aiter.jit.core import AITER_CSRC_DIR

    return load_jit(
        "kimi_k3_attn_res_read",
        cuda_files=["kimi_k3/attn_res/read_hip.cuh"],
        cuda_wrappers=[
            ("run", "k3_attn_res_read::K3AttnResReadKernel::run"),
            ("ar_run", "k3_attn_res_read::K3AttnResArReadKernel::run"),
        ],
        # AITER's CustomAllreduce, whose meta buffers hold the all-gather scratch
        extra_include_paths=[f"{AITER_CSRC_DIR}/include"],
        extra_cuda_cflags=["-O3"],
    )


def gather_shape_covered(
    rows: int, slice_width: int, bank: torch.Tensor, nvb: int, write: bool = False
) -> bool:
    """Whether the gather reads take [rows, slice_width] slices of bank's stream."""
    return (
        0 < rows <= MAX_ROWS
        and slice_width == SLICE_SIZE
        and 1 <= nvb <= MAX_NVB
        and bank.dtype == torch.bfloat16
        and bank.dim() == 3
        and bank.shape[0] == rows
        and bank.shape[2] == HIDDEN_SIZE
        and bank.stride(-1) == 1
        and bank.stride(0) % 8 == 0
        and bank.stride(1) % 8 == 0
        and (not write or nvb < bank.shape[1])
    )


def tail_covered(shared: torch.Tensor, prefix: Optional[torch.Tensor]) -> bool:
    """Whether gather_tail_attn_res_read takes the tail's shared output and prefix."""
    return (
        shared.dtype == torch.bfloat16
        and shared.shape[1] == HIDDEN_SIZE
        and shared.stride(1) == 1
        and shared.stride(0) % 8 == 0
        and shared.data_ptr() % 16 == 0
        and (
            prefix is None
            or (
                prefix.dtype == torch.bfloat16
                and prefix.shape == shared.shape
                and prefix.stride() == shared.stride()
                and prefix.data_ptr() % 16 == 0
            )
        )
    )


def gather_tail_attn_res_read(
    part: torch.Tensor,
    sq: torch.Tensor,
    shared: torch.Tensor,
    prefix: Optional[torch.Tensor],
    tail_eps: float,
    bank: torch.Tensor,
    nvb: int,
    cw: torch.Tensor,
    ow: torch.Tensor,
    score_eps: float,
    out_eps: float,
    comm: LLComm,
    *,
    write_bank: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(rmsnorm(mix(bank[:, :nvb], stream)) * ow, stream) with stream the
    all-gather of every rank's [T, H / 8] column slice (rank r holds columns
    [r * H / 8, (r + 1) * H / 8)), finished from the column tail's partials
    (k3_latent_tail_partials of its up_proj rows):
    bf16(bf16(sum_k part[k] / rms) + shared[:, cols] [+ prefix[:, cols]]).
    cw and ow are fp32 [H]; write_bank also stores the stream into bank[:, nvb]."""
    rows = bank.shape[0]
    stream = bank.new_empty((rows, HIDDEN_SIZE))
    out = bank.new_empty((rows, HIDDEN_SIZE))
    _module().run(
        comm.comm._ptr,
        prefix if prefix is not None else shared,
        bank,
        cw,
        ow,
        stream,
        out,
        comm.epochs,
        part,
        sq,
        shared,
        nvb,
        prefix is not None,
        write_bank,
        float(score_eps),
        float(out_eps),
        float(tail_eps),
    )
    return out, stream


# Measured in-server on MI350X: faster than aiter's all-reduce + the Triton read only at
# 4 rows (EAGLE C1 verify).
AR_READ_ROWS: tuple[int, ...] = (4,)


def all_reduce_covered(
    partial: torch.Tensor,
    residual: Optional[torch.Tensor],
    bank: torch.Tensor,
    nvb: int,
) -> bool:
    """Whether all_reduce_attn_res_read takes the attention output's partial."""

    def dense_rows(x):
        return (
            x.dtype == torch.bfloat16
            and x.shape == (bank.shape[0], HIDDEN_SIZE)
            and x.is_contiguous()
            and x.data_ptr() % 16 == 0
        )

    return (
        partial.shape[0] in AR_READ_ROWS
        and dense_rows(partial)
        and (residual is None or dense_rows(residual))
        and gather_shape_covered(partial.shape[0], SLICE_SIZE, bank, nvb)
    )


def all_reduce_attn_res_read(
    partial: torch.Tensor,
    residual: Optional[torch.Tensor],
    bank: torch.Tensor,
    nvb: int,
    cw: torch.Tensor,
    ow: torch.Tensor,
    score_eps: float,
    out_eps: float,
    comm: LLComm,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(rmsnorm(mix(bank[:, :nvb], stream)) * ow, stream) with stream =
    bf16(residual + all-reduce(partial)) over TP8 (residual None: the sum
    alone); partial and residual are [T, H] bf16, cw and ow fp32 [H]. The
    caller checked all_reduce_covered()."""
    rows = partial.shape[0]
    stream = partial.new_empty((rows, HIDDEN_SIZE))
    out = partial.new_empty((rows, HIDDEN_SIZE))
    _module().ar_run(
        comm.comm._ptr,
        partial,
        residual if residual is not None else partial,
        bank,
        cw,
        ow,
        stream,
        out,
        comm.ar_epochs,
        comm.ar_pushed,
        nvb,
        residual is not None,
        float(score_eps),
        float(out_eps),
    )
    return out, stream
