"""gfx950 DCP8 all-to-all of attention partials and LSE in one IPC kernel."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce
    from tvm_ffi.module import Module

WORLD_SIZE: int = 8
# Kimi-K3.1's 96 heads under DCP8
HEADS_PER_RANK: int = 12
HEAD_DIM: int = 128
# must match kMaxRows
MAX_ROWS: int = 1024
_SCRATCH_BYTES: int = MAX_ROWS * WORLD_SIZE * HEADS_PER_RANK * (HEAD_DIM * 2 + 4)


@cache_once
def _module() -> Module:
    from aiter.jit.core import AITER_CSRC_DIR

    args = make_cpp_args(HEADS_PER_RANK)
    return load_jit(
        "dcp_ipc_a2a_hip",
        *args,
        cuda_files=["distributed/dcp_ipc_a2a_hip.cuh"],
        cuda_wrappers=[("exchange", f"dcp_ipc_a2a::DcpIpcA2aHip<{args}>::exchange")],
        extra_include_paths=[f"{AITER_CSRC_DIR}/include"],
        extra_cuda_cflags=["-O3"],
    )


def create_ipc_comm(cpu_group, device: torch.device) -> Optional[CustomAllreduce]:
    """A collective over the DCP ranks' CPU group; call outside graph capture."""
    from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce

    # Dedicated: its meta buffer (Signal + 2 * max_size) holds the scratch.
    comm = CustomAllreduce(cpu_group, device, max_size=_SCRATCH_BYTES // 2)
    if comm.disabled:
        return None
    _module()  # compiled here, not at first use inside graph capture
    return comm


def ipc_exchange_covers(out: torch.Tensor, lse: torch.Tensor) -> bool:
    return (
        out.dtype == torch.bfloat16
        and 0 < out.shape[0] <= MAX_ROWS
        and out.shape[1:] == (WORLD_SIZE * HEADS_PER_RANK, HEAD_DIM)
        and out.is_contiguous()
        and out.data_ptr() % 16 == 0
        and lse.is_contiguous()
    )


def ipc_exchange(
    comm: CustomAllreduce, out: torch.Tensor, lse: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """(recv_out [N, B, H/N, 128], recv_lse [N, B, H/N]); needs ipc_exchange_covers()."""
    rows = out.shape[0]
    recv_out = out.new_empty((WORLD_SIZE, rows, HEADS_PER_RANK, HEAD_DIM))
    recv_lse = lse.new_empty((WORLD_SIZE, rows, HEADS_PER_RANK))
    _module().exchange(comm._ptr, out, lse, recv_out, recv_lse)
    return recv_out, recv_lse
