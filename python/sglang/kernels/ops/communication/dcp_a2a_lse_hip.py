"""Fused ROCm XGMI push DCP all-to-all + LSE combine (csrc/distributed/dcp_a2a_lse_hip.cuh)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import msgspec
import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

SUPPORTED_WORLD_SIZES: tuple[int, ...] = (2, 4, 8)
SUPPORTED_HEAD_DIMS: tuple[int, ...] = (128, 512)
DEFAULT_MAX_ROWS: int = 192

_DTYPE_CPP = {
    torch.bfloat16: ("bf16", "__hip_bfloat16"),
    torch.float16: ("fp16", "__half"),
}


class DcpLlComm(msgspec.Struct, frozen=True):
    """Dedicated AITER CustomAllreduce scratch + rank-local epoch table for DCP push A2A."""

    comm: Any
    epochs: torch.Tensor
    world_size: int
    head_dim: int
    max_slots: int
    max_grid: int


@cache_once
def _module(world_size: int, head_dim: int, dtype_tag: str, cpp_type: str) -> Module:
    from aiter.jit.core import AITER_CSRC_DIR

    return load_jit(
        f"dcp_a2a_lse_hip_w{world_size}_d{head_dim}_{dtype_tag}",
        cuda_files=["distributed/dcp_a2a_lse_hip.cuh"],
        cuda_wrappers=[
            (
                "run",
                f"dcp_a2a_lse_hip::DcpA2ALseKernel<{world_size}, {head_dim}, {cpp_type}>::run",
            ),
        ],
        extra_include_paths=[f"{AITER_CSRC_DIR}/include"],
        extra_cuda_cflags=["-O3"],
    )


def create_dcp_ll_comm(
    cpu_group,
    device: torch.device,
    *,
    world_size: int,
    head_dim: int,
    num_local_heads: int,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> DcpLlComm | None:
    """Collective over ``cpu_group``; every DCP rank calls outside CUDA graph capture."""
    if (
        world_size not in SUPPORTED_WORLD_SIZES
        or head_dim not in SUPPORTED_HEAD_DIMS
        or num_local_heads <= 0
        or max_rows <= 0
    ):
        return None
    from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce

    max_slots = max_rows * num_local_heads
    vecs_per_slot = (head_dim // 8) + 1
    scratch_bytes = 2 * world_size * max_slots * vecs_per_slot * 16
    # aiter sizes the uncached meta buffer as Signal + 2 * max_size.
    comm = CustomAllreduce(cpu_group, device, max_size=(scratch_bytes + 1) // 2)
    if comm.disabled or comm.world_size != world_size:
        return None
    epochs = torch.zeros(max_slots, dtype=torch.int32, device=device)
    cu_count = torch.cuda.get_device_properties(device).multi_processor_count
    max_grid = max(1, cu_count * 4)
    return DcpLlComm(
        comm=comm,
        epochs=epochs,
        world_size=world_size,
        head_dim=head_dim,
        max_slots=max_slots,
        max_grid=max_grid,
    )


def can_run_dcp_a2a_lse_hip(
    comm: DcpLlComm | None,
    cp_attn_out: torch.Tensor,
    cp_attn_lse: torch.Tensor,
) -> bool:
    """Whether ``dcp_a2a_lse_hip`` covers these tensors and communicator."""
    if comm is None or cp_attn_out.dim() != 3 or cp_attn_lse.dim() != 2:
        return False
    b, h, d = cp_attn_out.shape
    if (
        b <= 0
        or d != comm.head_dim
        or h % comm.world_size != 0
        or cp_attn_lse.shape != (b, h)
        or cp_attn_lse.dtype != torch.float32
        or cp_attn_out.dtype not in _DTYPE_CPP
    ):
        return False
    return b * (h // comm.world_size) <= comm.max_slots


def dcp_a2a_lse_hip(
    cp_attn_out: torch.Tensor,
    cp_attn_lse: torch.Tensor,
    *,
    comm: DcpLlComm,
    is_lse_base_on_e: bool = True,
) -> torch.Tensor:
    """Fused XGMI push all-to-all + fp32 LSE combine returning ``[B, H // world, D]``."""
    b, h, d = cp_attn_out.shape
    h_local = h // comm.world_size
    dtype_tag, cpp_type = _DTYPE_CPP[cp_attn_out.dtype]
    out_in = cp_attn_out.contiguous()
    if out_in.data_ptr() % 16 != 0:
        out_in = out_in.clone()
    out = cp_attn_out.new_empty((b, h_local, d))
    _module(comm.world_size, d, dtype_tag, cpp_type).run(
        out_in,
        cp_attn_lse.contiguous(),
        out,
        comm.comm._ptr,
        comm.epochs,
        comm.max_slots,
        comm.max_grid,
        is_lse_base_on_e,
    )
    return out
