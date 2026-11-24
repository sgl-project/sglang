"""ROCm gfx950 runtime adapter for fused XGMI push DCP all-to-all + LSE combine."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from sglang.kernels.ops.communication.dcp_a2a_lse_hip import (
    DEFAULT_MAX_ROWS,
    SUPPORTED_HEAD_DIMS,
    SUPPORTED_WORLD_SIZES,
    DcpLlComm,
    can_run_dcp_a2a_lse_hip,
    create_dcp_ll_comm,
    dcp_a2a_lse_hip,
)
from sglang.srt.utils import is_gfx95_supported

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state import GroupCoordinator

_COMM_CACHE: dict[tuple[str, int, int, int], DcpLlComm | None] = {}


def init_dcp_ll_comm_hip(
    cp_group: GroupCoordinator,
    *,
    head_dim: int,
    num_local_heads: int,
) -> DcpLlComm | None:
    """Pre-allocate the DCP push-A2A communicator outside CUDA graph capture."""
    from sglang.srt.runtime_context import get_exec

    world_size = cp_group.world_size
    if (
        world_size not in SUPPORTED_WORLD_SIZES
        or head_dim not in SUPPORTED_HEAD_DIMS
        or num_local_heads <= 0
    ):
        return None
    key = (cp_group.unique_name, world_size, head_dim, num_local_heads)
    if key in _COMM_CACHE:
        return _COMM_CACHE[key]
    if torch.cuda.is_current_stream_capturing():
        return None
    if not is_gfx95_supported() or get_exec().comm.disable_custom_all_reduce:
        _COMM_CACHE[key] = None
        return None
    comm = create_dcp_ll_comm(
        cp_group.cpu_group,
        torch.device("cuda", torch.cuda.current_device()),
        world_size=world_size,
        head_dim=head_dim,
        num_local_heads=num_local_heads,
        max_rows=DEFAULT_MAX_ROWS,
    )
    _COMM_CACHE[key] = comm
    return comm


def maybe_dcp_a2a_lse_reduce_hip(
    cp_attn_out: torch.Tensor,
    cp_attn_lse: torch.Tensor,
    cp_group: GroupCoordinator,
    *,
    is_lse_base_on_e: bool = True,
) -> torch.Tensor | None:
    """Run fused XGMI push A2A + LSE combine on gfx950, or return None to fall back."""
    world_size = cp_group.world_size
    if world_size not in SUPPORTED_WORLD_SIZES or cp_attn_out.dim() != 3:
        return None
    b, h, d = cp_attn_out.shape
    if (
        b <= 0
        or b > DEFAULT_MAX_ROWS
        or d not in SUPPORTED_HEAD_DIMS
        or h % world_size != 0
    ):
        return None
    comm = init_dcp_ll_comm_hip(
        cp_group,
        head_dim=d,
        num_local_heads=h // world_size,
    )
    if not can_run_dcp_a2a_lse_hip(comm, cp_attn_out, cp_attn_lse):
        return None
    return dcp_a2a_lse_hip(
        cp_attn_out,
        cp_attn_lse,
        comm=comm,
        is_lse_base_on_e=is_lse_base_on_e,
    )
