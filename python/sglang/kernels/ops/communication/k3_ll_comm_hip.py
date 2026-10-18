"""The communicator of Kimi-K3's TP8 LL collectives on gfx950 (csrc/kimi_k3/comm/ll_hip.cuh):
the attention-residual read's stream gather and the attention output's all-reduce
read (attention/k3_attn_res_read_hip).

Their scratch is the meta buffer tail of a dedicated aiter CustomAllreduce,
which is never used for anything else: one region and one row of flags per
collective.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Optional

import torch

if TYPE_CHECKING:
    from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce

WORLD_SIZE: int = 8
MAX_ROWS: int = 64
# Per region: 2 epochs x world x rows x lines-per-row x 16 B (8 payload bytes a
# line). The kernels hardcode the region order and line counts.
_LINES_PER_ROW = (2 * 7168 // WORLD_SIZE // 8,)  # gather read: an H / 8 bf16 slice
# Then the all-reduce read's region: whole rows, two 16-byte lines per 8 bf16.
AR_MAX_ROWS: int = 16
_AR_LINES_PER_ROW = 2 * 7168 // 8
_SCRATCH_BYTES: int = (
    sum(2 * WORLD_SIZE * MAX_ROWS * n * 16 for n in _LINES_PER_ROW)
    + 2 * WORLD_SIZE * AR_MAX_ROWS * _AR_LINES_PER_ROW * 16
)


class LLComm(NamedTuple):
    comm: CustomAllreduce
    # rank-local flags: [MAX_ROWS] attention-residual gather rows
    epochs: torch.Tensor
    # rank-local, [AR_MAX_ROWS] each: the all-reduce read's row flags, and per row the
    # count of this rank's pushers that took its flag (kept at 0 between launches)
    ar_epochs: torch.Tensor
    ar_pushed: torch.Tensor


def create_ll_comm(cpu_group, device: torch.device) -> Optional[LLComm]:
    """Collective over ``cpu_group`` (a non-NCCL group): every rank calls it,
    outside graph capture."""
    from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce

    # aiter sizes the meta buffer as Signal + 2 * max_size.
    comm = CustomAllreduce(cpu_group, device, max_size=(_SCRATCH_BYTES + 1) // 2)
    if comm.disabled or comm.world_size != WORLD_SIZE:
        return None
    epochs = torch.zeros(MAX_ROWS, dtype=torch.int32, device=device)
    ar_epochs = torch.zeros(AR_MAX_ROWS, dtype=torch.int32, device=device)
    ar_pushed = torch.zeros(AR_MAX_ROWS, dtype=torch.int32, device=device)
    return LLComm(comm, epochs, ar_epochs, ar_pushed)
