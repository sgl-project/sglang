"""Attention-TP2 all-gather / reduce-scatter through symmetric memory, for DP
attention where each DP group is a TP2 pair and the rows are small (decode)."""

from __future__ import annotations

import logging

import torch

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_parallel, get_resources

logger = logging.getLogger(__name__)

# Arbitrary; covers decode (graph max bs x verify tokens) with room, larger rows take NCCL.
_MAX_GATHERED_ROWS = 4096


class _AttnTpPushComm:
    """Each rank stores the peer's rows into the peer's buffer, then one GPU
    barrier. Two generations per op kind: a slot is rewritten two ops later,
    after a barrier that follows its last read."""

    def __init__(self, *, group, max_rows: int, hidden: int, device):
        import torch.distributed._symmetric_memory as symm_mem

        self.rank = group.rank_in_group
        self.peer = 1 - self.rank
        self.max_rows = max_rows
        self.hidden = hidden
        shape = (max_rows // 2, hidden)
        self.slots = {"gather": [], "scatter": []}
        self.gen = {"gather": 0, "scatter": 0}
        for kind in self.slots:
            for _ in range(2):
                local = symm_mem.empty(shape, dtype=torch.bfloat16, device=device)
                handle = symm_mem.rendezvous(local, group.device_group.group_name)
                peer = handle.get_buffer(self.peer, shape, torch.bfloat16)
                self.slots[kind].append((local, peer, handle))

    def takes(self, full: torch.Tensor, shard: torch.Tensor) -> bool:
        return (
            full.dim() == 2
            and full.dtype == torch.bfloat16
            and shard.dtype == torch.bfloat16
            and full.shape[1] == self.hidden
            and 0 < full.shape[0] <= self.max_rows
            and full.shape[0] == 2 * shard.shape[0]
        )

    def _exchange(self, kind: str, rows_for_peer: torch.Tensor) -> torch.Tensor:
        """Send ``rows_for_peer``; returns the rows the peer sent."""
        local, peer, handle = self.slots[kind][self.gen[kind]]
        self.gen[kind] ^= 1
        rows = rows_for_peer.shape[0]
        # An elementwise kernel, not copy_: copy_ into peer-mapped memory takes the
        # driver's peer memcpy, measured ~15 us slower per call on B300.
        torch.bitwise_or(
            rows_for_peer.view(torch.int16), 0, out=peer[:rows].view(torch.int16)
        )
        handle.barrier(channel=0)
        return local[:rows]

    def all_gather_into_tensor(self, output: torch.Tensor, shard: torch.Tensor):
        rows = shard.shape[0]
        received = self._exchange("gather", shard)
        output[self.rank * rows : (self.rank + 1) * rows].copy_(shard)
        output[self.peer * rows : (self.peer + 1) * rows].copy_(received)

    def reduce_scatter_tensor(self, output: torch.Tensor, partial: torch.Tensor):
        rows = output.shape[0]
        received = self._exchange(
            "scatter", partial[self.peer * rows : (self.peer + 1) * rows]
        )
        torch.add(
            partial[self.rank * rows : (self.rank + 1) * rows], received, out=output
        )


def initialize_attn_tp_push_comm(*, hidden_size: int, uses_megamoe: bool) -> None:
    """Runs once per worker after distributed setup, on every rank of the group."""
    parallel = get_parallel()
    group = parallel.attn_tp_group
    # MegaMoE syncs all EP ranks every layer, which orders a slot's reuse across forwards.
    if (
        not envs.SGLANG_OPT_ATTN_TP2_PUSH_COMM.get()
        or not uses_megamoe
        or not torch.cuda.is_available()
        or group.world_size != 2
        or parallel.attn_tp_size == parallel.tp_size
    ):
        return
    get_resources().attn_tp_push_comm = _AttnTpPushComm(
        group=group,
        max_rows=_MAX_GATHERED_ROWS,
        hidden=hidden_size,
        device=torch.device("cuda", torch.cuda.current_device()),
    )
    logger.info("Attention TP2 collectives use the symmetric-memory push comm")


def push_all_gather_into_tensor(output: torch.Tensor, input: torch.Tensor) -> bool:
    """Returns False when the rows are not ones the comm takes (caller uses NCCL)."""
    comm = get_resources().attn_tp_push_comm
    if comm is None or not comm.takes(output, input):
        return False
    comm.all_gather_into_tensor(output, input)
    return True


def push_reduce_scatter_tensor(output: torch.Tensor, input: torch.Tensor) -> bool:
    comm = get_resources().attn_tp_push_comm
    if comm is None or not comm.takes(input, output):
        return False
    comm.reduce_scatter_tensor(output, input)
    return True
