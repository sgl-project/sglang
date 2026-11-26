"""Gloo control votes and HCCL migration communication for Ascend EPLB."""

from __future__ import annotations

import time
from collections import defaultdict

import numpy as np
import torch
import torch.distributed as dist

# Per-peer warmup payload in bytes, independent of HCCL buffer capacity.
_EPLB_WARMUP_BYTES_PER_PEER = 4 << 20


# Forward votes use a Gloo group separate from the background plan exchange.
class EplbControl:
    def __init__(self, group: dist.ProcessGroup):
        self.group = group
        self.wait_seconds = 0.0
        # The forward thread owns votes and waits for each reduction to finish.
        self._vote_values = np.empty(8, dtype=np.int64)
        self._vote_flags = torch.from_numpy(self._vote_values)

    def vote(
        self, epoch: int, phase: int, layer: int, ready: bool, error: bool
    ) -> tuple[bool, bool]:
        # MAX(x), MAX(-x) detects mismatched protocol positions on every rank.
        self._vote_values[:] = (
            epoch,
            -epoch,
            phase,
            -phase,
            layer,
            -layer,
            int(not ready),
            int(error),
        )
        start = time.perf_counter()
        dist.all_reduce(self._vote_flags, op=dist.ReduceOp.MAX, group=self.group)
        self.wait_seconds += time.perf_counter() - start
        (
            max_epoch,
            neg_min_epoch,
            max_phase,
            neg_min_phase,
            max_layer,
            neg_min_layer,
            any_unready,
            any_error,
        ) = self._vote_flags.tolist()
        if (
            max_epoch != -neg_min_epoch
            or max_phase != -neg_min_phase
            or max_layer != -neg_min_layer
        ):
            raise RuntimeError("EPLB ranks reached different protocol boundaries")
        return any_unready == 0, bool(any_error)

    def close(self) -> None:
        dist.destroy_process_group(self.group)


def _make_comm_rank_lookup(old_layout, new_layout, local_experts):
    def index_ranks(layout):
        by_expert = defaultdict(list)
        for slot, expert in enumerate(layout):
            rank = slot // local_experts
            ranks = by_expert[expert]
            # Preserve rank order for the shared source assignment.
            if not ranks or ranks[-1] != rank:
                ranks.append(rank)
        return by_expert

    old_ranks = index_ranks(old_layout)
    new_ranks = index_ranks(new_layout)

    def lookup(expert):
        sources = old_ranks[expert]
        source_ranks = set(sources)
        return sources, [rank for rank in new_ranks[expert] if rank not in source_ranks]

    return lookup


def _hccl_expert_order(old_layout, new_layout, num_logical, world_size, batch_size):
    """Spread busy rank pairs across HCCL batches without changing the transfers."""
    block_size = 4
    if batch_size % block_size or num_logical <= batch_size:
        return None

    num_batches = (num_logical + batch_size - 1) // batch_size
    padded = num_batches * batch_size
    ranks = np.arange(world_size)[:, None]
    owners = np.zeros((world_size, padded), dtype=bool)
    receivers = np.zeros_like(owners)
    owners[ranks, np.asarray(old_layout).reshape(world_size, -1)] = True
    receivers[ranks, np.asarray(new_layout).reshape(world_size, -1)] = True
    receivers &= ~owners

    # The updater splits each expert's destinations evenly among its old
    # owners, giving the remainder to the lowest owner ranks.
    quotient, remainder = np.divmod(
        receivers.sum(axis=0), owners.sum(axis=0).clip(min=1)
    )
    sends = owners * (quotient + (owners.cumsum(axis=0) <= remainder))
    endpoint_rows = np.concatenate((sends, receivers), axis=0)
    contiguous = endpoint_rows.reshape(-1, num_batches, batch_size).sum(axis=2)
    contiguous_cost = contiguous.max(axis=0).sum()
    if contiguous_cost == 0:
        return None
    interleaved = endpoint_rows.reshape(
        -1, batch_size // block_size, num_batches, block_size
    ).sum(axis=(1, 3))

    # Require a substantial drop in busiest-endpoint rows to offset extra
    # submissions on lightly loaded ranks.
    if interleaved.max(axis=0).sum() * 3 > contiguous_cost * 2:
        return None
    return (
        np.arange(padded)
        .reshape(-1, num_batches, block_size)
        .transpose(1, 0, 2)
        .flatten()
        .tolist()
    )


def _submit_hccl_batches(p2p_op_infos, expert_order, batch_size, work_handles):
    ops_by_expert = dict(p2p_op_infos)
    if not any(ops_by_expert.values()):
        return
    for start in range(0, len(expert_order), batch_size):
        batch_ops = [
            op
            for expert in expert_order[start : start + batch_size]
            for op in ops_by_expert.get(expert, ())
        ]
        if batch_ops:
            work = dist.batch_isend_irecv(batch_ops)
            work_handles.extend(work)
            for request in work:
                request.wait()


def _validate_migration_topology(group_size: int) -> None:
    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    world_size = dist.get_world_size()
    # TP and WORLD must have the same members and rank numbering.
    if parallel.pp_size != 1 or not (
        parallel.tp_size == parallel.moe_ep_size == group_size == world_size
    ):
        raise ValueError(
            "NPU EPLB requires PP=1 and TP size = EP size = EPLB group size = "
            f"WORLD size; got PP={parallel.pp_size}, TP={parallel.tp_size}, "
            f"EP={parallel.moe_ep_size}, EPLB group={group_size}, WORLD={world_size}."
        )


def prewarm_eplb_communication() -> None:
    """Warm the migration communicator before KV sizing."""
    from sglang.srt.runtime_context import get_exec, get_parallel

    # The async runtime warms its dedicated group after weight loading.
    if get_exec().moe.eplb_async_rebalance:
        return
    group = get_parallel().tp_group.device_group
    world_size = dist.get_world_size(group)
    _validate_migration_topology(world_size)
    if world_size == 1:
        return
    buffers = _allocate_migration_warmup_buffers(world_size)
    _warm_migration_group(group, *buffers)
    del buffers
    torch.npu.empty_cache()


def _allocate_migration_warmup_buffers(
    world_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    send = torch.empty(_EPLB_WARMUP_BYTES_PER_PEER, dtype=torch.uint8, device="npu")
    recv = torch.empty(
        (world_size - 1, send.numel()), dtype=send.dtype, device=send.device
    )
    return send, recv


def _warm_migration_group(
    group: dist.ProcessGroup, send: torch.Tensor, recv: torch.Tensor
) -> None:
    ops = []
    rank = dist.get_rank()
    world_size = dist.get_world_size(group)
    for slot, peer in enumerate(p for p in range(world_size) if p != rank):
        ops.extend(
            [
                dist.P2POp(dist.isend, send, peer, group=group),
                dist.P2POp(dist.irecv, recv[slot], peer, group=group),
            ]
        )
    for work in dist.batch_isend_irecv(ops):
        work.wait()
    dist.all_reduce(send.view(torch.int32), group=group)
    torch.npu.synchronize()
