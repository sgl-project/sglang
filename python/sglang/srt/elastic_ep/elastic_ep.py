from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Iterator, List, Optional

import torch

from sglang.srt.distributed import parallel_state
from sglang.srt.managers.schedule_batch import ServerArgs
from sglang.srt.utils import is_cpu, is_cuda

logger = logging.getLogger(__name__)


@dataclass
class ElasticEPState:
    """Elastic EP active-rank state.

    Naming convention in this class:
    - global: tensor indices are world ranks.
    - pg: the mask originated from a process group, already mapped to global.
    - _cpu: tensor storage is on CPU; it does not mean CPU process-group source.
    """

    active_ranks: torch.Tensor
    committed_active_ranks_cpu: torch.Tensor
    last_handled_committed_active_ranks_cpu: torch.Tensor
    staging_active_ranks_cpu_slots: tuple[torch.Tensor, torch.Tensor]
    staging_global_pg_active_ranks_cpu_slots: tuple[torch.Tensor, torch.Tensor]
    next_staging_slot: int = 0
    pending_staging_slots: list[int] = field(default_factory=list)

    def submit_active_snapshot(
        self,
        global_pg_active_ranks: torch.Tensor,
        non_blocking: bool = True,
    ) -> None:
        """Enqueue copies of world-size active-rank snapshots.

        With non_blocking=True (overlap path) the scheduler relies on
        copy_done.synchronize() (same forward_stream, recorded strictly later)
        to act as the barrier before reading the staging buffers. Non-overlap
        callers pass non_blocking=False so the staging buffers are valid as
        soon as this call returns.
        """
        slot = self.next_staging_slot
        self.next_staging_slot ^= 1
        staging_active_ranks_cpu = self.staging_active_ranks_cpu_slots[slot]
        staging_global_pg_active_ranks_cpu = (
            self.staging_global_pg_active_ranks_cpu_slots[slot]
        )
        self.pending_staging_slots.append(slot)
        assert len(self.pending_staging_slots) <= 2

        if self.active_ranks.device.type == "cuda":
            staging_active_ranks_cpu.copy_(self.active_ranks, non_blocking=non_blocking)
        else:
            # CPU backend: fully synchronous, no stream involved.
            staging_active_ranks_cpu.copy_(self.active_ranks)

        assert global_pg_active_ranks.numel() == self.committed_active_ranks_cpu.numel()
        if global_pg_active_ranks.device.type == "cuda":
            staging_global_pg_active_ranks_cpu.copy_(
                global_pg_active_ranks, non_blocking=non_blocking
            )
        else:
            staging_global_pg_active_ranks_cpu.copy_(global_pg_active_ranks)

    def commit_active_snapshot(
        self, global_pg_active_ranks_cpu: torch.Tensor, consensus_cpu_group
    ) -> bool:
        """Commit the oldest unconsumed world-size snapshot.

        The committed mirror is a sticky fault latch: ranks return to active
        only through explicit recover/reset. Caller must ensure the staging
        buffers are ready, either by passing non_blocking=False to
        submit_active_snapshot (non-overlap path) or by synchronizing the
        stream after submit (e.g. via copy_done.synchronize, overlap path).
        global_pg_active_ranks_cpu is a read-only process-group observation.
        Returns True if a snapshot was committed.
        """
        if not self.pending_staging_slots:
            return False
        slot = self.pending_staging_slots.pop(0)
        snapshot = self.staging_active_ranks_cpu_slots[slot]
        global_pg_snapshot = self.staging_global_pg_active_ranks_cpu_slots[slot]

        assert (
            global_pg_active_ranks_cpu.numel()
            == self.committed_active_ranks_cpu.numel()
        )
        self.committed_active_ranks_cpu.bitwise_and_(snapshot)
        self.committed_active_ranks_cpu.bitwise_and_(global_pg_snapshot)
        self.committed_active_ranks_cpu.bitwise_and_(global_pg_active_ranks_cpu)
        torch.distributed.all_reduce(
            self.committed_active_ranks_cpu,
            op=torch.distributed.ReduceOp.MIN,
            group=consensus_cpu_group,
        )
        return True

    def is_stale_snapshot(self) -> bool:
        """Return True iff a new fault has appeared since the last handled
        snapshot. Call commit_active_snapshot() before this check."""
        return not torch.equal(
            self.committed_active_ranks_cpu,
            self.last_handled_committed_active_ranks_cpu,
        )

    def mark_snapshot_handled(self) -> None:
        """Snap last_handled to the current staging value so subsequent
        is_stale_snapshot() calls only flag *new* faults."""
        self.last_handled_committed_active_ranks_cpu.copy_(
            self.committed_active_ranks_cpu
        )

    def clear_pending_snapshots(self) -> None:
        self.pending_staging_slots.clear()

    def sync_active_to_cpu(self):
        self.committed_active_ranks_cpu.copy_(self.active_ranks.detach().cpu())

    def snapshot_active_to_last(self):
        self.last_handled_committed_active_ranks_cpu.copy_(
            self.active_ranks.detach().cpu()
        )

    def reset(self):
        self.active_ranks.fill_(1)
        self.snapshot_active_to_last()
        self.sync_active_to_cpu()
        self.clear_pending_snapshots()
        for staging_active_ranks_cpu in self.staging_active_ranks_cpu_slots:
            staging_active_ranks_cpu.fill_(1)
        for (
            staging_global_pg_active_ranks_cpu
        ) in self.staging_global_pg_active_ranks_cpu_slots:
            staging_global_pg_active_ranks_cpu.fill_(1)


class ElasticEPStateManager:
    _instance: Optional[ElasticEPState] = None

    @classmethod
    def instance(cls) -> ElasticEPState:
        return cls._instance

    @classmethod
    def init(cls, server_args: ServerArgs):
        if cls._instance is not None:
            return cls._instance

        if server_args.elastic_ep_backend is not None:
            cls._instance = cls._build_state(ep_size=None, device=None)
            if server_args.elastic_ep_rejoin:
                # Mask out peer ranks to perform cuda graph capture on its own
                cls._instance.active_ranks.zero_()
                cls._instance.active_ranks[torch.distributed.get_rank()] = 1
                cls._instance.snapshot_active_to_last()
                cls._instance.sync_active_to_cpu()

        return cls._instance

    @staticmethod
    def _select_device() -> torch.device:
        if is_cuda():
            return torch.device("cuda")
        elif is_cpu():
            return torch.device("cpu")
        else:
            raise NotImplementedError("Only CUDA and CPU support elastic ep now.")

    @classmethod
    def _build_state(
        cls,
        *,
        ep_size: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> ElasticEPState:
        active = cls.healthy_rank_state(ep_size=ep_size, device=device)
        active_cpu = active.detach().cpu().clone()
        # Initialize staging to the healthy initial state so is_stale_snapshot()
        # before the first submit returns False (no false positive on startup).
        if active.device.type == "cuda":
            staging_active_ranks_cpu_slots = (
                active_cpu.clone().pin_memory(),
                active_cpu.clone().pin_memory(),
            )
        else:
            staging_active_ranks_cpu_slots = (active_cpu.clone(), active_cpu.clone())

        if active.device.type == "cuda":
            staging_global_pg_active_ranks_cpu_slots = (
                active_cpu.clone().pin_memory(),
                active_cpu.clone().pin_memory(),
            )
        else:
            staging_global_pg_active_ranks_cpu_slots = (
                active_cpu.clone(),
                active_cpu.clone(),
            )

        return ElasticEPState(
            active_ranks=active,
            committed_active_ranks_cpu=active_cpu.clone(),
            last_handled_committed_active_ranks_cpu=active_cpu,
            staging_active_ranks_cpu_slots=staging_active_ranks_cpu_slots,
            staging_global_pg_active_ranks_cpu_slots=(
                staging_global_pg_active_ranks_cpu_slots
            ),
            pending_staging_slots=[],
        )

    @classmethod
    def healthy_rank_state(
        cls, *, ep_size: Optional[int] = None, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        size = ep_size if ep_size is not None else torch.distributed.get_world_size()
        dev = device if device is not None else cls._select_device()

        return torch.ones(size, dtype=torch.int32, device=dev)


# ---------------------------------------------------------------------------
# Helpers for elastic EP recovery
# ---------------------------------------------------------------------------


_PEER_STATE_POLL_INTERVAL_SEC = 0.01


def _iter_live_parallel_groups() -> Iterator[parallel_state.GroupCoordinator]:
    groups = []
    for group_ref in parallel_state._groups.values():
        group = group_ref()
        if group is not None:
            groups.append(group)
    for group in sorted(groups, key=lambda x: x.unique_name):
        yield group


def _map_global_to_group_local_ranks(
    group_ranks: List[int], global_ranks: List[int]
) -> List[int]:
    rank_to_local = {rank: idx for idx, rank in enumerate(group_ranks)}
    return [rank_to_local[rank] for rank in global_ranks if rank in rank_to_local]


def _wait_for_peer_state(mooncake_ep, process_group, ranks: List[int]) -> None:
    # Relaunched ranks become recoverable asynchronously, so we poll until the
    # target process group reports all requested peers as ready.
    while not all(mooncake_ep.get_peer_state(process_group, ranks)):
        time.sleep(_PEER_STATE_POLL_INTERVAL_SEC)


def _maybe_create_message_queue(group) -> None:
    if not group.use_message_queue_broadcaster or group.world_size <= 1:
        return

    from sglang.srt.distributed.device_communicators.shm_broadcast import MessageQueue

    group.mq_broadcaster = MessageQueue.create_from_process_group(
        group.cpu_group, 1 << 22, 6
    )


def _refresh_ep_members() -> None:
    from sglang.srt.layers.moe.token_dispatcher.mooncake import EPBuffer

    EPBuffer._buffer.update_ep_member()


def can_recover_ranks(global_ranks: List[int]) -> bool:
    from mooncake import ep as mooncake_ep

    world_group = torch.distributed.group.WORLD
    return all(mooncake_ep.get_peer_state(world_group, global_ranks))


def recover_ranks(global_ranks: List[int]) -> None:
    from mooncake import ep as mooncake_ep

    # Recover the world process group first, then recover each derived process group
    # using ranks mapped into that group's local rank space.
    world_group = torch.distributed.group.WORLD
    mooncake_ep.recover_ranks(world_group, global_ranks)

    for group in _iter_live_parallel_groups():
        group_local_ranks = _map_global_to_group_local_ranks(group.ranks, global_ranks)
        if not group_local_ranks:
            continue

        device_group = group.device_group
        _wait_for_peer_state(mooncake_ep, device_group, group_local_ranks)
        mooncake_ep.recover_ranks(device_group, group_local_ranks)

        cpu_group = group.cpu_group
        _wait_for_peer_state(mooncake_ep, cpu_group, group_local_ranks)
        mooncake_ep.recover_ranks(cpu_group, group_local_ranks)
        _maybe_create_message_queue(group)

    _refresh_ep_members()


def join_process_groups():
    from mooncake import ep as mooncake_ep

    def join_process_group(label: str, process_group) -> None:
        logger.info("Recovered rank joining Mooncake process group %s", label)
        mooncake_ep.join_group(process_group)

    join_process_group(
        "default_world",
        torch.distributed.group.WORLD,
    )

    for group in _iter_live_parallel_groups():
        if group.world_size <= 1:
            continue

        join_process_group(
            f"{group.unique_name}:device",
            group.device_group,
        )
        join_process_group(
            f"{group.unique_name}:cpu",
            group.cpu_group,
        )
        _maybe_create_message_queue(group)

    _refresh_ep_members()
