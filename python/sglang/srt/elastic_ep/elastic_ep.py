from __future__ import annotations

from dataclasses import dataclass, field
import logging
import time
from typing import Iterator, List, Optional

import torch

from sglang.srt.distributed import parallel_state
from sglang.srt.managers.schedule_batch import ServerArgs
from sglang.srt.utils import is_cpu, is_cuda

logger = logging.getLogger(__name__)


@dataclass
class ElasticEPState:
    active_ranks: Optional[torch.Tensor]
    active_ranks_cpu: Optional[torch.Tensor]
    last_handled_active_ranks_cpu: Optional[torch.Tensor]
    staging_active_ranks_cpu_slots: Optional[tuple[torch.Tensor, torch.Tensor]]
    tp_active_ranks_cpu: Optional[torch.Tensor] = None
    staging_tp_active_ranks_cpu_slots: Optional[
        tuple[torch.Tensor, torch.Tensor]
    ] = None
    next_staging_slot: int = 0
    pending_staging_slots: list[int] = field(default_factory=list)

    def submit_active_snapshot(
        self, tp_active_ranks: Optional[torch.Tensor] = None
    ) -> None:
        """Enqueue an async copy of active_ranks into the pinned staging
        buffer. No explicit event is recorded; the scheduler relies on
        copy_done.synchronize() (same forward_stream, recorded strictly
        later) to act as the barrier before reading the staging buffer."""
        if self.active_ranks is None or self.staging_active_ranks_cpu_slots is None:
            return

        slot = self.next_staging_slot
        self.next_staging_slot ^= 1
        staging_active_ranks_cpu = self.staging_active_ranks_cpu_slots[slot]
        self.pending_staging_slots.append(slot)
        assert len(self.pending_staging_slots) <= 2

        if self.active_ranks.device.type == "cuda":
            staging_active_ranks_cpu.copy_(self.active_ranks, non_blocking=True)
        else:
            # CPU backend: fully synchronous, no stream involved.
            staging_active_ranks_cpu.copy_(self.active_ranks)

        if tp_active_ranks is not None:
            if (
                self.tp_active_ranks_cpu is None
                or self.staging_tp_active_ranks_cpu_slots is None
            ):
                raise RuntimeError("TP active-rank snapshot buffers are not initialized")
            assert tp_active_ranks.numel() == self.tp_active_ranks_cpu.numel()
            staging_tp_active_ranks_cpu = self.staging_tp_active_ranks_cpu_slots[slot]
            if tp_active_ranks.device.type == "cuda":
                staging_tp_active_ranks_cpu.copy_(tp_active_ranks, non_blocking=True)
            else:
                staging_tp_active_ranks_cpu.copy_(tp_active_ranks)

    def commit_active_snapshot(self) -> bool:
        """Commit the oldest unconsumed staging snapshot into CPU mirrors.

        Caller must have already synchronized the stream the submit was enqueued
        on (e.g. via copy_done.synchronize) so staging buffers are ready.
        Returns True if a snapshot was committed.
        """
        if (
            not self.pending_staging_slots
            or self.staging_active_ranks_cpu_slots is None
        ):
            return False
        slot = self.pending_staging_slots.pop(0)
        snapshot = self.staging_active_ranks_cpu_slots[slot]
        self.active_ranks_cpu.copy_(snapshot)
        if self.staging_tp_active_ranks_cpu_slots is not None:
            self.tp_active_ranks_cpu.copy_(
                self.staging_tp_active_ranks_cpu_slots[slot]
            )
        return True

    def is_stale_snapshot(self) -> bool:
        """Return True iff a new fault has appeared since the last handled
        snapshot. Call commit_active_snapshot() before this check."""
        return not torch.equal(
            self.active_ranks_cpu, self.last_handled_active_ranks_cpu
        )

    def mark_snapshot_handled(self) -> None:
        """Snap last_handled to the current staging value so subsequent
        is_stale_snapshot() calls only flag *new* faults."""
        if self.active_ranks_cpu is not None:
            self.last_handled_active_ranks_cpu.copy_(self.active_ranks_cpu)

    def clear_pending_snapshots(self) -> None:
        self.pending_staging_slots.clear()

    def sync_active_to_cpu(self):
        if self.active_ranks is not None and self.active_ranks_cpu is not None:
            self.active_ranks_cpu.copy_(self.active_ranks.detach().cpu())

    def snapshot_active_to_last(self):
        if (
            self.active_ranks is not None
            and self.last_handled_active_ranks_cpu is not None
        ):
            self.last_handled_active_ranks_cpu.copy_(self.active_ranks.detach().cpu())

    def reset(self):
        if self.active_ranks is not None:
            self.active_ranks.fill_(1)
            self.snapshot_active_to_last()
            self.sync_active_to_cpu()
        self.clear_pending_snapshots()
        if self.staging_active_ranks_cpu_slots is not None:
            for staging_active_ranks_cpu in self.staging_active_ranks_cpu_slots:
                staging_active_ranks_cpu.fill_(1)
        if self.tp_active_ranks_cpu is not None:
            self.tp_active_ranks_cpu.fill_(1)
        if self.staging_tp_active_ranks_cpu_slots is not None:
            for staging_tp_active_ranks_cpu in self.staging_tp_active_ranks_cpu_slots:
                staging_tp_active_ranks_cpu.fill_(1)


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
            cls._instance = cls._build_state(
                ep_size=None, server_args=server_args, device=None
            )
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
        server_args: Optional[ServerArgs] = None,
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

        if server_args is not None:
            tp_size = server_args.tp_size
            tp_active_ranks_cpu = torch.ones(tp_size, dtype=torch.int32, device="cpu")
            if active.device.type == "cuda":
                staging_tp_active_ranks_cpu_slots = (
                    tp_active_ranks_cpu.clone().pin_memory(),
                    tp_active_ranks_cpu.clone().pin_memory(),
                )
            else:
                staging_tp_active_ranks_cpu_slots = (
                    tp_active_ranks_cpu.clone(),
                    tp_active_ranks_cpu.clone(),
                )
        else:
            tp_active_ranks_cpu = None
            staging_tp_active_ranks_cpu_slots = None

        return ElasticEPState(
            active_ranks=active,
            active_ranks_cpu=active_cpu.clone(),
            last_handled_active_ranks_cpu=active_cpu,
            staging_active_ranks_cpu_slots=staging_active_ranks_cpu_slots,
            tp_active_ranks_cpu=tp_active_ranks_cpu,
            staging_tp_active_ranks_cpu_slots=staging_tp_active_ranks_cpu_slots,
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


def _get_process_group_backend(process_group, device: str):
    return process_group._get_backend(torch.device(device))


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


def _wait_for_peer_state(mooncake_ep, backend, ranks: List[int]) -> None:
    # Relaunched ranks become recoverable asynchronously, so we poll until the
    # target backend reports all requested peers as ready.
    while not all(mooncake_ep.get_peer_state(backend, ranks)):
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

    world_backend = _get_process_group_backend(torch.distributed.group.WORLD, "cuda")
    return all(mooncake_ep.get_peer_state(world_backend, global_ranks))


def try_recover_ranks(global_ranks: List[int]) -> bool:
    if not can_recover_ranks(global_ranks):
        # The relaunched ranks have not finished initializing yet.
        return False
    recover_ranks(global_ranks)
    return True


def recover_ranks(global_ranks: List[int]) -> None:
    from mooncake import ep as mooncake_ep

    # Recover the world backend first, then recover each derived process group
    # using ranks mapped into that group's local rank space.
    world_backend = _get_process_group_backend(torch.distributed.group.WORLD, "cuda")
    mooncake_ep.recover_ranks(world_backend, global_ranks)

    for group in _iter_live_parallel_groups():
        group_local_ranks = _map_global_to_group_local_ranks(group.ranks, global_ranks)
        if not group_local_ranks:
            continue

        device_backend = _get_process_group_backend(group.device_group, "cuda")
        _wait_for_peer_state(mooncake_ep, device_backend, group_local_ranks)
        mooncake_ep.recover_ranks(device_backend, group_local_ranks)

        cpu_backend = _get_process_group_backend(group.cpu_group, "cpu")
        _wait_for_peer_state(mooncake_ep, cpu_backend, group_local_ranks)
        mooncake_ep.recover_ranks(cpu_backend, group_local_ranks)
        _maybe_create_message_queue(group)

    _refresh_ep_members()


def join_process_groups():
    from mooncake import ep as mooncake_ep

    def join_backend(label: str, backend) -> None:
        logger.info("Recovered rank joining Mooncake backend %s", label)
        mooncake_ep.join_group(backend)

    join_backend(
        "default_world",
        _get_process_group_backend(torch.distributed.group.WORLD, "cuda"),
    )

    for group in _iter_live_parallel_groups():
        if group.world_size <= 1:
            continue

        join_backend(
            f"{group.unique_name}:device",
            _get_process_group_backend(group.device_group, "cuda"),
        )
        join_backend(
            f"{group.unique_name}:cpu",
            _get_process_group_backend(group.cpu_group, "cpu"),
        )
        _maybe_create_message_queue(group)

    _refresh_ep_members()
