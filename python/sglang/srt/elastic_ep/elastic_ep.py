from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Iterator, List, Optional

import torch

from sglang.srt.distributed import get_world_group, parallel_state
from sglang.srt.distributed.utils import get_global_tcp_store
from sglang.srt.managers.schedule_batch import ServerArgs
from sglang.srt.utils import is_cpu, is_cuda

logger = logging.getLogger(__name__)

_SCALE_COHORT_KEY_PREFIX = "elastic_ep/scale_cohort"


def register_scale_cohort(rank_offset: int, target_ep_size: int) -> None:
    store = get_global_tcp_store()
    if store is None:
        raise RuntimeError("Elastic EP scale-up requires the global TCPStore.")
    store.set(f"{_SCALE_COHORT_KEY_PREFIX}/{rank_offset}", str(target_ep_size).encode())


def get_scale_cohort_target(rank_offset: int) -> Optional[int]:
    store = get_global_tcp_store()
    if store is None:
        return None
    key = f"{_SCALE_COHORT_KEY_PREFIX}/{rank_offset}"
    if not store.check([key]):
        return None
    return int(store.get(key).decode())


@dataclass
class ElasticEPState:
    """Elastic EP active-rank state.

    Naming convention in this class:
    - global: tensor indices are world ranks.
    - pg: the mask originated from a process group, already mapped to global.
      PG masks may cover only a prefix of capacity because launch-time groups
      exclude later-admitted ranks.
    - _cpu: tensor storage is on CPU; it does not mean CPU process-group source.
    """

    active_ranks: torch.Tensor
    committed_active_ranks_cpu: torch.Tensor
    last_handled_committed_active_ranks_cpu: torch.Tensor
    staging_active_ranks_cpu_slots: tuple[torch.Tensor, torch.Tensor]
    staging_global_pg_active_ranks_cpu_slots: tuple[torch.Tensor, torch.Tensor]
    next_staging_slot: int = 0
    pending_staging_slots: list[int] = field(default_factory=list)
    effective_ep_size: int = 0
    pending_ep_size: Optional[int] = None
    scale_phase: str = "idle"
    last_error: Optional[str] = None
    pending_since: Optional[float] = None
    original_ep_size: int = 0
    has_scaled: bool = False
    ep_join_rank_offset: int = 0

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

        num_pg_ranks = global_pg_active_ranks.numel()
        assert num_pg_ranks <= self.committed_active_ranks_cpu.numel()
        if global_pg_active_ranks.device.type == "cuda":
            staging_global_pg_active_ranks_cpu[:num_pg_ranks].copy_(
                global_pg_active_ranks, non_blocking=non_blocking
            )
        else:
            staging_global_pg_active_ranks_cpu[:num_pg_ranks].copy_(
                global_pg_active_ranks
            )

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

        num_pg_ranks = global_pg_active_ranks_cpu.numel()
        assert num_pg_ranks <= self.committed_active_ranks_cpu.numel()
        self.committed_active_ranks_cpu.bitwise_and_(snapshot)
        self.committed_active_ranks_cpu.bitwise_and_(global_pg_snapshot)
        self.committed_active_ranks_cpu[:num_pg_ranks].bitwise_and_(
            global_pg_active_ranks_cpu
        )
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
        # Reserved slots stay inactive until their ranks join.
        self.active_ranks.zero_()
        self.active_ranks[: self.effective_ep_size] = 1
        self.realign_snapshots_to_active()

    def realign_snapshots_to_active(self):
        """Re-baseline all snapshot state to the current active_ranks pattern.

        Must run after any legitimate rewrite of active_ranks (fault recovery
        reset, scale-up admission): the committed mirror is AND-sticky, so stale
        staging slots holding the old pattern would otherwise re-kill ranks that
        just became active, and a committed/last-handled mismatch would trigger a
        spurious fault retract at the next result boundary. Pending staging slots
        are overwritten, not dropped, so each submitted forward still has one
        commit; those commits become idempotent ANDs against the new baseline.
        Realign drains the device before overwriting the staging buffers so an
        in-flight D2H copy cannot race the overwrite.
        """
        if self.active_ranks.device.type == "cuda":
            # Drain any in-flight async staging copies before overwriting the
            # pinned buffers; realign only runs on rare scale/recovery events.
            torch.cuda.synchronize()

        self.snapshot_active_to_last()
        self.sync_active_to_cpu()
        pattern = self.committed_active_ranks_cpu
        for staging_active_ranks_cpu in self.staging_active_ranks_cpu_slots:
            staging_active_ranks_cpu.copy_(pattern)
        for (
            staging_global_pg_active_ranks_cpu
        ) in self.staging_global_pg_active_ranks_cpu_slots:
            staging_global_pg_active_ranks_cpu.copy_(pattern)


class ElasticEPStateManager:
    _instance: Optional[ElasticEPState] = None
    _on_scale: Optional[Callable[[int, int], None]] = None

    @classmethod
    def instance(cls) -> ElasticEPState:
        return cls._instance

    @classmethod
    def init(cls, server_args: ServerArgs):
        if cls._instance is not None:
            return cls._instance

        if server_args.elastic_ep_backend is not None:
            world_size = torch.distributed.get_world_size()
            active_rank_capacity = server_args.max_ep_size or world_size
            assert active_rank_capacity >= world_size, (
                f"--max-ep-size ({active_rank_capacity}) must be >= "
                f"world_size ({world_size})."
            )

            inst = cls._build_state(
                ep_size=active_rank_capacity,
                effective_ep_size=world_size,
                device=None,
            )
            inst.effective_ep_size = world_size
            inst.original_ep_size = world_size

            if server_args.moe_a2a_backend == "nixl":
                cls._on_scale = cls._on_scale_nixl

            inst.ep_join_rank_offset = server_args.ep_join_rank_offset
            if server_args.is_ep_joiner:
                cls._init_joiner_state(inst, server_args)

            cls._instance = inst

        return cls._instance

    @classmethod
    def _init_joiner_state(cls, inst: ElasticEPState, server_args: ServerArgs) -> None:
        global_rank = torch.distributed.get_rank()
        inst.active_ranks.zero_()
        inst.active_ranks[global_rank] = 1
        inst.realign_snapshots_to_active()

        if server_args.ep_join_mode == "scale":
            inst.effective_ep_size = (
                server_args.ep_join_rank_offset + server_args.tp_size
            )
            inst.original_ep_size = (
                server_args.elastic_ep_initial_size or server_args.ep_join_rank_offset
            )
            inst.has_scaled = True
        else:
            world_size = torch.distributed.get_world_size()
            inst.effective_ep_size = world_size
            inst.original_ep_size = world_size

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
        effective_ep_size: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> ElasticEPState:
        active = cls.healthy_rank_state(ep_size=ep_size, device=device)
        if effective_ep_size is not None:
            active[effective_ep_size:].zero_()
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

    @classmethod
    def request_scale(cls, n: int) -> bool:
        inst = cls._instance
        if inst is None:
            return False
        if (
            inst.pending_ep_size is not None
            or inst.scale_phase == "recovery_unsupported"
        ):
            return False
        inst.pending_ep_size = n
        inst.scale_phase = "waiting_for_cohort"
        inst.last_error = None
        inst.pending_since = time.monotonic()
        return True

    @classmethod
    def begin_scale(cls) -> bool:
        inst = cls._instance
        if (
            inst is None
            or inst.pending_ep_size is None
            or inst.scale_phase != "waiting_for_cohort"
        ):
            return False
        inst.scale_phase = "pending"
        return True

    @classmethod
    def mark_joining(cls) -> None:
        cls._mark_phase("joining")

    @classmethod
    def mark_configuring_data_plane(cls) -> None:
        cls._mark_phase("configuring_data_plane")

    @classmethod
    def mark_syncing_new_world(cls) -> None:
        cls._mark_phase("syncing_new_world")

    @classmethod
    def _mark_phase(cls, phase: str) -> None:
        inst = cls._instance
        if inst is not None and inst.pending_ep_size is not None:
            inst.scale_phase = phase

    @classmethod
    def commit_scale(cls) -> None:
        inst = cls._instance
        if inst is None or inst.pending_ep_size is None:
            return
        inst.effective_ep_size = inst.pending_ep_size
        inst.pending_ep_size = None
        inst.has_scaled = True
        inst.scale_phase = "serving_expanded"
        inst.last_error = None
        inst.pending_since = None
        inst.reset()

    @classmethod
    def fail_scale(cls, error: str) -> None:
        inst = cls._instance
        if inst is None:
            return
        inst.pending_ep_size = None
        inst.scale_phase = "failed"
        inst.last_error = error
        inst.pending_since = None
        inst.reset()

    @classmethod
    def fail_recovery(cls, error: str) -> None:
        inst = cls._instance
        if inst is None:
            return
        inst.scale_phase = "recovery_unsupported"
        inst.last_error = error

    @classmethod
    def get_effective_ep_size(cls) -> int:
        inst = cls._instance
        assert inst is not None, "Elastic EP state is not initialized."
        return inst.effective_ep_size

    @classmethod
    def get_pending_ep_size(cls) -> Optional[int]:
        inst = cls._instance
        if inst is None:
            return None
        return inst.pending_ep_size

    @classmethod
    def get_scale_phase(cls) -> str:
        inst = cls._instance
        if inst is None:
            return "disabled"
        return inst.scale_phase

    @classmethod
    def get_last_error(cls) -> Optional[str]:
        inst = cls._instance
        if inst is None:
            return None
        return inst.last_error

    @classmethod
    def get_ep_join_rank_offset(cls) -> int:
        inst = cls._instance
        if inst is None:
            return 0
        return inst.ep_join_rank_offset

    @classmethod
    def on_scale(cls, from_ep_size: int, to_ep_size: int) -> None:
        if cls._on_scale is not None:
            cls._on_scale(from_ep_size, to_ep_size)

    @staticmethod
    def _on_scale_nixl(from_ep_size: int, to_ep_size: int) -> None:
        from sglang.srt.layers.moe.token_dispatcher.nixl import NixlEPBuffer

        NixlEPBuffer.on_scale(from_ep_size, to_ep_size)

    @classmethod
    def is_scaling(cls) -> bool:
        """Return whether a scale or recovery operation is pending.

        The CPU snapshot is authoritative because rank polling uses it too.
        """
        inst = cls._instance
        if inst is None:
            return False
        if inst.scale_phase == "recovery_unsupported":
            return False
        if inst.pending_ep_size is not None:
            return True
        active_count = int(
            inst.committed_active_ranks_cpu[: inst.effective_ep_size].sum().item()
        )
        return active_count < inst.effective_ep_size


def elastic_expanded_world_enabled() -> bool:
    """Return whether execution uses ranks admitted after server launch.

    Launch-time TP groups exclude ranks admitted during scale-up.
    """
    from sglang.srt.runtime_context import get_server_args

    inst = ElasticEPStateManager.instance()
    if inst is None:
        return False
    sa = get_server_args()
    if sa.max_ep_size is None:
        return False
    active_target_size = inst.effective_ep_size
    if inst.pending_ep_size is not None and inst.scale_phase in (
        "configuring_data_plane",
        "syncing_new_world",
    ):
        active_target_size = inst.pending_ep_size

    return active_target_size > inst.original_ep_size


def _refresh_ep_members() -> None:
    from sglang.srt.layers.moe.token_dispatcher.mooncake import EPBuffer

    buffer = EPBuffer.get_existing_buffer()
    if buffer is not None:
        buffer.update_ep_member()


_PEER_STATE_POLL_INTERVAL_SEC = 0.01


def _iter_live_parallel_groups() -> Iterator[parallel_state.GroupCoordinator]:
    groups = []
    for group_ref in parallel_state._groups.values():
        group = group_ref()
        if group is not None:
            groups.append(group)
    yield from sorted(groups, key=lambda group: group.unique_name)


def _map_global_to_group_local_ranks(
    group_ranks: List[int], global_ranks: List[int]
) -> List[int]:
    rank_to_local = {rank: index for index, rank in enumerate(group_ranks)}
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


def _try_recover_world(global_ranks: List[int]) -> bool:
    from mooncake import ep as mooncake_ep

    world_backend = torch.distributed.group.WORLD
    if not all(mooncake_ep.get_peer_state(world_backend, global_ranks)):
        return False

    mooncake_ep.recover_ranks(world_backend, global_ranks)
    logger.debug("[Elastic EP][recover] WORLD recover_ranks(%s) done", global_ranks)
    return True


def try_admit_scale_ranks(global_ranks: List[int]) -> bool:
    """Admit append-only ranks into the expandable WORLD group."""
    if not _try_recover_world(global_ranks):
        return False

    _refresh_ep_members()
    return True


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
        local_ranks = _map_global_to_group_local_ranks(group.ranks, global_ranks)
        if not local_ranks:
            continue

        _wait_for_peer_state(mooncake_ep, group.device_group, local_ranks)
        mooncake_ep.recover_ranks(group.device_group, local_ranks)
        _wait_for_peer_state(mooncake_ep, group.cpu_group, local_ranks)
        mooncake_ep.recover_ranks(group.cpu_group, local_ranks)
        _maybe_create_message_queue(group)

    _refresh_ep_members()


def join_process_group(label: str, process_group) -> None:
    from mooncake import ep as mooncake_ep

    logger.info("Recovered rank joining Mooncake process group %s", label)
    mooncake_ep.join_group(process_group)


def _join_world_group() -> None:
    join_process_group(
        "default_world",
        torch.distributed.group.WORLD,
    )


def join_scale_process_group() -> None:
    """Join the expandable WORLD group for an append-only scale operation."""
    _join_world_group()
    _refresh_ep_members()


def join_process_groups() -> None:
    """Rejoin WORLD and every launch-time parallel group after recovery."""
    _join_world_group()
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


def get_healthy_expert_location_src_rank(
    *, invoked_in_elastic_ep_rejoin_path: bool
) -> int:
    world_group = get_world_group()
    # NOTE: do not key off `self.server_args.elastic_ep_rejoin` here.
    # A rank that was started as a rejoin rank may later act as a healthy
    # rank in a subsequent recovery cycle.
    local_rejoin_flag = bool(invoked_in_elastic_ep_rejoin_path)
    gathered_rejoin_flags = world_group.all_gather_object(local_rejoin_flag)

    for rank_in_group, is_rejoin_rank in enumerate(gathered_rejoin_flags):
        if not is_rejoin_rank:
            return world_group.ranks[rank_in_group]

    raise RuntimeError(
        "No healthy rank found for broadcasting expert location metadata. "
        "All ranks are marked as elastic_ep_rejoin."
    )
