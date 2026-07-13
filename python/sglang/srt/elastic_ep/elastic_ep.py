from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, Iterator, List, Optional

import torch

from sglang.srt.distributed import parallel_state
from sglang.srt.managers.schedule_batch import ServerArgs
from sglang.srt.utils import is_cpu, is_cuda

logger = logging.getLogger(__name__)


@dataclass
class ElasticEPState:
    active_ranks: Optional[torch.Tensor]
    last_active_ranks: Optional[torch.Tensor]
    active_ranks_cpu: Optional[torch.Tensor]
    effective_ep_size: int = 0
    pending_ep_size: Optional[int] = None
    scale_phase: str = "idle"
    last_error: Optional[str] = None
    pending_since: Optional[float] = None
    original_ep_size: int = 0
    has_scaled: bool = False
    ep_join_rank_offset: int = 0

    def is_active_equal_last(self) -> bool:
        return torch.equal(self.active_ranks, self.last_active_ranks)

    def sync_active_to_cpu(self):
        if self.active_ranks is not None:
            self.active_ranks_cpu = self.active_ranks.detach().cpu().clone()

    def snapshot_active_to_last(self):
        if self.active_ranks is not None:
            self.last_active_ranks = self.active_ranks.clone()

    def reset(self):
        if self.active_ranks is not None:
            # Reserved slots stay inactive until their ranks join.
            self.active_ranks.zero_()
            self.active_ranks[: self.effective_ep_size] = 1
            self.snapshot_active_to_last()
            self.sync_active_to_cpu()


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

            inst = cls._build_state(ep_size=active_rank_capacity, device=None)
            inst.effective_ep_size = world_size
            inst.original_ep_size = world_size
            if active_rank_capacity > world_size:
                inst.active_ranks[world_size:].zero_()
                inst.snapshot_active_to_last()
                inst.sync_active_to_cpu()

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
        inst.snapshot_active_to_last()
        inst.sync_active_to_cpu()

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
        cls, *, ep_size: Optional[int] = None, device: Optional[torch.device] = None
    ) -> ElasticEPState:
        active = cls.healthy_rank_state(ep_size=ep_size, device=device)
        return ElasticEPState(
            active_ranks=active,
            last_active_ranks=active.clone(),
            active_ranks_cpu=active.detach().cpu().clone(),
        )

    @classmethod
    def healthy_rank_state(
        cls, *, ep_size: Optional[int] = None, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        size = ep_size if ep_size is not None else torch.distributed.get_world_size()
        dev = device if device is not None else cls._select_device()

        return torch.ones(size, dtype=torch.int32, device=dev)

    @classmethod
    def begin_scale(cls, n: int) -> bool:
        inst = cls._instance
        if inst is None:
            return False
        if inst.pending_ep_size is not None:
            return False
        inst.pending_ep_size = n
        inst.scale_phase = "pending"
        inst.last_error = None
        inst.pending_since = time.monotonic()
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
        if inst is None or inst.active_ranks_cpu is None:
            return False
        if inst.pending_ep_size is not None:
            return True
        active_count = int(inst.active_ranks_cpu[: inst.effective_ep_size].sum().item())
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
    if getattr(sa, "max_ep_size", None) is None:
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


def _wait_for_peer_state(mooncake_ep, backend, ranks: List[int]) -> None:
    while not all(mooncake_ep.get_peer_state(backend, ranks)):
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


def try_recover_ranks(global_ranks: List[int]) -> bool:
    """Recover ranks in WORLD and every launch-time parallel group."""
    if not _try_recover_world(global_ranks):
        return False

    from mooncake import ep as mooncake_ep

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
    return True


def _join_world_group() -> None:
    from mooncake import ep as mooncake_ep

    mooncake_ep.join_group(torch.distributed.group.WORLD)


def join_scale_process_group() -> None:
    """Join the expandable WORLD group for an append-only scale operation."""
    _join_world_group()
    _refresh_ep_members()


def join_process_groups() -> None:
    """Rejoin WORLD and every launch-time parallel group after recovery."""
    from mooncake import ep as mooncake_ep

    _join_world_group()
    for group in _iter_live_parallel_groups():
        if group.world_size <= 1:
            continue
        mooncake_ep.join_group(group.device_group)
        mooncake_ep.join_group(group.cpu_group)
        _maybe_create_message_queue(group)

    _refresh_ep_members()
