from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Iterator, List, Optional

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
            self.active_ranks.fill_(1)
            self.snapshot_active_to_last()
            self.sync_active_to_cpu()


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


def try_recover_ranks(global_ranks: List[int]) -> bool:
    from mooncake import ep as mooncake_ep

    world_backend = _get_process_group_backend(torch.distributed.group.WORLD, "cuda")
    if not all(mooncake_ep.get_peer_state(world_backend, global_ranks)):
        # The relaunched ranks have not finished initializing yet.
        return False

    # Recover the world backend first, then recover each derived process group
    # using ranks mapped into that group's local rank space.
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
    return True


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
