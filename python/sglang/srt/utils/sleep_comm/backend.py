"""Platform-supplied executor for a platform's optional sleep-mode comm tag.

Core owns the tag bookkeeping and decides when ``release``/``resume`` run; the
backend owns how device communication resources are freed and restored. A
platform opts in by calling ``register_sleep_comm_backend()`` from its own
initialization path. The helpers at the bottom are optional.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, ClassVar, FrozenSet, List, Optional

import torch
import torch.distributed

from sglang.srt.distributed import parallel_state

logger = logging.getLogger(__name__)

_BACKEND_CLS: Optional[type] = None


class SleepCommBackend(ABC):
    """Release/resume implementation for one platform tag."""

    #: The memory-occupation tag this backend serves.
    tag: ClassVar[str]

    #: The backend owns graph drop/recapture, so core skips the TMS graph pause.
    handles_cuda_graph: ClassVar[bool] = False

    #: Tags that must be resumed together, because ``resume`` depends on them.
    resume_requires: ClassVar[FrozenSet[str]] = frozenset()

    @classmethod
    def unsupported_reason(cls) -> Optional[str]:
        """Why this backend cannot serve its tag here, or None when it can."""
        return None

    def __init__(self, tp_worker, draft_worker):
        self.tp_worker = tp_worker
        self.draft_worker = draft_worker

    @abstractmethod
    def release(self) -> None:
        """Release everything the tag covers. Fully platform-defined."""

    @abstractmethod
    def resume(self) -> None:
        """Restore what ``release`` freed. Fully platform-defined."""


def register_sleep_comm_backend(backend_cls: type) -> None:
    """Register the current process's backend (one accelerator per process)."""
    global _BACKEND_CLS
    _BACKEND_CLS = backend_cls


def get_sleep_comm_backend_class() -> Optional[type]:
    """Backend class registered by the current process, or None."""
    return _BACKEND_CLS


def enabled_platform_tags() -> tuple:
    """Tags the current process's registered backend serves."""
    backend_cls = get_sleep_comm_backend_class()
    return (backend_cls.tag,) if backend_cls is not None else ()


def make_sleep_comm_backend(tp_worker, draft_worker) -> Optional[SleepCommBackend]:
    backend_cls = get_sleep_comm_backend_class()
    if backend_cls is None:
        return None
    return backend_cls(tp_worker=tp_worker, draft_worker=draft_worker)


def iter_rebuildable_groups() -> List[parallel_state.GroupCoordinator]:
    """Alive GroupCoordinators that can rebuild their device group, by name."""
    groups = [
        group
        for group in parallel_state.get_all_groups()
        if group.can_rebuild_device_group
    ]
    groups.sort(key=lambda group: group.unique_name)
    return groups


def world_cpu_group():
    return parallel_state.get_world_group().cpu_group


def barrier(cpu_group, what: str) -> None:
    logger.debug(f"[sleep-comm] barrier: {what}")
    torch.distributed.barrier(group=cpu_group)


def group_signatures(groups) -> List[Any]:
    """(unique_name, group_ranks) pairs; the rank set must match across ranks."""
    return [
        (group.unique_name, tuple(tuple(r) for r in group.rebuild_group_ranks))
        for group in groups
    ]


def assert_ranks_agree(signatures: List[Any], cpu_group) -> None:
    """Abort before touching anything if ranks disagree on the group set."""
    gathered: List[Any] = [None] * torch.distributed.get_world_size(cpu_group)
    torch.distributed.all_gather_object(gathered, signatures, group=cpu_group)
    if any(other != signatures for other in gathered):
        raise RuntimeError(
            f"[sleep-comm] ranks disagree on the comm-cleanup group set: "
            f"{gathered}. Aborting; no resource has been modified."
        )


def all_ranks_ok(flag: bool, cpu_group) -> bool:
    """AND-reduce ``flag`` over ``cpu_group`` (MIN on int32)."""
    tensor = torch.tensor([1 if flag else 0], dtype=torch.int32)
    torch.distributed.all_reduce(
        tensor, op=torch.distributed.ReduceOp.MIN, group=cpu_group
    )
    return tensor.item() != 0
