from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

from sglang.srt.managers.schedule_batch import ServerArgs
from sglang.srt.utils import is_cpu, is_cuda


@dataclass
class ElasticEPState:
    active_ranks: Optional[torch.Tensor]
    active_ranks_cpu: Optional[torch.Tensor]
    last_handled_active_ranks_cpu: Optional[torch.Tensor]
    staging_active_ranks_cpu: list[torch.Tensor] = field(default_factory=list)
    staging_events: list[torch.cuda.Event] = field(default_factory=list)
    staging_event_recorded: list[bool] = field(default_factory=list)
    next_staging_slot: int = 0
    latest_staging_slot: Optional[int] = None
    pending_cpu_snapshot: bool = False

    def submit_active_snapshot(self) -> None:
        if self.active_ranks is None:
            return

        if self.active_ranks.device.type != "cuda":
            self.active_ranks_cpu.copy_(self.active_ranks)
            self.pending_cpu_snapshot = True
            return

        slot = self._get_staging_slot()
        self.staging_active_ranks_cpu[slot].copy_(self.active_ranks, non_blocking=True)
        event = self.staging_events[slot]
        event.record()
        self.staging_event_recorded[slot] = True
        self.latest_staging_slot = slot
        self.next_staging_slot = (slot + 1) % len(self.staging_active_ranks_cpu)

    def try_consume_ready_snapshot(self) -> bool:
        if self.active_ranks is None:
            return False

        if self.active_ranks.device.type != "cuda":
            if not self.pending_cpu_snapshot:
                return False
            self.pending_cpu_snapshot = False
            return not torch.equal(
                self.active_ranks_cpu, self.last_handled_active_ranks_cpu
            )

        slot = self.latest_staging_slot
        if slot is None:
            return False

        event = self.staging_events[slot]
        if not self.staging_event_recorded[slot] or not event.query():
            return False

        active_ranks_cpu = self.staging_active_ranks_cpu[slot]
        self.latest_staging_slot = None
        self.staging_event_recorded[slot] = False
        if torch.equal(active_ranks_cpu, self.last_handled_active_ranks_cpu):
            return False

        self.active_ranks_cpu.copy_(active_ranks_cpu)
        return True

    def mark_snapshot_handled(self) -> None:
        if self.active_ranks_cpu is not None:
            self.last_handled_active_ranks_cpu.copy_(self.active_ranks_cpu)

    def _get_staging_slot(self) -> int:
        slot = self.next_staging_slot
        event = self.staging_events[slot]
        if self.staging_event_recorded[slot] and not event.query():
            event.synchronize()
        return slot


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
        active_cpu = active.detach().cpu().clone()
        staging_active_ranks_cpu = []
        staging_events = []
        staging_event_recorded = []
        if active.device.type == "cuda":
            num_slots = 2
            staging_active_ranks_cpu = [
                torch.empty(active_cpu.shape, dtype=active_cpu.dtype, pin_memory=True)
                for _ in range(num_slots)
            ]
            staging_events = [
                torch.cuda.Event(enable_timing=False) for _ in range(num_slots)
            ]
            staging_event_recorded = [False] * num_slots

        return ElasticEPState(
            active_ranks=active,
            active_ranks_cpu=active_cpu.clone(),
            last_handled_active_ranks_cpu=active_cpu,
            staging_active_ranks_cpu=staging_active_ranks_cpu,
            staging_events=staging_events,
            staging_event_recorded=staging_event_recorded,
        )

    @classmethod
    def healthy_rank_state(
        cls, *, ep_size: Optional[int] = None, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        size = ep_size if ep_size is not None else torch.distributed.get_world_size()
        dev = device if device is not None else cls._select_device()

        return torch.ones(size, dtype=torch.int32, device=dev)
