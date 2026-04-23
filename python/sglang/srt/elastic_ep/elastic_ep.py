from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from sglang.srt.managers.schedule_batch import ServerArgs
from sglang.srt.utils import is_cpu, is_cuda


@dataclass
class ElasticEPState:
    active_ranks: Optional[torch.Tensor]
    active_ranks_cpu: Optional[torch.Tensor]
    last_handled_active_ranks_cpu: Optional[torch.Tensor]
    staging_active_ranks_cpu: Optional[torch.Tensor]

    def submit_active_snapshot(self) -> None:
        """Enqueue an async copy of active_ranks into the pinned staging
        buffer. No explicit event is recorded; the scheduler relies on
        copy_done.synchronize() (same forward_stream, recorded strictly
        later) to act as the barrier before reading the staging buffer."""
        if self.active_ranks is None or self.staging_active_ranks_cpu is None:
            return

        if self.active_ranks.device.type == "cuda":
            self.staging_active_ranks_cpu.copy_(self.active_ranks, non_blocking=True)
        else:
            # CPU backend: fully synchronous, no stream involved.
            self.staging_active_ranks_cpu.copy_(self.active_ranks)

    def is_stale_snapshot(self) -> bool:
        """Publish the most recently submitted staging snapshot into
        active_ranks_cpu (so downstream readers like EPLB rebalance see
        the current state), then return True iff a new fault has appeared
        since the last mark_snapshot_handled. Caller must have already
        synchronized the stream the submit was enqueued on (e.g. via
        copy_done.synchronize) so staging_active_ranks_cpu is known to
        hold the latest snapshot."""
        if self.staging_active_ranks_cpu is None:
            return False
        self.active_ranks_cpu.copy_(self.staging_active_ranks_cpu)
        return not torch.equal(
            self.active_ranks_cpu, self.last_handled_active_ranks_cpu
        )

    def mark_snapshot_handled(self) -> None:
        """Snap last_handled to the current staging value so subsequent
        is_stale_snapshot() calls only flag *new* faults."""
        if self.staging_active_ranks_cpu is not None:
            self.last_handled_active_ranks_cpu.copy_(self.staging_active_ranks_cpu)
        elif self.active_ranks_cpu is not None:
            self.last_handled_active_ranks_cpu.copy_(self.active_ranks_cpu)


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
        # Initialize staging to the healthy initial state so is_stale_snapshot()
        # before the first submit returns False (no false positive on startup).
        if active.device.type == "cuda":
            staging_active_ranks_cpu = active_cpu.clone().pin_memory()
        else:
            staging_active_ranks_cpu = active_cpu.clone()

        return ElasticEPState(
            active_ranks=active,
            active_ranks_cpu=active_cpu.clone(),
            last_handled_active_ranks_cpu=active_cpu,
            staging_active_ranks_cpu=staging_active_ranks_cpu,
        )

    @classmethod
    def healthy_rank_state(
        cls, *, ep_size: Optional[int] = None, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        size = ep_size if ep_size is not None else torch.distributed.get_world_size()
        dev = device if device is not None else cls._select_device()

        return torch.ones(size, dtype=torch.int32, device=dev)
