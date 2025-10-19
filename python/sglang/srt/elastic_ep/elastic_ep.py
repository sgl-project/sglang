from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Union

import torch

from sglang.srt.managers.schedule_batch import ServerArgs
from sglang.srt.utils import is_cpu, is_cuda


@dataclass
class ElasticEPState:
    _active_ranks: Optional[torch.Tensor]
    _last_active_ranks: Optional[torch.Tensor]
    _active_ranks_cpu: Optional[torch.Tensor]
    on_forward: Optional[Callable] = None
    rank_status: Optional[torch.Tensor] = None

    def is_active_equal_last(self) -> bool:
        return torch.equal(self._active_ranks, self._last_active_ranks)

    def sync_active_to_cpu(self):
        if self._active_ranks is not None:
            self._active_ranks_cpu = self._active_ranks.detach().cpu().clone()

    def snapshot_active_to_last(self):
        if self._active_ranks is not None:
            self._last_active_ranks = self._active_ranks.clone()


class ElasticEPStateManager:
    _instance: Optional[ElasticEPState] = None
    _lock = threading.Lock()

    @staticmethod
    def on_forward_mooncake(
        state: ElasticEPState, status: torch.Tensor = None, **kwargs
    ):
        state._active_ranks = state.rank_status.to(dtype=torch.int32)

    @staticmethod
    def on_forward_deepep(state: ElasticEPState, status: torch.Tensor = None, **kwargs):
        state._active_ranks = 1 - state.rank_status.to(torch.int32)

    @classmethod
    def instance(cls) -> ElasticEPState:
        return cls._instance

    @classmethod
    def init(cls, server_args: ServerArgs):
        with cls._lock:
            if cls._instance is not None:
                return cls._instance

            if server_args.elastic_ep_backend is not None:
                cls._instance = cls._build_state(
                    ep_size=None,
                    device=None,
                    backend_type=server_args.elastic_ep_backend,
                )
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
        ep_size: Optional[int],
        device: Optional[torch.device],
        backend_type: str = "none",
    ) -> ElasticEPState:

        active = cls.create_rank_state(ep_size=ep_size, device=device, value=1)

        if backend_type == "mooncake":
            on_forward = cls.on_forward_mooncake
        elif backend_type == "deepep":
            on_forward = cls.on_forward_deepep
        else:
            on_forward = None

        return ElasticEPState(
            _active_ranks=active,
            _last_active_ranks=active.clone(),
            _active_ranks_cpu=active.detach().cpu().clone(),
            rank_status=active.clone(),
            on_forward=on_forward,
        )

    @classmethod
    def create_rank_state(
        cls, *, ep_size: Optional[int], device: Optional[torch.device], value: int = 1
    ) -> torch.Tensor:
        size = ep_size if ep_size is not None else torch.distributed.get_world_size()
        dev = device if device is not None else cls._select_device()

        return torch.full((size,), value, dtype=torch.int32, device=dev)
