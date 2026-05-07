from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.managers.tp_worker import TpModelWorker


class BaseDraftWorker(ABC):
    @abstractmethod
    def draft():
        pass

    @abstractmethod
    def draft_extend():
        pass

    def alloc_memory_pool(self, **kwargs):
        pass

    def init_backends(self):
        """Initialize standard backends (no cuda graphs) then draft-specific backends.

        Subclasses should wrap this with their context managers (draft_tp_context,
        speculative_moe_backend_context, etc.) rather than reimplementing the logic.
        """
        self.draft_worker.init_backends(disable_cuda_graph=True)
        self.init_attention_backend()
        self.init_cuda_graphs()


class BaseSpecWorker(ABC):
    @property
    @abstractmethod
    def target_worker(self) -> TpModelWorker:
        pass

    @property
    @abstractmethod
    def draft_worker(self) -> BaseDraftWorker:
        pass

    @abstractmethod
    def clear_cache_pool(self):
        # TODO: move this abstract method to BaseTpWorker and call through self.model_runner
        pass

    def alloc_memory_pool(self, **kwargs):
        pass

    def init_backends(self):
        pass
