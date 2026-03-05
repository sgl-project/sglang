from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, NamedTuple, Optional

import torch


class LoadOperation(NamedTuple):
    rid: str
    device_indices: torch.Tensor
    node: Any


class BaseKVConnector(ABC):
    def __init__(self, tp_group: Any = None, tp_rank: int = 0):
        """Initialize connector with tensor-parallel context.

        Args:
            tp_group: Tensor-parallel process group.
            tp_rank: Rank within the tp group.
        """
        self.tp_group = tp_group
        self.tp_rank = tp_rank

    @abstractmethod
    def get_new_hit_length(
        self,
        token_ids: List[int],
        token_mask: torch.Tensor,
        update_state_for_load: bool = False,
        rid: Optional[str] = None,
    ) -> int:
        """Return number of externally cached tokens beyond the device hit.

        Args:
            token_ids: Full token id sequence.
            token_mask: Boolean mask; True for positions to match.
            update_state_for_load: Lock internal state until the load is
                started or cancelled via *rid*.
            rid: Request id for tracking the subsequent load task.
        Returns:
            Number of new matched tokens.
        """
        ...

    @abstractmethod
    def cancel_load_task(self, rid: str) -> None:
        """Cancel a locked load, releasing state acquired by get_new_hit_length.

        Args:
            rid: Request id previously passed to get_new_hit_length.
        """
        ...

    @abstractmethod
    def start_load_kv(
        self,
        task_id: int,
        load_ops: List[LoadOperation],
    ) -> None:
        """Start a batch of external-to-GPU load operations.

        Args:
            task_id: Caller-assigned id for completion tracking.
            load_ops: Pending load operations to execute.
        """
        ...

    @abstractmethod
    def check_completed_load_tasks(self) -> List[int]:
        """Return task_ids of completed load tasks.

        Returns:
            List of completed load task_ids.
        """
        ...

    @abstractmethod
    def start_store_kv(
        self,
        task_id: int,
        token_ids: List[int],
        kv_indices: torch.Tensor,
    ) -> None:
        """Asynchronously store KV cache to external storage.

        Args:
            task_id: Caller-assigned id for completion tracking.
            token_ids: Token id sequence to store.
            kv_indices: Corresponding GPU KV pool indices.
        """
        ...

    @abstractmethod
    def check_completed_store_tasks(self) -> List[int]:
        """Return task_ids of completed store tasks.

        Returns:
            List of completed store task_ids.
        """
        ...

    def prefetch(self, rid: str, token_ids: List[int]) -> None:
        """Start prefetching KV cache from external storage.

        Args:
            rid: Request id.
            token_ids: Token id sequence to prefetch.
        """
        pass

    def check_prefetch_completed(self, rid: str) -> bool:
        """Return True if the prefetch for *rid* has completed.

        Args:
            rid: Request id previously passed to prefetch.
        Returns:
            True if complete or no prefetch was needed.
        """
        return True

    def cancel_prefetch(self, rid: str) -> None:
        """Cancel an in-progress or pending prefetch for *rid*.

        Args:
            rid: Request id previously passed to prefetch.
        """
        pass

    @property
    def layer_done_counter(self) -> Any:
        """Return the layer-wise transfer counter, or None."""
        return None

    def register_layer_transfer_counter(self, kvcache: Any) -> None:
        """Register the layer transfer counter with the KV cache pool.

        Args:
            kvcache: KV cache pool instance.
        """
        pass

    def reset(self) -> None:
        """Reset connector state (called on cache reset)."""
        pass

    def shutdown(self) -> None:
        """Cleanup resources."""
        pass
