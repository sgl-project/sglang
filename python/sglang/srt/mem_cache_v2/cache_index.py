from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

import torch

type KeyFunc_t = Callable[[list[int]], Any]


@dataclass(frozen=True)
class MatchResult:
    """Represent the result of match_prefix.
    Fields:
        matched_indices: Indices of tokens in the matched prefix (tensor).
        allocation_key: Key for cache index to manage the allocation of the request.
    """

    matched_indices: torch.Tensor
    allocation_key: Any


class CacheIndex(ABC):
    @abstractmethod
    def match_prefix(self, key: tuple[int, ...] | Any) -> MatchResult:
        """
        Match the longest prefix of key that exists in the cache.

        Returns:
            MatchResult with:
                - matched_indices: tensor of indices for the matched prefix
                - allocation_key: node/key representing the matched prefix location
        """
        pass

    @abstractmethod
    def insert(
        self, key: tuple[int, ...], value: torch.Tensor, allocation_key: Any
    ) -> tuple[Any, torch.Tensor]:
        """
        Insert key-value into the cache.

        Args:
            key: Token sequence to insert
            value: Indices tensor corresponding to the key
            allocation_key: from previous match/insert

        Returns:
            Tuple of (new_allocation_key, tree_indices) where:
                - new_allocation_key: for the inserted sequence endpoint
                - tree_indices: Indices for the portion of key that's now in the tree
                  (used to calculate cached_len and determine overlaps for freeing)
        """
        pass

    @abstractmethod
    def allocate(self, allocation_key: Any) -> torch.Tensor:
        """
        Lock nodes from allocation_key to root, preparing for computation.

        Returns:
            Tensor of indices that are not ready (e.g., need transfer)
        """
        pass

    def free(self, allocation_key: Any) -> int:
        """
        Unlock/release nodes from allocation_key to root.

        Returns:
            Number of tokens unlocked (sum of node values along the path)
        """
        return 0

    @abstractmethod
    def evict(self, num_tokens: int) -> torch.Tensor:
        """
        Evict at least num_tokens from the cache (e.g., LRU policy).
        """
        pass


class ReqPool:
    def __init__(self, size: int, max_context_len: int, device: str):
        self.size = size
        self.max_context_len = max_context_len
        self.free_slots = list(range(size))
        self.device = device
        self.clear()

    def clear(self):
        self.free_slots = list(range(self.size))
        self.cpu_pool = torch.zeros(
            (self.size, self.max_context_len),
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        self.gpu_pool = torch.zeros(
            (self.size, self.max_context_len), dtype=torch.int32, device=self.device
        )

    def alloc(self):
        if len(self.free_slots) == 0:
            raise RuntimeError("No free slots in req pool")
        return self.free_slots.pop(0)

    def free(self, indices: list[int]):
        self.free_slots.extend(indices)

    def sync(self, indices: list[int]):
        for idx in indices:
            self.gpu_pool[idx].copy_(self.cpu_pool[idx], non_blocking=True)
