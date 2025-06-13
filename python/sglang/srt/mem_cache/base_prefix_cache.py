from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, List, NamedTuple
import torch

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
else:
    Req = Any  # Placeholder for Req type when not type checking

class MatchResult(NamedTuple):
    device_indices: torch.Tensor
    device_last_node: Any
    host_indices: torch.Tensor
    host_last_node: Any

class BasePrefixCache(ABC):
    """Cache can be indexed by either rid or key."""

    def empty_indices(self, device: str | torch.device) -> torch.Tensor:
        return torch.empty((0,), dtype=torch.int64, device=device)

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def match_prefix(self, key: List[int], **kwargs) -> MatchResult:
        pass

    @abstractmethod
    def cache_finished_req(self, req: Req, **kwargs):
        pass

    @abstractmethod
    def cache_unfinished_req(self, req: Req, **kwargs):
        pass

    @abstractmethod
    def evict(self, num_tokens: int):
        pass

    @abstractmethod
    def inc_lock_ref(self, node: Any):
        pass

    @abstractmethod
    def dec_lock_ref(self, node: Any):
        pass

    def evictable_size(self):
        return 0

    def protected_size(self):
        return 0

    def total_size(self):
        raise NotImplementedError()

    def pretty_print(self):
        raise NotImplementedError()

    def take_events(self):
        return []
