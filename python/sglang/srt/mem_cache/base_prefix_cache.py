from abc import ABC, abstractmethod
from typing import Any, List, Tuple


class BasePrefixCache(ABC):
    """Cache can be indexed by either rid or key."""

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def match_prefix(self, **kwargs) -> Tuple[List[int], int]:
        pass

    @abstractmethod
    def insert(self, **kwargs):
        pass

    @abstractmethod
    def cache_finished_req(self, **kwargs):
        pass

    @abstractmethod
    def cache_unfinished_req(self, **kwargs):
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
