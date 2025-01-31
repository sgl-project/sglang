from abc import ABC, abstractmethod
from typing import Callable, List, Tuple


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
    def evict(self, num_tokens: int, evict_callback: Callable):
        pass

    @abstractmethod
    def inc_lock_ref(self, node):
        pass

    @abstractmethod
    def dec_lock_ref(self, node):
        pass

    @abstractmethod
    def evictable_size(self):
        pass

    @abstractmethod
    def protected_size(self):
        raise NotImplementedError()

    def total_size(self):
        raise NotImplementedError()

    def pretty_print(self):
        raise NotImplementedError()
