from abc import ABC, abstractmethod


class BasePrefixCache(ABC):
    """Cache can be indexed by either rid or key."""

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def match_prefix(self, **kwargs):
        pass

    @abstractmethod
    def insert(self, **kwargs):
        pass

    @abstractmethod
    def cache_req(self, **kwargs):
        pass

    @abstractmethod
    def evict(self, num_tokens, evict_callback):
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

    def total_size(self):
        raise NotImplementedError

    def pretty_print(self):
        raise NotImplementedError
