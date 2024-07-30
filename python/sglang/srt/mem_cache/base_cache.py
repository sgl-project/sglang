from abc import ABC, abstractmethod


class BaseCache(ABC):
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def match_prefix(self, key):
        pass

    @abstractmethod
    def insert(self, key, value=None):
        pass

    @abstractmethod
    def cache_req(
        self,
        token_ids,
        last_uncached_pos,
        req_pool_idx,
        del_in_memory_pool=True,
        old_last_node=None,
    ):
        pass

    @abstractmethod
    def total_size(self):
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

    def pretty_print(self):
        raise NotImplementedError
