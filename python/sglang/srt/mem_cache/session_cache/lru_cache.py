from enum import IntEnum
from typing import Callable, List, Optional

import torch


class LRUSessionCacheStatus(IntEnum):
    UNFINISHED = 0
    FINISHED = 1
    LOADING = 2
    LOADED = 3
    WRITING = 4
    WRITTEN = 5


class LRUSessionCacheEntry:
    def __init__(
        self,
        sid: str,
        kv_indices: torch.Tensor,
        status: LRUSessionCacheStatus = LRUSessionCacheStatus.UNFINISHED,
    ):
        self.sid = sid
        self.kv_indices = kv_indices
        self.status = status
        self.lock_ref = 0
        self.lock_size = 0
        self.length = len(self.kv_indices)
        self.prev = None
        self.next = None

    def set_kv_indices(self, kv_indices):
        self.kv_indices = kv_indices
        self.length = len(self.kv_indices)

    def get_length(self):
        return self.length

    def get_lock_ref(self):
        return self.lock_ref

    def inc_lock_ref(self):
        self.lock_ref += 1

    def dec_lock_ref(self):
        self.lock_ref -= 1

    def get_kv_indices(self):
        return self.kv_indices

    def get_status(self):
        return self.status

    def set_status(self, status: LRUSessionCacheStatus):
        self.status = status

    def is_unfinished(self):
        return self.status == LRUSessionCacheStatus.UNFINISHED

    def is_finished(self):
        return self.status == LRUSessionCacheStatus.FINISHED

    def is_loading(self):
        return self.status == LRUSessionCacheStatus.LOADING

    def is_loaded(self):
        return self.status == LRUSessionCacheStatus.LOADED

    def is_writing(self):
        return self.status == LRUSessionCacheStatus.WRITING

    def is_written(self):
        return self.status == LRUSessionCacheStatus.WRITTEN


class LRUSessionCache:
    def __init__(self):
        self.reset()

    def reset(self):
        self.head = LRUSessionCacheEntry("head", [])
        self.tail = LRUSessionCacheEntry("tail", [])
        self.head.next = self.tail
        self.tail.prev = self.head
        self.cache = {}  # sid -> LRUSessionCacheEntry

    def _move_to_front(self, entry: LRUSessionCacheEntry):
        entry.prev.next = entry.next
        entry.next.prev = entry.prev

        entry.prev = self.head
        entry.next = self.head.next
        self.head.next.prev = entry
        self.head.next = entry

    def get(self, sid: str) -> Optional[LRUSessionCacheEntry]:
        if sid in self.cache:
            entry = self.cache[sid]
            self._move_to_front(entry)
            return entry
        return None

    def set(
        self,
        sid: str,
        kv_indices: torch.Tensor,
        status: LRUSessionCacheStatus = LRUSessionCacheStatus.UNFINISHED,
    ):
        if sid in self.cache:
            tem_entry = self.cache[sid]
            tem_entry.set_kv_indices(kv_indices)
            tem_entry.set_status = status
            self._move_to_front(tem_entry)
        else:
            entry = LRUSessionCacheEntry(sid, kv_indices, status)
            self.cache[sid] = entry
            entry.prev = self.head
            entry.next = self.head.next
            self.head.next.prev = entry
            self.head.next = entry

    def delete(self, sid: str):
        if sid not in self.cache:
            return None
        entry = self.cache[sid]
        if entry.prev:
            entry.prev.next = entry.next
        if entry.next:
            entry.next.prev = entry.prev
        del self.cache[sid]
        return entry

    def evict(self):
        if self.tail.prev == self.head:
            return
        entry = self.tail.prev
        self.delete(entry.sid)

    def evict_by_cond(self, condition: Callable[[LRUSessionCacheEntry], bool]):
        current = self.tail.prev
        while current != self.head:
            if condition(current):
                return self.delete(current.sid)
            current = current.prev
        return None

    def exist(self, sid: str) -> bool:
        return sid in self.cache
