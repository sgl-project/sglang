import concurrent.futures
import logging
import threading
from collections import OrderedDict
from functools import wraps
from typing import List, Optional

import torch

from sglang.srt.mem_cache.hicache_storage import HiCacheStorage
from sglang.srt.mem_cache.storage.hf3fs.client_hf3fs import Hf3fsClient

logger = logging.getLogger(__name__)


class AtomicCounter:
    def __init__(self, n: int):
        assert n > 0
        self.n = n
        self._value = 0
        self._lock = threading.Lock()

    def next(self) -> int:
        with self._lock:
            current = self._value
            self._value = (current + 1) % self.n
            return current


def synchronized():
    def _decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            with self.lock:
                return func(self, *args, **kwargs)

        return wrapper

    return _decorator


class HiCacheHF3FS(HiCacheStorage):
    def __init__(
        self,
        file_path: str,
        file_size: int,
        numjobs: int,
        bytes_per_page: int,
        entries: int,
        dtype: torch.dtype,
    ):
        self.file_path = file_path
        self.file_size = file_size
        self.numjobs = numjobs
        self.bytes_per_page = bytes_per_page
        self.entries = entries
        self.dtype = dtype

        self.numel = self.bytes_per_page // self.dtype.itemsize

        self.num_pages = self.file_size // self.bytes_per_page

        logger.info(
            f"file_path = {self.file_path}, "
            f"file_size = {self.file_size/(2**30):.2f} GB, "
            f"numjobs = {self.numjobs}, "
            f"bytes_per_page = {self.bytes_per_page/(2**20):.2f} MB, "
            f"entries = {self.entries}, "
            f"num_pages = {self.num_pages}"
        )

        self.ac = AtomicCounter(self.numjobs)
        self.clients = [
            Hf3fsClient(
                self.file_path, self.file_size, self.bytes_per_page, self.entries
            )
            for _ in range(numjobs)
        ]
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.numjobs, thread_name_prefix="HiCacheHF3FS"
        )

        # Implemented a preliminary single-file page_hash -> file_offset index as interim storage.
        # Future iterations may adopt a global KVCache manager to coordinate external cache instances
        # through centralized metadata orchestration.
        self.lock = threading.RLock()
        self.free_pages = list(range(self.num_pages))
        self.key_to_index = OrderedDict()

    def get(
        self, key: str, target_location: Optional[torch.Tensor] = None
    ) -> torch.Tensor | None:
        return self.batch_get([key], target_location)[0]

    @synchronized()
    def batch_get(
        self,
        keys: List[str],
        target_locations: Optional[List[torch.Tensor]] = None,
    ) -> List[torch.Tensor | None]:
        batch_indices, file_offsets = [], []
        for i, key in enumerate(keys):
            if key not in self.key_to_index:
                continue
            batch_indices.append(i)
            file_offsets.append(self.key_to_index[key] * self.bytes_per_page)
            self.key_to_index.move_to_end(key)
        # TODO: target_locations
        file_results = [
            torch.empty(self.numel, dtype=self.dtype) for _ in range(len(batch_indices))
        ]

        futures = [
            self.executor.submit(
                self.clients[self.ac.next()].batch_read,
                file_offsets[i : i + self.entries],
                file_results[i : i + self.entries],
            )
            for i in range(0, len(batch_indices), self.entries)
        ]
        read_results = [result for future in futures for result in future.result()]

        results = [None] * len(keys)
        for batch_index, file_result, read_result in zip(
            batch_indices, file_results, read_results
        ):
            if read_result == self.bytes_per_page:
                results[batch_index] = file_result

        return results

    def set(self, key: str, value: torch.Tensor) -> bool:
        return self.batch_set([key], [value])[0]

    def batch_set(self, keys: List[str], values: List[torch.Tensor]) -> bool:
        indices = self.get_batch_set_indices(keys)
        batch_indices, file_offsets, file_values = [], [], []
        for i, (value, (is_written, index)) in enumerate(zip(values, indices)):
            if is_written or index == -1:
                continue
            batch_indices.append(i)
            file_offsets.append(index * self.bytes_per_page)
            file_values.append(value.contiguous())

        futures = [
            self.executor.submit(
                self.clients[self.ac.next()].batch_write,
                file_offsets[i : i + self.entries],
                file_values[i : i + self.entries],
            )
            for i in range(0, len(batch_indices), self.entries)
        ]
        write_results = [
            result == self.bytes_per_page
            for future in futures
            for result in future.result()
        ]

        results = [index[0] for index in indices]
        for batch_index, write_result in zip(batch_indices, write_results):
            key = keys[batch_index]
            index = indices[batch_index][1]
            if write_result:
                self.key_to_index[key] = index
                self.key_to_index.move_to_end(key)
            else:
                self.free_pages.append(index)
            results[batch_index] = write_result
        return results

    @synchronized()
    def get_batch_set_indices(self, keys: List[str]) -> list:
        ionum = len(keys)
        # results: tuples of (is_written: bool, page_idx: int)
        # - is_written: True = hit (no I/O), False = write (miss)
        # - page_idx: page storing data
        results = [None] * min(ionum, self.num_pages)
        if ionum > self.num_pages:
            results.extend([(False, -1)] * (ionum - self.num_pages))

        new_keys = []
        for batch_index, key in enumerate(keys[: self.num_pages]):
            if key in self.key_to_index:
                results[batch_index] = (True, self.key_to_index[key])
                self.key_to_index.move_to_end(key)
            else:
                new_keys.append((batch_index, key))

        for batch_index, _ in new_keys:
            index = (
                self.free_pages.pop()
                if len(self.free_pages) > 0
                else self.key_to_index.popitem(last=False)[1]
            )
            results[batch_index] = (False, index)

        return results

    @synchronized()
    def delete(self, key: str) -> None:
        if key not in self.key_to_index:
            return
        index = self.key_to_index.pop(key)
        self.free_pages.append(index)

    @synchronized()
    def exists(self, key: str) -> bool:
        return key in self.key_to_index

    @synchronized()
    def clear(self) -> None:
        self.free_pages = list(range(self.num_pages))
        self.key_to_index.clear()

    def close(self) -> None:
        for c in self.clients:
            c.close()
        self.executor.shutdown(wait=True)
