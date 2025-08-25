import atexit
import concurrent.futures
import json
import logging
import os
import signal
import threading
from abc import ABC, abstractmethod
from functools import wraps
from typing import Any, List, Optional, Tuple

import torch

from sglang.srt.distributed import get_tensor_model_parallel_rank
from sglang.srt.layers.dp_attention import (
    get_attention_tp_rank,
    is_dp_attention_enabled,
)
from sglang.srt.mem_cache.hicache_storage import HiCacheStorage
from sglang.srt.mem_cache.storage.hf3fs.client_hf3fs import Hf3fsClient

logger = logging.getLogger(__name__)


class Hf3fsMetadataInterface(ABC):
    """Interface for HF3FS metadata operations."""

    @abstractmethod
    def initialize(self, rank: int, num_pages: int) -> None:
        """Initialize the metadata service with specified number of pages."""
        pass

    @abstractmethod
    def reserve_and_allocate_page_indices(
        self,
        rank: int,
        keys: List[Tuple[str, str]],
    ) -> List[Tuple[bool, int]]:
        """
        Reserve and allocate page indices for the specified keys.
        Args:
            rank: The rank of the process.
            keys: The keys to reserve and allocate page indices for. Each tuple contains a key and the key of its prefix block.
        Returns:
            List[Tuple[bool, int]]: A list of tuples, where each tuple contains a boolean indicating whether the key has existed and an integer indicating the allocated page index.
        """
        pass

    @abstractmethod
    def confirm_write(
        self,
        rank: int,
        written_keys_to_confirm: List[Tuple[str, int]],
        pages_to_release: List[int],
    ) -> None:
        """
        Confirm that key-value pairs have been successfully written to storage.
        Args:
            rank: The rank of the process.
            written_keys_to_confirm: A list of tuples, where each tuple contains a key and its corresponding page index.
            pages_to_release: A list of page indices to be released.
        """
        pass

    @abstractmethod
    def get_page_indices(self, rank: int, keys: List[str]) -> List[Optional[int]]:
        """
        Get page indices for the specified keys.
        Args:
            rank: The rank of the process.
            keys: A list of keys.
        Returns:
            List[Optional[int]]: A list of integers representing the page indices for the specified keys.
                                 If a key is not found, the corresponding index will be None.
        """
        pass

    @abstractmethod
    def delete_keys(self, rank: int, keys: List[str]) -> None:
        """Delete specified keys and their associated pages."""
        pass

    @abstractmethod
    def exists(self, rank: int, keys: List[str]) -> List[bool]:
        """Check if the specified keys exist."""
        pass

    @abstractmethod
    def clear(self, rank: int) -> None:
        """Clear all key-value pairs and page allocations for the specified rank."""
        pass


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
    default_env_var: str = "SGLANG_HICACHE_HF3FS_CONFIG_PATH"

    def __init__(
        self,
        rank: int,
        file_path: str,
        file_size: int,
        numjobs: int,
        bytes_per_page: int,
        entries: int,
        dtype: torch.dtype,
        metadata_client: Hf3fsMetadataInterface,
    ):
        self.rank = rank
        self.file_path = file_path
        self.file_size = file_size
        self.numjobs = numjobs
        self.bytes_per_page = bytes_per_page
        self.entries = entries
        self.dtype = dtype
        self.metadata_client = metadata_client

        self.numel = self.bytes_per_page // self.dtype.itemsize
        self.num_pages = self.file_size // self.bytes_per_page

        logger.info(
            f"[Rank {self.rank}] HiCacheHF3FS Client Initializing: "
            f"file_path={self.file_path}, "
            f"file_size={self.file_size / (2 ** 30):.2f} GB, "
            f"num_pages={self.num_pages}"
        )

        self.ac = AtomicCounter(self.numjobs)
        self.clients = [
            Hf3fsClient(
                self.file_path, self.file_size, self.bytes_per_page, self.entries
            )
            for _ in range(numjobs)
        ]
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.numjobs, thread_name_prefix=f"HiCacheHF3FS-Rank{self.rank}"
        )

        self.metadata_client.initialize(self.rank, self.num_pages)
        self.lock = threading.RLock()

        atexit.register(self.close)

        signal.signal(signal.SIGINT, lambda sig, frame: self.close())
        signal.signal(signal.SIGTERM, lambda sig, frame: self.close())
        signal.signal(signal.SIGQUIT, lambda sig, frame: self.close())

    @staticmethod
    def from_env_config(
        bytes_per_page: int, dtype: torch.dtype, rank: int = None
    ) -> "HiCacheHF3FS":
        from sglang.srt.mem_cache.storage.hf3fs.mini_3fs_metadata_server import (
            Hf3fsGlobalMetadataClient,
            Hf3fsLocalMetadataClient,
        )

        if rank is None:
            rank = (
                get_attention_tp_rank()
                if is_dp_attention_enabled()
                else get_tensor_model_parallel_rank()
            )

        config_path = os.getenv(HiCacheHF3FS.default_env_var)
        if not config_path:
            return HiCacheHF3FS(
                rank=rank,
                file_path=f"/data/hicache.{rank}.bin",
                file_size=1 << 40,
                numjobs=16,
                bytes_per_page=bytes_per_page,
                entries=8,
                dtype=dtype,
                metadata_client=Hf3fsLocalMetadataClient(),
            )

        try:
            with open(config_path, "r") as f:
                config = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load config from {config_path}: {str(e)}")

        # Check required keys (metadata_server_url is now optional)
        required_keys = {
            "file_path_prefix",
            "file_size",
            "numjobs",
            "entries",
        }
        missing_keys = required_keys - set(config.keys())
        if missing_keys:
            raise ValueError(f"Missing required keys in config: {missing_keys}")

        # Choose metadata client based on configuration
        if "metadata_server_url" in config and config["metadata_server_url"]:
            # Use global metadata client to connect to metadata server
            metadata_server_url = config["metadata_server_url"]
            metadata_client = Hf3fsGlobalMetadataClient(metadata_server_url)
            logger.info(
                f"Using global metadata client with server url: {metadata_server_url}"
            )
        else:
            # Use local metadata client for single-machine deployment
            metadata_client = Hf3fsLocalMetadataClient()

        return HiCacheHF3FS(
            rank=rank,
            file_path=f"{config['file_path_prefix']}.{rank}.bin",
            file_size=int(config["file_size"]),
            numjobs=int(config["numjobs"]),
            bytes_per_page=bytes_per_page,
            entries=int(config["entries"]),
            dtype=dtype,
            metadata_client=metadata_client,
        )

    def get(
        self,
        key: str,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> torch.Tensor | None:
        return self.batch_get(
            [key],
            [target_location] if target_location is not None else None,
            [target_sizes] if target_sizes is not None else None,
        )[0]

    @synchronized()
    def batch_get(
        self,
        keys: List[str],
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> List[torch.Tensor | None]:
        page_indices = self.metadata_client.get_page_indices(self.rank, keys)

        batch_indices, file_offsets = [], []
        for i, page_index in enumerate(page_indices):
            if page_index is not None:
                batch_indices.append(i)
                file_offsets.append(page_index * self.bytes_per_page)

        if target_locations is not None:
            for target_location in target_locations:
                assert target_location.is_contiguous()
            file_results = target_locations
        else:
            file_results = [
                torch.empty(self.numel, dtype=self.dtype)
                for _ in range(len(batch_indices))
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
            else:
                logger.error(
                    f"[Rank {self.rank}] HiCacheHF3FS get {keys[batch_index]} failed"
                )

        return results

    def set(
        self,
        key: str,
        value: Optional[Any] = None,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        return self.batch_set(
            [key],
            [value] if value is not None else None,
            [target_location] if target_location is not None else None,
            [target_sizes] if target_sizes is not None else None,
        )

    def batch_set(
        self,
        keys: List[str],
        values: Optional[Any] = None,
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        # Todo: Add prefix block's hash key
        key_with_prefix = [(key, "") for key in keys]
        indices = self.metadata_client.reserve_and_allocate_page_indices(
            self.rank, key_with_prefix
        )

        batch_indices, file_offsets, file_values = [], [], []
        pages_to_release = []

        for i, (value, (is_written, page_index)) in enumerate(zip(values, indices)):
            if is_written or page_index == -1:
                continue

            batch_indices.append(i)
            file_offsets.append(page_index * self.bytes_per_page)
            assert value.is_contiguous()
            file_values.append(value)

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

        written_keys_to_confirm = []
        results = [index[0] for index in indices]
        for batch_index, write_result in zip(batch_indices, write_results):
            key = keys[batch_index]
            page_index = indices[batch_index][1]
            if write_result:
                written_keys_to_confirm.append((key, page_index))
            else:
                logger.error(f"[Rank {self.rank}] HiCacheHF3FS set {key} failed")
                pages_to_release.append(page_index)
            results[batch_index] = write_result

        if len(written_keys_to_confirm) > 0 or len(pages_to_release) > 0:
            self.metadata_client.confirm_write(
                self.rank, written_keys_to_confirm, pages_to_release
            )

        return all(results)

    @synchronized()
    def delete(self, key: str) -> None:
        self.metadata_client.delete_keys(self.rank, [key])

    @synchronized()
    def exists(self, key: str) -> bool:
        result = self.metadata_client.exists(self.rank, [key])
        return result[0] if result else False

    @synchronized()
    def clear(self) -> None:
        self.metadata_client.clear(self.rank)

    def close(self) -> None:
        try:
            for c in self.clients:
                c.close()
            self.executor.shutdown(wait=True)
        except Exception as e:
            logger.error(f"close HiCacheHF3FS: {e}")
        logger.info("close HiCacheHF3FS")
