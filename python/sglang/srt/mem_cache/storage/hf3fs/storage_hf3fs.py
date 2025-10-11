import atexit
import concurrent.futures
import json
import logging
import os
import signal
import threading
import time
from abc import ABC, abstractmethod
from functools import wraps
from typing import Any, List, Optional, Tuple

import torch

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorage,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
)
from sglang.srt.mem_cache.memory_pool_host import HostKVCache
from sglang.srt.mem_cache.storage.hf3fs.hf3fs_client import Hf3fsClient
from sglang.srt.metrics.collector import StorageMetrics

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


def create_hf3fs_client(
    path: str, size: int, bytes_per_page: int, entries: int, use_mock: bool = False
) -> Hf3fsClient:
    """Factory function to create appropriate HF3FS client.

    Args:
        path: File path for storage
        size: Total size of storage file
        bytes_per_page: Bytes per page
        entries: Number of entries for batch operations
        use_mock: Whether to use mock client instead of real usrbio client

    Returns:
    """
    if use_mock:
        from sglang.srt.mem_cache.storage.hf3fs.hf3fs_client import Hf3fsMockClient

        logger.info(f"[Rank Using Hf3fsMockClient for testing")
        return Hf3fsMockClient(path, size, bytes_per_page, entries)
    else:
        from sglang.srt.mem_cache.storage.hf3fs.hf3fs_usrbio_client import (
            Hf3fsUsrBioClient,
        )

        return Hf3fsUsrBioClient(path, size, bytes_per_page, entries)


class HiCacheHF3FS(HiCacheStorage):
    """HiCache backend that stores KV cache pages in HF3FS files."""

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
        is_mla_model: bool = False,
        is_page_first_layout: bool = False,
        use_mock_client: bool = False,
    ):
        self.rank = rank
        self.file_path = file_path
        self.file_size = file_size
        self.numjobs = numjobs
        self.bytes_per_page = bytes_per_page
        self.gb_per_page = bytes_per_page / (1 << 30)
        self.entries = entries
        self.dtype = dtype
        self.metadata_client = metadata_client
        self.is_mla_model = is_mla_model
        self.is_page_first_layout = is_page_first_layout
        self.numel = self.bytes_per_page // self.dtype.itemsize
        self.num_pages = self.file_size // self.bytes_per_page
        self.skip_backup = False
        if self.is_mla_model and self.rank != 0:
            self.skip_backup = True
            self.rank = 0

        self.is_zero_copy = False

        logger.info(
            f"[Rank {self.rank}] HiCacheHF3FS Client Initializing: "
            f"file_path={self.file_path}, "
            f"file_size={self.file_size / (2 ** 30):.2f} GB, "
            f"num_pages={self.num_pages}, "
            f"is_mla_model={self.is_mla_model}"
        )

        self.ac = AtomicCounter(self.numjobs)
        self.clients = [
            create_hf3fs_client(
                self.file_path,
                self.file_size,
                self.bytes_per_page,
                self.entries,
                use_mock_client,
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

        self.prefetch_pgs = []
        self.backup_pgs = []
        self.prefetch_bandwidth = []
        self.backup_bandwidth = []

    @staticmethod
    def from_env_config(
        bytes_per_page: int,
        dtype: torch.dtype,
        storage_config: HiCacheStorageConfig = None,
    ) -> "HiCacheHF3FS":
        """Create a HiCacheHF3FS instance from environment configuration.

        Environment:
            - Uses env var stored in `HiCacheHF3FS.default_env_var` to locate a JSON config.
            - Falls back to a local single-machine config when the env var is not set.

        Raises:
            ValueError: If MLA Model is requested without global metadata server or required keys are missing.
        """
        from sglang.srt.mem_cache.storage.hf3fs.mini_3fs_metadata_server import (
            Hf3fsGlobalMetadataClient,
            Hf3fsLocalMetadataClient,
        )

        use_mock_client = False
        if storage_config is not None:
            rank, is_mla_model, is_page_first_layout = (
                storage_config.tp_rank,
                storage_config.is_mla_model,
                storage_config.is_page_first_layout,
            )

            if storage_config.extra_config is not None:
                use_mock_client = storage_config.extra_config.get(
                    "use_mock_hf3fs_client", False
                )
        else:
            rank, is_mla_model, is_page_first_layout = (
                0,
                False,
                False,
            )

        mla_unsupported_msg = f"MLA model is not supported without global metadata server, please refer to https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/storage/hf3fs/docs/deploy_sglang_3fs_multinode.md"

        config_path = os.getenv(HiCacheHF3FS.default_env_var)
        if not config_path:
            if is_mla_model:
                raise ValueError(mla_unsupported_msg)

            return HiCacheHF3FS(
                rank=rank,
                file_path=f"/data/hicache.{rank}.bin",
                file_size=1 << 40,
                numjobs=16,
                bytes_per_page=bytes_per_page,
                entries=8,
                dtype=dtype,
                metadata_client=Hf3fsLocalMetadataClient(),
                is_page_first_layout=is_page_first_layout,
                use_mock_client=use_mock_client,
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
        if config.get("metadata_server_url"):
            # Use global metadata client to connect to metadata server
            metadata_server_url = config["metadata_server_url"]
            metadata_client = Hf3fsGlobalMetadataClient(metadata_server_url)

            logger.info(
                f"Using global metadata client with server url: {metadata_server_url}"
            )
        else:
            # Enable MLA optimization only when using the global metadata client
            if is_mla_model:
                raise ValueError(mla_unsupported_msg)

            # Use local metadata client for single-machine deployment
            metadata_client = Hf3fsLocalMetadataClient()

        rank_for_path = 0 if is_mla_model else rank
        return HiCacheHF3FS(
            rank=rank,
            # Let all ranks use the same file path for MLA model
            file_path=f"{config['file_path_prefix']}.{rank_for_path}.bin",
            file_size=int(config["file_size"]),
            numjobs=int(config["numjobs"]),
            bytes_per_page=bytes_per_page,
            entries=int(config["entries"]),
            dtype=dtype,
            metadata_client=metadata_client,
            is_mla_model=is_mla_model,
            is_page_first_layout=is_page_first_layout,
            use_mock_client=use_mock_client,
        )

    @synchronized()
    def _batch_get(
        self,
        keys: List[str],
        values: List[torch.Tensor],
    ) -> List[bool]:
        page_indices = self.metadata_client.get_page_indices(self.rank, keys)

        batch_indices, file_offsets = [], []
        for i, page_index in enumerate(page_indices):
            if page_index is not None:
                batch_indices.append(i)
                file_offsets.append(page_index * self.bytes_per_page)

        for target_location in values:
            assert target_location.is_contiguous()
        file_results = values

        start_time = time.perf_counter()

        futures = [
            self.executor.submit(
                self.clients[self.ac.next()].batch_read,
                file_offsets[i : i + self.entries],
                file_results[i : i + self.entries],
            )
            for i in range(0, len(batch_indices), self.entries)
        ]
        read_results = [result for future in futures for result in future.result()]

        end_time = time.perf_counter()
        ionum = len(batch_indices)
        self.prefetch_pgs.append(ionum)
        self.prefetch_bandwidth.append(
            ionum / (end_time - start_time) * self.gb_per_page
        )

        results = [False] * len(keys)
        for batch_index, read_result in zip(batch_indices, read_results):
            if read_result == self.bytes_per_page:
                results[batch_index] = True
            else:
                logger.error(
                    f"[Rank {self.rank}] HiCacheHF3FS get {keys[batch_index]} failed"
                )

        return results

    @synchronized()
    def _batch_set(
        self,
        keys: List[str],
        values: Optional[Any] = None,
    ) -> List[bool]:
        # In MLA backend, only one rank needs to backup the KV cache
        if self.skip_backup:
            return True

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

        start_time = time.perf_counter()

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

        end_time = time.perf_counter()
        ionum = len(batch_indices)
        self.backup_pgs.append(ionum)
        self.backup_bandwidth.append(ionum / (end_time - start_time) * self.gb_per_page)

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

        return results

    def delete(self, key: str) -> None:
        self.metadata_client.delete_keys(self.rank, [key])

    def exists(self, key: str) -> bool:
        result = self.metadata_client.exists(self.rank, [key])
        return result[0] if result else False

    def batch_exists(
        self, keys: List[str], extra_info: Optional[HiCacheStorageExtraInfo] = None
    ) -> int:
        factor = 1
        if self.is_zero_copy and not self.is_mla_model:
            keys = self._get_mha_zero_copy_keys(keys)
            factor = 2

        results = self.metadata_client.exists(self.rank, keys)

        i = 0
        while i < len(keys) and results[i]:
            i += 1

        return i // factor

    def clear(self) -> None:
        try:
            self.metadata_client.clear(self.rank)
            logger.info(f"Cleared HiCacheHF3FS for rank {self.rank}")
        except Exception as e:
            logger.error(f"Failed to clear HiCacheHF3FS: {e}")

    def close(self) -> None:
        try:
            for c in self.clients:
                c.close()
            self.executor.shutdown(wait=True)
        except Exception as e:
            logger.error(f"close HiCacheHF3FS: {e}")
        logger.info("close HiCacheHF3FS")

    @synchronized()
    def get_stats(self):
        storage_metrics = StorageMetrics()
        storage_metrics.prefetch_pgs.extend(self.prefetch_pgs)
        storage_metrics.backup_pgs.extend(self.backup_pgs)
        storage_metrics.prefetch_bandwidth.extend(self.prefetch_bandwidth)
        storage_metrics.backup_bandwidth.extend(self.backup_bandwidth)
        self.prefetch_pgs.clear()
        self.backup_pgs.clear()
        self.prefetch_bandwidth.clear()
        self.backup_bandwidth.clear()
        return storage_metrics

    def register_mem_pool_host(self, mem_pool_host: HostKVCache):
        super().register_mem_pool_host(mem_pool_host)
        self.is_zero_copy = self.mem_pool_host.layout in [
            "page_first",
            "page_first_direct",
        ]

        logger.info(f"{self.is_zero_copy=}, layout={self.mem_pool_host.layout}")

    def _get_mha_zero_copy_keys(self, keys: List[str]) -> List[str]:
        _keys = []
        for k in keys:
            _keys.append(f"{k}-k")
            _keys.append(f"{k}-v")
        return _keys

    def _get_mha_zero_copy_values(
        self, values: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        _values = []
        for value in values:
            _values.append(value[0])
            _values.append(value[1])
        return _values

    def _batch_get_preprocess(self, keys, host_indices):
        page_num = len(host_indices) // self.mem_pool_host.page_size
        # host_indices to kv_buffer
        flat = not self.is_zero_copy
        values = (
            [
                self.mem_pool_host.get_data_page(
                    host_indices[i * self.mem_pool_host.page_size], flat=flat
                )
                for i in range(page_num)
            ]
            if self.is_zero_copy
            else [
                self.mem_pool_host.get_dummy_flat_data_page() for _ in range(page_num)
            ]
        )

        if self.is_zero_copy and not self.is_mla_model:
            keys = self._get_mha_zero_copy_keys(keys)
            values = self._get_mha_zero_copy_values(values)

        return keys, values

    def _batch_get_postprocess(self, host_indices, values, results):
        page_num = len(host_indices) // self.mem_pool_host.page_size

        if self.is_zero_copy:
            if not self.is_mla_model:
                results = [
                    (results[2 * i] and results[2 * i + 1]) for i in range(page_num)
                ]
                results = results[:page_num]
            return results

        for i in range(page_num):
            if not results[i]:
                break
            self.mem_pool_host.set_from_flat_data_page(
                host_indices[i * self.mem_pool_host.page_size], values[i]
            )

        return results

    def batch_get_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        keys, values = self._batch_get_preprocess(keys, host_indices)
        results = self._batch_get(keys, values)
        return self._batch_get_postprocess(host_indices, values, results)

    def _batch_set_preprocess(self, keys, host_indices):
        page_num = len(host_indices) // self.mem_pool_host.page_size
        # host_indices to kv_buffer
        flat = not self.is_zero_copy
        values = [
            self.mem_pool_host.get_data_page(
                host_indices[i * self.mem_pool_host.page_size], flat=flat
            )
            for i in range(page_num)
        ]

        if self.is_zero_copy and not self.is_mla_model:
            keys = self._get_mha_zero_copy_keys(keys)
            values = self._get_mha_zero_copy_values(values)

        return keys, values

    def batch_set_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        len_keys = len(keys)
        keys, values = self._batch_set_preprocess(keys, host_indices)
        results = self._batch_set(keys, values)
        return results

    # Deprecated
    def get(
        self,
        key: str,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> torch.Tensor | None:
        pass

    # Deprecated
    def batch_get(
        self,
        keys: List[str],
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> List[torch.Tensor | None] | int:
        pass

    # Deprecated
    def set(
        self,
        key: str,
        value: Optional[Any] = None,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        pass

    # Deprecated
    def batch_set(
        self,
        keys: List[str],
        values: Optional[Any] = None,
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        pass
