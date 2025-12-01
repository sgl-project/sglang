import ctypes
import hashlib
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional

import torch

from sglang.srt.mem_cache.memory_pool_host import HostKVCache

logger = logging.getLogger(__name__)


def get_hash_str(token_ids: List[int], prior_hash: str = None) -> str:
    hasher = hashlib.sha256()

    if prior_hash:
        hasher.update(bytes.fromhex(prior_hash))

    for t in token_ids:
        if isinstance(t, tuple):
            # EAGLE bigram mode: hash both elements to uniquely identify the bigram
            for elem in t:
                hasher.update(elem.to_bytes(4, byteorder="little", signed=False))
        else:
            # Regular mode: single integer token
            hasher.update(t.to_bytes(4, byteorder="little", signed=False))

    return hasher.hexdigest()


def hash_str_to_int64(hash_str: str) -> int:
    """Convert SHA256 hex string to signed 64-bit integer for events.

    Takes first 16 hex characters (64 bits) and converts to signed int64 range.
    """
    # Take first 16 hex chars to get 64-bit value
    uint64_val = int(hash_str[:16], 16)
    # Convert to signed int64 range [-2^63, 2^63-1]
    if uint64_val >= 2**63:
        return uint64_val - 2**64
    return uint64_val


@dataclass
class HiCacheStorageConfig:
    tp_rank: int
    tp_size: int
    pp_rank: int
    pp_size: int
    is_mla_model: bool
    is_page_first_layout: bool
    model_name: Optional[str]
    extra_config: Optional[dict] = None


@dataclass
class HiCacheStorageExtraInfo:
    prefix_keys: Optional[List[str]] = (None,)
    extra_info: Optional[dict] = None


class HiCacheStorage(ABC):
    """
    HiCacheStorage is a class that provides a generic key-value interface for storing and retrieving KV cache.
    It abstracts the underlying storage mechanism, allowing different implementations to be used.
    """

    # todo, the page size of storage backend does not have to be the same as the same as host memory pool

    def register_mem_pool_host(self, mem_pool_host: HostKVCache):
        self.mem_pool_host = mem_pool_host

    def batch_get_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        """
        Retrieve values for multiple keys.
        Returns a list of booleans indicating success for each key.
        """
        pass

    def batch_set_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        """
        Store multiple key-value pairs.
        Returns a list of booleans indicating success for each key.
        """
        pass

    @abstractmethod
    def get(
        self,
        key: str,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> torch.Tensor | None:
        """
        Retrieve the value associated with the given key.
        Returns None if the key does not exist.
        """
        pass

    # TODO: Deprecate
    @abstractmethod
    def batch_get(
        self,
        keys: List[str],
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> List[torch.Tensor | None] | int:
        """
        Retrieve values for multiple keys.
        Returns a list of tensors or None for each key.
        """
        pass

    @abstractmethod
    def set(
        self,
        key: str,
        value: Optional[Any] = None,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        """
        Store the value associated with the given key.
        Returns True if the operation was successful, False otherwise.
        """
        pass

    # TODO: Deprecate
    @abstractmethod
    def batch_set(
        self,
        keys: List[str],
        values: Optional[Any] = None,
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        """
        Store multiple key-value pairs.
        Returns True if all operations were successful, False otherwise.
        """
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """
        Check if the key exists in the storage.
        Returns True if the key exists, False otherwise.
        """
        pass

    # TODO: Use a finer-grained return type (e.g., List[bool])
    def batch_exists(
        self, keys: List[str], extra_info: Optional[HiCacheStorageExtraInfo] = None
    ) -> int:
        """
        Check if the keys exist in the storage.
        return the number of consecutive existing keys from the start.
        Can be overridden by subclasses for more efficient implementation.
        """
        for i in range(len(keys)):
            if not self.exists(keys[i]):
                return i
        return len(keys)

    def clear(self) -> None:
        pass

    def get_stats(self):
        return None


class HiCacheFile(HiCacheStorage):

    def __init__(
        self, storage_config: HiCacheStorageConfig, file_path: str = "/tmp/hicache"
    ):
        self.file_path = os.getenv("SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR", file_path)

        tp_rank, tp_size, model_name, is_mla_model = (
            storage_config.tp_rank,
            storage_config.tp_size,
            storage_config.model_name,
            storage_config.is_mla_model,
        )
        model_name = "-".join(model_name.split("/")) if model_name else ""
        if is_mla_model:
            self.config_suffix = f"_{model_name}"
        else:
            self.config_suffix = f"_{model_name}_{tp_rank}_{tp_size}"

        if not os.path.exists(self.file_path) and tp_rank == 0:
            os.makedirs(self.file_path)
            logger.info(f"Created HiCacheFile storage directory at {self.file_path}")

        # Will be set in register_mem_pool_host
        self.is_mla_backend = is_mla_model

    def _get_suffixed_key(self, key: str) -> str:
        return key + self.config_suffix

    def register_mem_pool_host(self, mem_pool_host: HostKVCache):
        super().register_mem_pool_host(mem_pool_host)
        # Store layout for use in preprocessing
        self.mem_pool_layout = mem_pool_host.layout

    def _batch_preprocess(self, keys, host_indices):
        """Preprocess keys and host_indices to get buffer metadata for zero-copy operations.
        Uses get_page_buffer_meta() to get memory pointers and sizes directly.
        Similar to mooncake_store's _batch_preprocess but adapted for file backend.
        
        Returns:
            ptr_list: List of memory pointers (for MHA: K and V pairs, for MLA: single pointer per page)
            element_size_list: List of sizes for each pointer
        """
        assert len(keys) > 0
        assert len(keys) == len(host_indices) // self.mem_pool_host.page_size
        # Get buffer metadata (pointers and sizes) for zero-copy access
        ptr_list, element_size_list = self.mem_pool_host.get_page_buffer_meta(host_indices)
        return ptr_list, element_size_list

    def _batch_exist(self, keys: List[str]) -> List[bool]:
        """Check if batch files exist for given keys.
        Returns a list of booleans indicating existence for each key.
        Similar to mooncake_store's _batch_exist but for file backend.
        """
        exist_results = []
        for key in keys:
            batch_file_path = os.path.join(
                self.file_path, f"{self._get_suffixed_key(key)}.batch.bin"
            )
            exist_results.append(os.path.exists(batch_file_path))
        return exist_results

    def _write_from_ptr(self, file_path: str, ptr: int, size: int) -> bool:
        """Write data from a memory pointer to a file using ctypes.
        This is a true zero-copy operation - no tensor operations involved.
        """
        try:
            buffer = (ctypes.c_uint8 * size).from_address(ptr)
            with open(file_path, "wb", buffering=0) as f:
                f.write(buffer)
            return True
        except Exception as e:
            logger.error(f"Failed to write from pointer to {file_path}: {e}")
            return False

    def _write_kv_pair_from_ptrs(
        self, file_path: str, k_ptr: int, k_size: int, v_ptr: int, v_size: int
    ) -> bool:
        """Write K and V data from memory pointers to a single file.
        For MHA models, we need to write K and V together in one file.
        This is a true zero-copy operation.
        """
        try:
            with open(file_path, "wb", buffering=0) as f:
                # Write K data
                k_buffer = (ctypes.c_uint8 * k_size).from_address(k_ptr)
                f.write(k_buffer)
                # Write V data
                v_buffer = (ctypes.c_uint8 * v_size).from_address(v_ptr)
                f.write(v_buffer)
            return True
        except Exception as e:
            logger.error(f"Failed to write KV pair from pointers to {file_path}: {e}")
            return False

    def get(
        self,
        key: str,
        target_location: torch.Tensor,
        target_sizes: Optional[Any] = None,
    ) -> torch.Tensor | None:
        key = self._get_suffixed_key(key)
        tensor_path = os.path.join(self.file_path, f"{key}.bin")
        try:
            expected = target_location.numel() * target_location.element_size()
            with open(tensor_path, "rb", buffering=0) as f:
                buf = memoryview(target_location.view(torch.uint8).contiguous().numpy())
                if f.readinto(buf) != expected:
                    raise IOError(f"Short read for {key}")
            return target_location
        except FileNotFoundError:
            logger.warning(f"Failed to fetch {key} from HiCacheFile storage.")
            return None

    def batch_get(
        self,
        keys: List[str],
        target_locations: List[torch.Tensor],
        target_sizes: Optional[Any] = None,
    ) -> List[torch.Tensor | None]:
        results = []
        for key, target_location in zip(keys, target_locations or [None] * len(keys)):
            # Stage 1: Build file path
            suffixed_key = self._get_suffixed_key(key)
            tensor_path = os.path.join(self.file_path, f"{suffixed_key}.bin")

            try:
                # Stage 2: Make contiguous
                expected = target_location.numel() * target_location.element_size()

                # Stage 3: Convert to numpy view
                buf = memoryview(target_location.view(torch.uint8).contiguous().numpy())

                # Stage 4: File I/O
                with open(tensor_path, "rb", buffering=0) as f:
                    if f.readinto(buf) != expected:
                        results.append(None)
                        continue

                results.append(target_location)
            except FileNotFoundError:
                results.append(None)

        return results

    def set(
        self,
        key: str,
        value: Optional[Any] = None,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        # Stage 1: Check if exists
        if self.exists(key):
            logger.debug(f"Key {key} already exists. Skipped.")
            return True

        # Stage 2: Build file path
        key = self._get_suffixed_key(key)
        tensor_path = os.path.join(self.file_path, f"{key}.bin")

        try:
            # Stage 3: Make contiguous
            contiguous_value = value.contiguous()

            # Stage 4: Convert to numpy view
            numpy_view = contiguous_value.view(dtype=torch.uint8).numpy()

            # Stage 5: File I/O
            numpy_view.tofile(tensor_path)

            return True
        except Exception as e:
            logger.error(f"Failed to save tensor {key}: {e}")
            return False

    def batch_set(
        self,
        keys: List[str],
        values: Optional[Any] = None,
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        for key, value in zip(keys, values):
            if not self.set(key, value):
                return False

        return True

    def exists(self, key: str) -> bool:
        """Check if key exists in storage. For v1 interface, check .batch.bin file."""
        suffixed_key = self._get_suffixed_key(key)
        # Check batch file first (v1 interface)
        batch_file_path = os.path.join(self.file_path, f"{suffixed_key}.batch.bin")
        if os.path.exists(batch_file_path):
            return True
        # Fallback to individual file (old interface)
        tensor_path = os.path.join(self.file_path, f"{suffixed_key}.bin")
        return os.path.exists(tensor_path)

    def clear(self) -> bool:
        try:
            for filename in os.listdir(self.file_path):
                file_path = os.path.join(self.file_path, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            logger.info("Cleared all entries in HiCacheFile storage.")
            return True
        except Exception as e:
            logger.error(f"Failed to clear HiCacheFile storage: {e}")
            return False

    def batch_get_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        """True batch get: read multiple tensors from one batch file."""
        if not keys or len(host_indices) == 0:
            return [False] * len(keys) if keys else []

        # Optimized: inline preprocessing to avoid function call overhead
        assert len(keys) > 0
        assert len(host_indices) % self.mem_pool_host.page_size == 0
        assert len(keys) == len(host_indices) // self.mem_pool_host.page_size

        page_num = len(keys)
        results = []

        for i in range(page_num):
            key = keys[i]

            # Stage 1: Build file path
            batch_file_path = os.path.join(
                self.file_path, f"{self._get_suffixed_key(key)}.batch.bin"
            )

            # Stage 2: Get tensor from memory pool
            page_start_idx = i * self.mem_pool_host.page_size
            actual_idx = host_indices[page_start_idx].item()
            page_tensor = self.mem_pool_host.get_data_page(actual_idx, flat=True)

            # Stage 3: Make contiguous
            if not page_tensor.is_contiguous():
                page_tensor = page_tensor.contiguous()

            try:
                # Stage 4: Convert to numpy view
                numpy_view = page_tensor.view(torch.uint8).numpy()
                expected_size = page_tensor.numel() * page_tensor.element_size()

                # Stage 5: File I/O
                with open(batch_file_path, "rb", buffering=0) as f:
                    if f.readinto(memoryview(numpy_view)) != expected_size:
                        results.append(False)
                        continue

                results.append(True)
            except FileNotFoundError:
                results.append(False)
            except Exception as e:
                logger.error(f"Failed to read batch file for {key}: {e}")
                results.append(False)

        return results

    def batch_set_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        """True batch set: merge multiple tensors into one file and use batch I/O.
        Reference implementation from mooncake_store to avoid contiguous() calls.
        Uses get_page_buffer_meta() to get memory pointers directly for zero-copy operations.
        """
        if not keys or len(host_indices) == 0:
            return []

        assert len(keys) > 0
        assert len(host_indices) % self.mem_pool_host.page_size == 0
        assert len(keys) == len(host_indices) // self.mem_pool_host.page_size

        page_num = len(keys)
        
        # Stage 1: Get buffer metadata (pointers and sizes) using get_page_buffer_meta
        # This is the key difference - we use the built-in method instead of manual address calculation
        ptr_list, element_size_list = self._batch_preprocess(keys, host_indices)
        
        # Stage 2: Check which files already exist (similar to mooncake_store's _batch_exist)
        exist_results = self._batch_exist(keys)
        
        # Stage 3: Prepare write operations only for non-existing keys
        write_keys = []
        write_indices = []
        write_results = [False] * page_num
        
        for i in range(page_num):
            if exist_results[i]:
                # File already exists, mark as success
                write_results[i] = True
            else:
                # File doesn't exist, need to write
                write_keys.append(keys[i])
                write_indices.append(i)
        
        # Stage 4: Write only non-existing files using pointers directly
        if self.is_mla_backend:
            # MLA: one pointer per page
            for key, idx in zip(write_keys, write_indices):
                batch_file_path = os.path.join(
                    self.file_path, f"{self._get_suffixed_key(key)}.batch.bin"
                )
                try:
                    # MLA returns one pointer per page
                    ptr = ptr_list[idx]
                    size = element_size_list[idx]
                    success = self._write_from_ptr(batch_file_path, ptr, size)
                    write_results[idx] = success
                except Exception as e:
                    logger.error(f"Failed to write batch for key {key}: {e}")
                    write_results[idx] = False
        else:
            # MHA: K and V pointer pairs per page
            for key, idx in zip(write_keys, write_indices):
                batch_file_path = os.path.join(
                    self.file_path, f"{self._get_suffixed_key(key)}.batch.bin"
                )
                try:
                    # MHA returns K and V as pairs: [K0, V0, K1, V1, ...]
                    # For each page, we have K and V pointers
                    k_idx = idx * 2
                    v_idx = idx * 2 + 1
                    k_ptr = ptr_list[k_idx]
                    k_size = element_size_list[k_idx]
                    v_ptr = ptr_list[v_idx]
                    v_size = element_size_list[v_idx]
                    success = self._write_kv_pair_from_ptrs(
                        batch_file_path, k_ptr, k_size, v_ptr, v_size
                    )
                    write_results[idx] = success
                except Exception as e:
                    logger.error(f"Failed to write batch for key {key}: {e}")
                    write_results[idx] = False
        
        return write_results
