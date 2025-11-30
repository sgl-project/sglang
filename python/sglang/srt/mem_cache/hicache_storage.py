import hashlib
import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
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

    def queue_cleanup(self, keys: List[str]):
        """
        Queue keys for background cleanup.
        This is an optional method that can be implemented by subclasses that need
        background cleanup functionality (e.g., file-based storage).
        Args:
            keys: List of keys to be deleted from storage
        """
        # Default implementation: do nothing
        # Subclasses can override this to implement background cleanup
        pass


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

        self.tp_rank = tp_rank
        if not os.path.exists(self.file_path) and self.tp_rank == 0:
            os.makedirs(self.file_path)
            logger.info(f"Created HiCacheFile storage directory at {self.file_path}")

        # Capacity control
        # Get capacity limit from config or environment variable
        # Default: no limit (None means unlimited)
        self.max_capacity_bytes = None
        if (
            storage_config.extra_config
            and "max_capacity_gb" in storage_config.extra_config
        ):
            self.max_capacity_bytes = int(
                storage_config.extra_config["max_capacity_gb"] * 1024 * 1024 * 1024
            )
        elif os.getenv("SGLANG_HICACHE_FILE_BACKEND_MAX_CAPACITY_GB"):
            self.max_capacity_bytes = int(
                float(os.getenv("SGLANG_HICACHE_FILE_BACKEND_MAX_CAPACITY_GB"))
                * 1024
                * 1024
                * 1024
            )
        # Current capacity tracking (in bytes)
        self._current_capacity_bytes = 0
        self._capacity_lock = threading.Lock()
        # LRU tracking for automatic capacity management
        # OrderedDict maintains insertion order, least recently used at the front
        self._lru_order = OrderedDict()  # key -> access_time
        self._lru_lock = threading.Lock()
        # Whether cleanup fs storage directory when init
        self._cleanup_on_init = False
        if (
            storage_config.extra_config
            and "cleanup_on_init" in storage_config.extra_config
        ):
            self._cleanup_on_init = bool(storage_config.extra_config["cleanup_on_init"])
        elif os.getenv("SGLANG_HICACHE_FILE_BACKEND_CLEANUP_ON_INIT"):
            self._cleanup_on_init = os.getenv(
                "SGLANG_HICACHE_FILE_BACKEND_CLEANUP_ON_INIT"
            ).lower() in ("true", "1", "yes")

        if self._cleanup_on_init and self.tp_rank == 0:
            self.clear()
        self._refresh_capacity()

        if self.max_capacity_bytes is not None:
            logger.info(
                f"HiCacheFile storage capacity limit: {self.max_capacity_bytes / (1024**3):.2f} GB, "
                f"current: {self._current_capacity_bytes / (1024**3):.2f} GB"
            )

        # Initialize LRU from existing files
        self._initialize_lru_from_files()

    def _refresh_capacity(self):
        """Refresh current capacity by scanning all files in storage directory."""
        try:
            total_size = 0
            if os.path.exists(self.file_path):
                for filename in os.listdir(self.file_path):
                    file_path = os.path.join(self.file_path, filename)
                    if os.path.isfile(file_path):
                        total_size += os.path.getsize(file_path)
            with self._capacity_lock:
                self._current_capacity_bytes = total_size
        except Exception as e:
            logger.warning(f"Failed to refresh storage capacity: {e}")

    def _initialize_lru_from_files(self):
        """Initialize LRU order from existing files based on file modification time."""
        try:
            if not os.path.exists(self.file_path):
                return

            # Get all files with their modification times
            file_times = []
            for filename in os.listdir(self.file_path):
                file_path = os.path.join(self.file_path, filename)
                if os.path.isfile(file_path) and filename.endswith(".bin"):
                    # Extract key from filename (remove .bin suffix and config_suffix)
                    if filename.endswith(self.config_suffix + ".bin"):
                        key = filename[
                            : -(len(self.config_suffix) + 4)
                        ]  # Remove .bin and suffix
                    else:
                        key = filename[:-4]  # Remove .bin
                    mtime = os.path.getmtime(file_path)
                    file_times.append((key, mtime))

            # Sort by modification time (oldest first) and populate LRU
            file_times.sort(key=lambda x: x[1])
            with self._lru_lock:
                for key, mtime in file_times:
                    self._lru_order[key] = mtime
        except Exception as e:
            logger.warning(f"Failed to initialize LRU from existing files: {e}")

    def _update_lru(self, key: str):
        """Update LRU order: move key to end (most recently used)."""
        with self._lru_lock:
            # Remove from current position and add to end
            if key in self._lru_order:
                self._lru_order.move_to_end(key)
            else:
                self._lru_order[key] = time.time()

    def _evict_lru_keys(self, target_bytes: int) -> List[str]:
        """
        Evict least recently used keys until we free at least target_bytes.
        Returns list of keys that were evicted.
        """
        if self.max_capacity_bytes is None:
            return []

        evicted_keys = []
        freed_bytes = 0

        # Collect keys to evict (with LRU lock held)
        keys_to_remove = []
        with self._lru_lock:
            # Iterate from oldest (front) to newest (back)
            for key in list(self._lru_order.keys()):
                if freed_bytes >= target_bytes:
                    break

                tensor_path = os.path.join(
                    self.file_path, f"{self._get_suffixed_key(key)}.bin"
                )
                if os.path.exists(tensor_path):
                    file_size = os.path.getsize(tensor_path)
                    keys_to_remove.append((key, file_size))
                    freed_bytes += file_size

            # Remove from LRU order immediately (before file deletion)
            for key, _ in keys_to_remove:
                if key in self._lru_order:
                    del self._lru_order[key]

        # Delete files and update capacity (outside LRU lock to avoid deadlock)
        for key, file_size in keys_to_remove:
            try:
                tensor_path = os.path.join(
                    self.file_path, f"{self._get_suffixed_key(key)}.bin"
                )
                if os.path.exists(tensor_path):
                    os.remove(tensor_path)
                    self._update_capacity_on_delete(file_size)
                    evicted_keys.append(key)
            except FileNotFoundError:
                # File was already deleted by another process/thread, ignore
                logger.warning(f"File not found (already deleted): {tensor_path}")
                pass
            except Exception as e:
                logger.warning(f"Failed to evict LRU key {key}: {e}")

        return evicted_keys

    def _update_capacity_on_add(self, file_size: int):
        """Update capacity after adding a file."""
        self._current_capacity_bytes += file_size

    def _update_capacity_on_delete(self, file_size: int):
        """Update capacity after deleting a file."""
        self._current_capacity_bytes = max(0, self._current_capacity_bytes - file_size)

    def get_current_capacity(self) -> int:
        """Get current storage capacity in bytes."""
        return self._current_capacity_bytes

    def get_capacity_ratio(self) -> float:
        """Get current capacity / max capacity ratio. Returns 0.0 if unlimited."""
        if self.max_capacity_bytes is None:
            return 0.0
        if self.max_capacity_bytes == 0:
            return 1.0
        return min(1.0, self._current_capacity_bytes / self.max_capacity_bytes)

    def is_over_capacity(self, threshold_ratio: float = 0.95) -> bool:
        """
        Check if storage is over capacity threshold.
        Args:
            threshold_ratio: Ratio at which to consider storage over capacity (default 0.95 = 95%)
        Returns:
            True if capacity exceeds threshold
        """
        if self.max_capacity_bytes is None:
            return False
        return self.get_capacity_ratio() >= threshold_ratio

    def _get_suffixed_key(self, key: str) -> str:
        return key + self.config_suffix

    def get(
        self,
        key: str,
        target_location: torch.Tensor,
        target_sizes: Optional[Any] = None,
    ) -> torch.Tensor | None:
        suffixed_key = self._get_suffixed_key(key)
        tensor_path = os.path.join(self.file_path, f"{suffixed_key}.bin")
        try:
            expected = target_location.numel() * target_location.element_size()
            with open(tensor_path, "rb", buffering=0) as f:
                buf = memoryview(target_location.view(torch.uint8).contiguous().numpy())
                if f.readinto(buf) != expected:
                    raise IOError(f"Short read for {key}")
            # Update LRU: mark as recently used
            self._update_lru(key)
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
        return [
            self.get(key, target_location)
            for key, target_location in zip(
                keys, target_locations or [None] * len(keys)
            )
        ]

    def set(
        self,
        key: str,
        value: Optional[Any] = None,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        if self.exists(key):
            logger.debug(f"Key {key} already exists. Skipped.")
            # Update LRU even if key already exists
            self._update_lru(key)
            return True

        suffixed_key = self._get_suffixed_key(key)
        tensor_path = os.path.join(self.file_path, f"{suffixed_key}.bin")
        try:
            # Calculate file size before writing
            file_size = value.numel() * value.element_size()

            # Check capacity before writing
            if self.max_capacity_bytes is not None:
                with self._capacity_lock:
                    needed_space = file_size
                    current_capacity = self._current_capacity_bytes

                    if current_capacity + needed_space > self.max_capacity_bytes:
                        # Need to free space
                        free_target = (
                            current_capacity + needed_space - self.max_capacity_bytes
                        )
                        # Free a bit extra to avoid frequent evictions (20% margin)
                        free_target = int(free_target * 1.2)

                        evicted = self._evict_lru_keys(free_target)
                        if evicted:
                            logger.debug(
                                f"Evicted {len(evicted)} LRU keys to free {free_target / (1024**2):.2f} MB "
                                f"for new key {key}"
                            )

                        # Check again after eviction
                        if (
                            self._current_capacity_bytes + needed_space
                            > self.max_capacity_bytes
                        ):
                            logger.warning(
                                f"Storage capacity exceeded after LRU eviction. "
                                f"Current: {self._current_capacity_bytes / (1024**3):.2f} GB, "
                                f"Limit: {self.max_capacity_bytes / (1024**3):.2f} GB, "
                                f"Requested: {needed_space / (1024**2):.2f} MB. "
                                f"Key {key} will not be stored."
                            )
                            return False

            value.contiguous().view(dtype=torch.uint8).numpy().tofile(tensor_path)
            # Update capacity after successful write
            actual_size = os.path.getsize(tensor_path)
            self._update_capacity_on_add(actual_size)
            # Update LRU: add as most recently used
            self._update_lru(key)
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
        key = self._get_suffixed_key(key)
        tensor_path = os.path.join(self.file_path, f"{key}.bin")
        return os.path.exists(tensor_path)

    def delete(self, key: str) -> bool:
        """
        Delete a key from storage.
        Args:
            key: The key to delete
        Returns:
            True if the key was deleted successfully, False otherwise.
        """
        suffixed_key = self._get_suffixed_key(key)
        tensor_path = os.path.join(self.file_path, f"{suffixed_key}.bin")
        try:
            if os.path.exists(tensor_path):
                file_size = os.path.getsize(tensor_path)
                os.remove(tensor_path)
                # Update capacity after successful deletion
                self._update_capacity_on_delete(file_size)
                # Remove from LRU
                with self._lru_lock:
                    if key in self._lru_order:
                        del self._lru_order[key]
                logger.debug(f"Deleted key {key} from HiCacheFile storage.")
                return True
            else:
                logger.debug(f"Key {key} does not exist in HiCacheFile storage.")
                return False
        except FileNotFoundError:
            # File was already deleted by another process/thread, ignore
            logger.warning(f"File not found (already deleted): {tensor_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete key {key} from HiCacheFile storage: {e}")
            return False

    def batch_delete(self, keys: List[str]) -> bool:
        """
        Delete multiple keys from storage.
        Args:
            keys: List of keys to delete
        Returns:
            True if all keys were deleted successfully, False otherwise.
        """
        success = True
        for key in keys:
            if not self.delete(key):
                success = False
        return success

    def clear(self) -> bool:
        try:
            logger.info(f"Starting clear() for HiCacheFile storage at {self.file_path}")

            if not os.path.exists(self.file_path):
                # Directory doesn't exist, nothing to clear
                self._current_capacity_bytes = 0
                with self._lru_lock:
                    self._lru_order.clear()
                logger.info(
                    "HiCacheFile storage directory does not exist, nothing to clear."
                )
                return True

            # Get list of files first to avoid issues with concurrent modifications
            files_to_delete = []
            for filename in os.listdir(self.file_path):
                file_path = os.path.join(self.file_path, filename)
                if os.path.isfile(file_path):
                    files_to_delete.append(file_path)

            # Delete files with error handling for missing files
            deleted_count = 0
            failed_count = 0
            for file_path in files_to_delete:
                try:
                    if os.path.exists(file_path):  # Double check before deletion
                        os.remove(file_path)
                        deleted_count += 1
                    else:
                        logger.debug(f"File already deleted: {file_path}")
                except FileNotFoundError:
                    # File was already deleted by another process/thread, ignore
                    logger.debug(f"File not found (already deleted): {file_path}")
                    pass
                except Exception as e:
                    failed_count += 1
                    logger.warning(f"Failed to delete file {file_path}: {e}")
                    if failed_count <= 5:  # Only log first 5 failures in detail
                        logger.warning(
                            f"Error details for {file_path}: {e}", exc_info=True
                        )

            logger.info(
                f"File deletion completed: {deleted_count} deleted, {failed_count} failed out of {len(files_to_delete)} total"
            )

            # Reset capacity and LRU after clearing
            logger.debug("Resetting capacity and LRU")
            self._current_capacity_bytes = 0  # Atomic assignment in CPython
            logger.debug("Acquiring LRU lock...")
            with self._lru_lock:
                logger.debug("LRU lock acquired, clearing LRU order")
                self._lru_order.clear()
            logger.debug("LRU lock released")

            logger.info(f"Cleared {deleted_count} entries from HiCacheFile storage.")
            return True
        except Exception as e:
            logger.error(f"Failed to clear HiCacheFile storage: {e}", exc_info=True)
            return False

    def queue_cleanup(self, keys: List[str]):
        """
        Delete storage keys when cache nodes are evicted.
        This method can be called by cache controller to notify storage backend
        that certain keys have been evicted from cache and can be deleted.

        Note: This is a synchronous operation that immediately deletes the keys.
        LRU-based automatic cleanup handles capacity management independently.

        Args:
            keys: List of hash keys to be deleted from storage
        """
        if not keys:
            return

        # Immediately delete keys synchronously when cache evicts them
        # No need for async queue since we want immediate capacity release
        for key in keys:
            try:
                self.delete(key)  # delete() handles LRU removal and capacity update
            except Exception as e:
                logger.warning(f"Failed to cleanup evicted key {key}: {e}")
