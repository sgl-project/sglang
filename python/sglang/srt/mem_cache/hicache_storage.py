import hashlib
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


def get_hash_str(token_ids: List[int], prior_hash: str = None) -> str:
    hasher = hashlib.sha256()

    if prior_hash:
        hasher.update(bytes.fromhex(prior_hash))

    for t in token_ids:
        hasher.update(t.to_bytes(4, byteorder="little", signed=False))

    return hasher.hexdigest()


@dataclass
class HiCacheStorageConfig:
    tp_rank: int
    tp_size: int
    is_mla_model: bool
    model_name: Optional[str]
    extra_config: Optional[dict] = None


class HiCacheStorage(ABC):
    """
    HiCacheStorage is a class that provides a generic key-value interface for storing and retrieving KV cache.
    It abstracts the underlying storage mechanism, allowing different implementations to be used.
    """

    # todo, potentially pass model and TP configs into storage backend
    # todo, the page size of storage backend does not have to be the same as the same as host memory pool

    def register_page_retriever(
        self, page_retriever: Callable[[list[str], torch.Tensor], Optional[Tuple]]
    ) -> None:
        """
        Register a page retriever function for rebuilding missing prefix pages during storage operations.

        The page retriever function should take page keys and host indices as parameters and return
        page data in a format suitable for the specific storage backend.
        """
        self.page_retriever = page_retriever

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

    @abstractmethod
    def batch_set(
        self,
        keys: List[str],
        values: Optional[Any] = None,
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
        prefix_pages: Optional[Tuple[List[str], torch.Tensor, int]] = None,
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

    def batch_exists(self, keys: List[str]) -> int:
        """
        Check if the keys exist in the storage.
        return the number of consecutive existing keys from the start.
        Can be overridden by subclasses for more efficient implementation.
        """
        for i in range(len(keys)):
            if not self.exists(keys[i]):
                return i
        return len(keys)


class HiCacheFile(HiCacheStorage):

    def __init__(
        self, storage_config: HiCacheStorageConfig, file_path: str = "/tmp/hicache"
    ):
        self.file_path = os.getenv("SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR", file_path)

        tp_rank, tp_size, is_mla = (
            storage_config.tp_rank,
            storage_config.tp_size,
            storage_config.is_mla_model,
        )
        self.tp_suffix = f"_{tp_rank}_{tp_size}" if tp_size > 1 and not is_mla else ""
        if not os.path.exists(self.file_path) and tp_rank == 0:
            os.makedirs(self.file_path)
            logger.info(f"Created HiCacheFile storage directory at {self.file_path}")

    def _get_suffixed_key(self, key: str) -> str:
        return key + self.tp_suffix

    def get(
        self,
        key: str,
        target_location: torch.Tensor,
        target_sizes: Optional[Any] = None,
    ) -> torch.Tensor | None:
        key = self._get_suffixed_key(key)
        tensor_path = os.path.join(self.file_path, f"{key}.bin")
        try:
            # Load directly into target_location's memory buffer
            with open(tensor_path, "rb") as f:
                target_location.set_(
                    torch.frombuffer(f.read(), dtype=target_location.dtype)
                    .reshape(target_location.shape)
                    .untyped_storage()
                )
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
            return True

        key = self._get_suffixed_key(key)
        tensor_path = os.path.join(self.file_path, f"{key}.bin")
        try:
            value.contiguous().view(dtype=torch.uint8).numpy().tofile(tensor_path)
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
        prefix_pages: Optional[Tuple[List[str], torch.Tensor, int]] = None,
    ) -> bool:
        # for key, value in zip(keys, values):
        #     if not self.set(key, value):
        #         return False

        if prefix_pages and self.page_retriever:
            prefix_page_hashes, prefix_host_indices, page_size = prefix_pages

            missing_pages = 0
            for i, page_key in enumerate(prefix_page_hashes):
                if not self.exists(page_key):
                    data = self.page_retriever(
                        [page_key],
                        prefix_host_indices[i * page_size : (i + 1) * page_size],
                    )
                    if data is not None and isinstance(data, torch.Tensor):
                        self.set(page_key, data)
                        missing_pages += 1

            if missing_pages > 0:
                logger.info(
                    f"Rebuild prefix missing pages, prefix page cnt: {len(prefix_page_hashes)}, missing page cnt: {missing_pages}"
                )

        return True

    def exists(self, key: str) -> bool:
        key = self._get_suffixed_key(key)
        tensor_path = os.path.join(self.file_path, f"{key}.bin")
        return os.path.exists(tensor_path)

    def delete(self, key: str) -> None:
        key = self._get_suffixed_key(key)
        tensor_path = os.path.join(self.file_path, f"{key}.bin")
        try:
            os.remove(tensor_path)
        except FileNotFoundError:
            logger.warning(f"Key {key} does not exist. Cannot delete.")
            return

    def clear(self) -> None:
        try:
            for filename in os.listdir(self.file_path):
                file_path = os.path.join(self.file_path, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            logger.info("Cleared all entries in HiCacheFile storage.")
        except Exception as e:
            logger.error(f"Failed to clear HiCacheFile storage: {e}")
