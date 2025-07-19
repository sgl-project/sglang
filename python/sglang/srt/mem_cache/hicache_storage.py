import hashlib
import logging
import os
from abc import ABC, abstractmethod
from typing import List, Optional

import torch

logger = logging.getLogger(__name__)


def get_hash_str(token_ids: List[int], prior_hash: Optional[str] = None) -> str:
    hasher = hashlib.sha256()

    if prior_hash:
        hasher.update(bytes.fromhex(prior_hash))

    for t in token_ids:
        hasher.update(t.to_bytes(4, byteorder="little", signed=False))

    return hasher.hexdigest()


class HiCacheStorage(ABC):
    """
    HiCacheStorage is a class that provides a generic key-value interface for storing and retrieving KV cache.
    It abstracts the underlying storage mechanism, allowing different implementations to be used.
    """

    # todo, translate tensor object access for different TP ranks
    # potentially pass model and TP configs into storage backend
    # todo, the page size of storage backend does not have to be the same as the same as host memory pool

    @abstractmethod
    def get(
        self, key: str, target_location: Optional[torch.Tensor] = None
    ) -> torch.Tensor | None:
        """
        Retrieve the value associated with the given key.
        Returns None if the key does not exist.
        """
        pass

    @abstractmethod
    def batch_get(
        self, keys: List[str], target_locations: Optional[List[torch.Tensor]] = None
    ) -> List[torch.Tensor | None]:
        """
        Retrieve values for multiple keys.
        Returns a list of tensors or None for each key.
        """
        pass

    @abstractmethod
    def set(self, key, value) -> bool:
        """
        Store the value associated with the given key.
        Returns True if the operation was successful, False otherwise.
        """
        pass

    @abstractmethod
    def batch_set(self, keys: List[str], values: List[torch.Tensor]) -> bool:
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


class HiCacheFile(HiCacheStorage):

    def __init__(self, file_path: str = "/tmp/hicache"):
        self.file_path = file_path
        if not os.path.exists(self.file_path):
            os.makedirs(self.file_path)
            logger.info(f"Created HiCacheFile storage directory at {self.file_path}")

    def get(
        self, key: str, target_location: Optional[torch.Tensor] = None
    ) -> torch.Tensor | None:
        tensor_path = os.path.join(self.file_path, f"{key}.bin")
        try:
            # todo: fixing the target_location logic to enable in-place loading
            loaded_tensor = torch.load(tensor_path)
            if isinstance(loaded_tensor, torch.Tensor):
                return loaded_tensor
            else:
                logger.error(f"Loaded data for key {key} is not a tensor.")
                return None
        except FileNotFoundError:
            return None

    def batch_get(
        self,
        keys: List[str],
        target_locations: Optional[List[torch.Tensor]] = None,
    ) -> List[torch.Tensor | None]:
        return [
            self.get(key, target_location)
            for key, target_location in zip(
                keys, target_locations or [None] * len(keys)
            )
        ]

    def set(self, key: str, value: torch.Tensor) -> bool:
        tensor_path = os.path.join(self.file_path, f"{key}.bin")
        if self.exists(key):
            logger.debug(f"Key {key} already exists. Skipped.")
            return True
        try:
            torch.save(value, tensor_path)
            return True
        except Exception as e:
            logger.error(f"Failed to save tensor {key}: {e}")
            return False

    def batch_set(self, keys: List[str], values: List[torch.Tensor]) -> bool:
        for key, value in zip(keys, values):
            if not self.set(key, value):
                return False
        return True

    def exists(self, key: str) -> bool:
        tensor_path = os.path.join(self.file_path, f"{key}.bin")
        return os.path.exists(tensor_path)

    def delete(self, key: str) -> None:
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
