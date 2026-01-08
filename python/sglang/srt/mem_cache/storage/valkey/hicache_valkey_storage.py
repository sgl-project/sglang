import logging
import os
from typing import Any, List, Optional

import torch

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorage,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
)

try:
    import valkey
except ImportError as e:
    raise ImportError("Please install valkey-py: pip install valkey") from e

logger = logging.getLogger(__name__)


class HiCacheValkeyStorage(HiCacheStorage):
    """HiCache storage backend using Valkey as the storage layer."""

    def __init__(self, storage_config: HiCacheStorageConfig, **kwargs):
        """Initialize Valkey storage backend.

        Args:
            storage_config: HiCache storage configuration
            **kwargs: Additional arguments including Valkey connection parameters
        """
        self.storage_config = storage_config

        # Extract Valkey connection parameters from kwargs or environment
        host = kwargs.get("host", os.getenv("VALKEY_HOST", "localhost"))
        port = int(kwargs.get("port", os.getenv("VALKEY_PORT", 6379)))
        db = int(kwargs.get("db", os.getenv("VALKEY_DB", 0)))
        password = kwargs.get("password", os.getenv("VALKEY_PASSWORD"))

        # Initialize Valkey connection
        self.client = valkey.Valkey(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=False,  # Keep binary data as bytes
        )

        # Test connection
        try:
            self.client.ping()
            logger.info(f"Connected to Valkey at {host}:{port}")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Valkey: {e}")

        # Create key prefix based on model configuration
        tp_rank, tp_size, model_name, is_mla_model = (
            storage_config.tp_rank,
            storage_config.tp_size,
            storage_config.model_name,
            storage_config.is_mla_model,
        )
        model_name = "-".join(model_name.split("/")) if model_name else "default"

        if is_mla_model:
            self.key_prefix = f"hicache:{model_name}"
        else:
            self.key_prefix = f"hicache:{model_name}:{tp_rank}:{tp_size}"

    def _get_full_key(self, key: str) -> str:
        """Get the full Valkey key with prefix."""
        return f"{self.key_prefix}:{key}"

    def get(
        self,
        key: str,
        target_location: Optional[torch.Tensor] = None,
        target_sizes: Optional[Any] = None,
    ) -> torch.Tensor | None:
        """Retrieve tensor from Valkey storage."""
        full_key = self._get_full_key(key)

        try:
            data = self.client.get(full_key)
            if data is None:
                return None

            if target_location is not None:
                # Load directly into target location
                tensor_bytes = memoryview(
                    target_location.view(torch.uint8).contiguous().numpy()
                )
                if len(data) != len(tensor_bytes):
                    logger.error(
                        f"Size mismatch for key {key}: expected {len(tensor_bytes)}, got {len(data)}"
                    )
                    return None
                tensor_bytes[:] = data
                return target_location
            else:
                # Deserialize tensor from bytes
                return torch.frombuffer(data, dtype=torch.uint8).clone()

        except Exception as e:
            logger.error(f"Failed to get key {key}: {e}")
            return None

    def batch_get(
        self,
        keys: List[str],
        target_locations: Optional[List[torch.Tensor]] = None,
        target_sizes: Optional[Any] = None,
    ) -> List[torch.Tensor | None]:
        """Retrieve multiple tensors from Valkey storage."""
        if not keys:
            return []

        full_keys = [self._get_full_key(key) for key in keys]

        try:
            # Use pipeline for batch operations
            pipe = self.client.pipeline()
            for full_key in full_keys:
                pipe.get(full_key)
            results = pipe.execute()

            tensors = []
            for i, (data, key) in enumerate(zip(results, keys)):
                if data is None:
                    tensors.append(None)
                    continue

                if target_locations and i < len(target_locations):
                    target_location = target_locations[i]
                    tensor_bytes = memoryview(
                        target_location.view(torch.uint8).contiguous().numpy()
                    )
                    if len(data) != len(tensor_bytes):
                        logger.error(
                            f"Size mismatch for key {key}: expected {len(tensor_bytes)}, got {len(data)}"
                        )
                        tensors.append(None)
                        continue
                    tensor_bytes[:] = data
                    tensors.append(target_location)
                else:
                    tensors.append(torch.frombuffer(data, dtype=torch.uint8).clone())

            return tensors

        except Exception as e:
            logger.error(f"Failed to batch get keys {keys}: {e}")
            return [None] * len(keys)

    def set(
        self,
        key: str,
        value: Optional[torch.Tensor] = None,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        """Store tensor in Valkey storage."""
        if value is None:
            logger.error("Value cannot be None for set operation")
            return False

        full_key = self._get_full_key(key)

        try:
            # Convert tensor to bytes
            tensor_bytes = value.contiguous().view(dtype=torch.uint8).numpy().tobytes()
            self.client.set(full_key, tensor_bytes)
            return True

        except Exception as e:
            logger.error(f"Failed to set key {key}: {e}")
            return False

    def batch_set(
        self,
        keys: List[str],
        values: Optional[List[torch.Tensor]] = None,
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        """Store multiple tensors in Valkey storage."""
        if not keys or not values or len(keys) != len(values):
            logger.error("Keys and values must be provided and have same length")
            return False

        full_keys = [self._get_full_key(key) for key in keys]

        try:
            # Use pipeline for batch operations
            pipe = self.client.pipeline()
            for full_key, value in zip(full_keys, values):
                if value is None:
                    continue
                tensor_bytes = (
                    value.contiguous().view(dtype=torch.uint8).numpy().tobytes()
                )
                pipe.set(full_key, tensor_bytes)
            pipe.execute()
            return True

        except Exception as e:
            logger.error(f"Failed to batch set keys {keys}: {e}")
            return False

    def exists(self, key: str) -> bool:
        """Check if key exists in Valkey storage."""
        full_key = self._get_full_key(key)

        try:
            return bool(self.client.exists(full_key))
        except Exception as e:
            logger.error(f"Failed to check existence of key {key}: {e}")
            return False

    def batch_exists(
        self, keys: List[str], extra_info: Optional[HiCacheStorageExtraInfo] = None
    ) -> int:
        """Check existence of multiple keys, returning count of consecutive existing keys."""
        if not keys:
            return 0

        full_keys = [self._get_full_key(key) for key in keys]

        try:
            # Use pipeline for batch operations
            pipe = self.client.pipeline()
            for full_key in full_keys:
                pipe.exists(full_key)
            results = pipe.execute()

            # Count consecutive existing keys from start
            for i, exists in enumerate(results):
                if not exists:
                    return i
            return len(keys)

        except Exception as e:
            logger.error(f"Failed to batch check existence of keys {keys}: {e}")
            return 0

    def clear(self) -> None:
        """Clear all keys with the current prefix."""
        try:
            pattern = f"{self.key_prefix}:*"
            keys = self.client.keys(pattern)
            if keys:
                self.client.delete(*keys)
                logger.info(f"Cleared {len(keys)} keys with prefix {self.key_prefix}")
        except Exception as e:
            logger.error(f"Failed to clear storage: {e}")

    def get_stats(self):
        """Get storage statistics."""
        try:
            info = self.client.info()
            pattern = f"{self.key_prefix}:*"
            key_count = len(self.client.keys(pattern))

            return {
                "backend": "valkey",
                "key_count": key_count,
                "memory_used": info.get("used_memory", 0),
                "memory_human": info.get("used_memory_human", "0B"),
                "connected_clients": info.get("connected_clients", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"backend": "valkey", "error": str(e)}
