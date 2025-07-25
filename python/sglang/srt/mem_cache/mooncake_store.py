"""
Mooncake distributed storage implementation for SGLang memory cache.

This module provides a distributed storage backend using Mooncake for caching
key-value pairs in SGLang's memory management system.
"""

import json
import logging
import os
from typing import List, Optional

import torch

from sglang.srt.mem_cache.hicache_storage import HiCacheStorage
from mooncake.store import MooncakeDistributedStore

logger = logging.getLogger(__name__)

class MooncakeStoreConfig:
    """Configuration class for Mooncake distributed store."""

    def __init__(
        self,
        local_hostname: str,
        metadata_server: str,
        global_segment_size: int,
        local_buffer_size: int,
        protocol: str = "tcp",
        device_name: str = "",
        master_server_address: str = "",
    ):
        """
        Initialize Mooncake store configuration.

        Args:
            local_hostname: Local hostname for the store
            metadata_server: Metadata server address
            global_segment_size: Size of global segment
            local_buffer_size: Size of local buffer
            protocol: Communication protocol (default: "tcp")
            device_name: Device name (default: "")
            master_server_address: Master server address (default: "")
        """
        self.local_hostname = local_hostname
        self.metadata_server = metadata_server
        self.global_segment_size = global_segment_size
        self.local_buffer_size = local_buffer_size
        self.protocol = protocol
        self.device_name = device_name
        self.master_server_address = master_server_address

    @classmethod
    def from_file(cls, file_path: str) -> "MooncakeStoreConfig":
        """
        Load configuration from a JSON file.

        Args:
            file_path: Path to the JSON configuration file

        Returns:
            MooncakeStoreConfig instance

        Raises:
            FileNotFoundError: If the config file doesn't exist
            json.JSONDecodeError: If the config file is invalid JSON
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                config = json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Invalid JSON in config file {file_path}: {e}")

        return cls(
            local_hostname=config.get("local_hostname"),
            metadata_server=config.get("metadata_server"),
            global_segment_size=config.get("global_segment_size"),
            local_buffer_size=config.get("local_buffer_size"),
            protocol=config.get("protocol", "tcp"),
            device_name=config.get("device_name", ""),
            master_server_address=config.get("master_server_address"),
        )

    @classmethod
    def load_from_env(cls) -> "MooncakeStoreConfig":
        """
        Load configuration from environment variable.

        Returns:
            MooncakeStoreConfig instance

        Raises:
            ValueError: If MOONCAKE_CONFIG_PATH environment variable is not set
        """
        config_file_path = os.getenv("MOONCAKE_CONFIG_PATH")
        if not config_file_path:
            raise ValueError(
                "The environment variable 'MOONCAKE_CONFIG_PATH' is not set."
            )
        return cls.from_file(config_file_path)


class MooncakeStore(HiCacheStorage):
    """
    Mooncake distributed storage implementation for HiCache.

    This class provides a distributed storage backend using Mooncake for caching
    key-value pairs across multiple nodes.
    """

    def __init__(self, rank: int, bytes_per_page: int, dtype: torch.dtype):
        """
        Initialize MooncakeStore.

        Args:
            rank: The rank of this store instance for key suffixing
            bytes_per_page: Number of bytes per page for tensor reconstruction
            dtype: Data type for tensor reconstruction

        Raises:
            RuntimeError: If store setup fails
        """
        super().__init__()
        self.store = MooncakeDistributedStore()
        self.config = MooncakeStoreConfig.load_from_env()
        logger.info("Mooncake Configuration loaded successfully.")
        self.suffix = f"_{rank}"

        # Store tensor reconstruction parameters
        self.bytes_per_page = bytes_per_page
        self.dtype = dtype
        self.numel = self.bytes_per_page // self.dtype.itemsize

        # Setup the distributed store
        setup_code = self.store.setup(
            self.config.local_hostname,
            self.config.metadata_server,
            self.config.global_segment_size,
            self.config.local_buffer_size,
            self.config.protocol,
            self.config.device_name,
            self.config.master_server_address,
        )

        if setup_code != 0:
            raise RuntimeError(f"Failed to setup MooncakeStore with code: {setup_code}")

        logger.info(f"MooncakeStore initialized successfully for rank {rank}")

    def _get_suffixed_key(self, key: str) -> str:
        """Add rank suffix to key for distributed storage."""
        return key + self.suffix

    def get(
        self, key: str, target_location: Optional[torch.Tensor] = None
    ) -> Optional[torch.Tensor]:
        """
        Retrieve a single tensor by key.

        Args:
            key: The key to retrieve
            target_location: Optional target tensor location (currently unused)

        Returns:
            The tensor if found, None otherwise
        """
        suffixed_key = self._get_suffixed_key(key)
        results = self.batch_get([suffixed_key], [target_location] if target_location else None)
        return results[0] if results else None

    def batch_get(
        self,
        keys: List[str],
        target_locations: Optional[List[torch.Tensor]] = None,
    ) -> List[Optional[torch.Tensor]]:
        """
        Retrieve multiple tensors by keys.

        Args:
            keys: List of keys to retrieve
            target_locations: Optional target tensor locations (currently unused)

        Returns:
            List of tensors or None for each key
        """
        # Note: target_locations parameter is kept for interface compatibility
        # but not currently used in this implementation
        _ = target_locations  # Acknowledge unused parameter

        if not keys:
            return []

        # Add suffix to all keys
        suffixed_keys = [self._get_suffixed_key(key) for key in keys]

        # Check which keys exist
        exists_result = self.store.batch_is_exist(suffixed_keys)
        if len(exists_result) != len(suffixed_keys):
            logger.error(f"Mismatch in exists_result length: {len(exists_result)} vs {len(suffixed_keys)}")
            return [None] * len(keys)

        # Find indices of existing keys
        existing_indices = [i for i, exists in enumerate(exists_result) if exists == 1]

        if not existing_indices:
            return [None] * len(keys)

        # Get buffers for existing keys
        existing_keys = [suffixed_keys[i] for i in existing_indices]
        try:
            buffers = self.store.batch_get_buffer(existing_keys)
        except Exception as e:
            logger.error(f"Failed to get buffers for keys {existing_keys}: {e}")
            return [None] * len(keys)

        # Process buffers and create result tensors
        results = [None] * len(keys)
        for i, (buffer, original_index) in enumerate(zip(buffers, existing_indices)):
            if buffer is not None:
                try:
                    # Convert buffer to tensor with proper dtype and shape
                    tensor_data = torch.frombuffer(buffer, dtype=torch.uint8).view(self.dtype).reshape(self.numel)
                    results[original_index] = tensor_data
                except Exception as e:
                    logger.error(f"Failed to convert buffer to tensor for key {existing_keys[i]}: {e}")
                    results[original_index] = None
            else:
                logger.warning(f"Buffer is None for existing key {existing_keys[i]}")

        return results

    def set(self, key: str, value: torch.Tensor) -> bool:
        """
        Store a single tensor by key.

        Args:
            key: The key to store
            value: The tensor to store

        Returns:
            True if successful, False otherwise
        """
        results = self.batch_set([key], [value])
        return results[0] if results else False

    def batch_set(self, keys: List[str], values: List[torch.Tensor]) -> List[bool]:
        """
        Store multiple tensors by keys.

        Args:
            keys: List of keys to store
            values: List of tensors to store

        Returns:
            List of success flags for each key-value pair
        """
        if not keys or not values or len(keys) != len(values):
            logger.error(f"Invalid input: keys length {len(keys)}, values length {len(values)}")
            return [False] * len(keys)

        # Add suffix to all keys
        suffixed_keys = [self._get_suffixed_key(key) for key in keys]

        # Check which keys already exist
        try:
            exists_result = self.store.batch_is_exist(suffixed_keys)
        except Exception as e:
            logger.error(f"Failed to check key existence: {e}")
            return [False] * len(keys)

        if len(exists_result) != len(suffixed_keys):
            logger.error(f"Mismatch in exists_result length: {len(exists_result)} vs {len(suffixed_keys)}")
            return [False] * len(keys)

        # Find indices of keys that don't exist (need to be stored)
        new_indices = [i for i, exists in enumerate(exists_result) if exists == 0]

        if not new_indices:
            # All keys already exist, return True for all
            logger.debug("All keys already exist, skipping storage")
            return [True] * len(keys)

        # Prepare data for batch storage
        new_keys = [suffixed_keys[i] for i in new_indices]
        new_values = []

        for i in new_indices:
            try:
                # Convert tensor to numpy array as bytes
                tensor_bytes = values[i].cpu().contiguous().view(torch.uint8).numpy()
                new_values.append(tensor_bytes)
            except Exception as e:
                logger.error(f"Failed to convert tensor to bytes for key {keys[i]}: {e}")
                return [False] * len(keys)

        # Perform batch storage
        try:
            ret = self.store.put_batch(new_keys, new_values)
        except Exception as e:
            logger.error(f"Failed to store batch: {e}")
            return [False] * len(keys)

        # Prepare results
        results = [True] * len(keys)  # Default to True for existing keys

        if ret != 0:
            # If batch storage failed, mark new keys as failed
            for i in new_indices:
                results[i] = False
            logger.error(f"Batch storage failed with return code: {ret}")

        return results

    def delete(self, key: str) -> None:
        """
        Delete a key from storage.

        Args:
            key: The key to delete
        """
        suffixed_key = self._get_suffixed_key(key)
        try:
            self.store.remove(suffixed_key)
            logger.debug(f"Successfully deleted key: {key}")
        except Exception as e:
            logger.error(f"Failed to delete key {key}: {e}")

    def exists(self, key: str) -> bool:
        """
        Check if a key exists in storage.

        Args:
            key: The key to check

        Returns:
            True if key exists, False otherwise
        """
        suffixed_key = self._get_suffixed_key(key)
        try:
            ret = self.store.is_exist(suffixed_key)
            return ret == 1
        except Exception as e:
            logger.error(f"Failed to check existence of key {key}: {e}")
            return False

    def clear(self) -> None:
        pass

    def close(self) -> None:
        pass
