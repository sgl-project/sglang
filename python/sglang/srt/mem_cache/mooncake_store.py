# SPDX-License-Identifier: Apache-2.0
"""
This file contains a new class `MooncakeStore` that allows developers to
think of KV cache transfer operations as putting new KV cache entries
into a remote KVStore-based lookup buffer and getting existing KV caches
from this remote lookup buffer.
"""
import ctypes
import json
import logging
import os
import uuid
from dataclasses import dataclass
from typing import List, Optional

import torch

DEFAULT_GLOBAL_SEGMENT_SIZE = int(4 * 1024 * 1024 * 1024)  # 3.125 GiB
DEFAULT_LOCAL_BUFFER_SIZE = 8 * 1024 * 1024 * 1024  # 8.0 GiB

logger = logging.getLogger(__name__)


@dataclass
class MooncakeStoreConfig:
    local_hostname: str
    metadata_server: str
    global_segment_size: int
    local_buffer_size: int
    protocol: str
    device_name: str
    master_server_address: str

    @staticmethod
    def from_file(file_path: str) -> "MooncakeStoreConfig":
        """Load the config from a JSON file."""
        with open(file_path) as fin:
            config = json.load(fin)
        return MooncakeStoreConfig(
            local_hostname=config.get("local_hostname"),
            metadata_server=config.get("metadata_server"),
            global_segment_size=config.get(
                "global_segment_size", DEFAULT_GLOBAL_SEGMENT_SIZE
            ),
            local_buffer_size=config.get(
                "local_buffer_size", DEFAULT_LOCAL_BUFFER_SIZE
            ),
            protocol=config.get("protocol", "tcp"),
            device_name=config.get("device_name", ""),
            master_server_address=config.get("master_server_address"),
        )

    @staticmethod
    def load_from_env() -> "MooncakeStoreConfig":
        """Load config from a file specified in the environment variable."""
        config_file_path = os.getenv("MOONCAKE_CONFIG_PATH")
        if config_file_path is None:
            raise ValueError(
                "The environment variable 'MOONCAKE_CONFIG_PATH' is not set."
            )
        return MooncakeStoreConfig.from_file(config_file_path)


class MooncakeStore:

    def __init__(self):
        try:
            from mooncake.store import MooncakeDistributedStore
        except ImportError as e:
            raise ImportError(
                "Please install mooncake by following the instructions at "
                "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "  # noqa: E501
                "to run vLLM with MooncakeConnector."
            ) from e

        try:
            self.store = MooncakeDistributedStore()
            self.config = MooncakeStoreConfig.load_from_env()
            logger.info("Mooncake Configuration loaded successfully.")

            self.store.setup(
                self.config.local_hostname,
                self.config.metadata_server,
                self.config.global_segment_size,
                self.config.local_buffer_size,
                self.config.protocol,
                self.config.device_name,
                self.config.master_server_address,
            )
            logger.info("Connect to Mooncake store successfully.")
            self.warmup()
            logger.info("Mooncake store warmup successfully.")

        except ValueError as e:
            logger.error("Configuration loading failed: %s", e)
            raise
        except Exception as exc:
            logger.error("An error occurred while loading the configuration: %s", exc)
            raise

    def warmup(self):
        warmup_key = "sglang_mooncake_store_warmup_key" + uuid.uuid4().hex
        # 10 MB
        warmup_value = bytes(10 * 1024 * 1024)
        self.store.put(warmup_key, warmup_value)
        assert self.store.is_exist(warmup_key) == 1
        self.store.get(warmup_key)
        self.store.remove(warmup_key)

    def register_buffer(
        self,
        buffer: torch.Tensor
    ) -> None:
        try:
            buffer_ptr = buffer.data_ptr()
            buffer_size = buffer.numel() * buffer.element_size()
            self.store.register_buffer(buffer_ptr, buffer_size)
        except TypeError as err:
            logger.error("Failed to register buffer to Mooncake Store: %s", err)
            raise TypeError("Mooncake Store Register Buffer Error.") from err

    def close(self):
        # MooncakeDistributedStore will automatically call the destructor, so
        # it is unnecessary to close it manually.
        pass

    def batch_put(
        self,
        key_strs: List[str],
        buffer_ptrs: List[int],
        buffer_sizes: List[int]
    ) -> None:
        assert len(key_strs) == len(buffer_ptrs) == len(buffer_sizes)
        if len(key_strs) == 0:
            return

        for i in range(len(key_strs)):
            if (key_strs[i] is None
                or buffer_ptrs[i] is None
                or buffer_sizes[i] is None):
                return

        self._put_batch_zero_copy_impl(key_strs, buffer_ptrs, buffer_sizes)

    def batch_get(
        self,
        key_strs: List[str],
        buffer_ptrs: List[int],
        buffer_sizes: List[int]
    ) -> None:
        logger.info("batch get called python")
        assert len(key_strs) == len(buffer_ptrs) == len(buffer_sizes)
        if len(key_strs) == 0:
            return

        for i in range(len(key_strs)):
            if (key_strs[i] is None
                or buffer_ptrs[i] is None
                or buffer_sizes[i] is None):
                return

        return self._get_batch_zero_copy_impl(key_strs, buffer_ptrs, buffer_sizes)

    def is_batch_exist(self, keys: List[str]):
        _keys = []
        for key in keys:
            if key is None:
                return None
            # Since mooncake store is stored in layer by layer,
            # only the first layer is checked here.
            _keys.append(f"{key}_0_k")
        return {k: v for k, v in zip(keys, self.store.batch_is_exist(_keys))}

    def _put_batch_zero_copy_impl(
        self,
        key_strs: List[str],
        buffer_ptrs: List[int],
        buffer_sizes: List[int]
    ) -> None:
        try:
            self.store.batch_put_from(key_strs, buffer_ptrs, buffer_sizes)
        except TypeError as err:
            logger.error("Failed to put value to Mooncake Store: %s", err)
            raise TypeError("Mooncake Store Put Type Error.") from err

    def _get_batch_zero_copy_impl(
        self,
        key_strs: List[str],
        buffer_ptrs: List[int],
        buffer_sizes: List[int]
    ) -> None:
        try:
            logger.info("batch get into called from python")
            self.store.batch_get_into(key_strs, buffer_ptrs, buffer_sizes)
        except TypeError as err:
            logger.error("Failed to get value from Mooncake Store: %s", err)
            raise TypeError("Mooncake Store Get Type Error.") from err
