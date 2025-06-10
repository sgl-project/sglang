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
from typing import Optional, List

import numpy as np
import torch
from safetensors.torch import load as safetensors_load
from safetensors.torch import save as safetensors_save

DEFAULT_GLOBAL_SEGMENT_SIZE = 0  # 3.125 GiB
DEFAULT_LOCAL_BUFFER_SIZE = 1024 * 1024 * 1024  # 1.0 GiB

logger = logging.getLogger(__name__)

def tensor_to_bytes(tensor):
    tensor = tensor.to("cpu")
    length = int(np.prod(tensor.shape).item())
    bytes_per_item = 2 # for bfloat16, other will change accordingly
    total_bytes = length * bytes_per_item
    ptr = tensor.data_ptr()
    new_ptr = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_ubyte))
    data = np.ctypeslib.as_array(new_ptr, (total_bytes,))  # no internal copy
    return data.tobytes()

def bytes_to_tensor(data_bytes, tensor_shape):
    return torch.frombuffer(data_bytes, dtype=torch.bfloat16).reshape(tensor_shape)

def safetensors_bytes_to_tensor(data: bytes):
    loaded_tensors = safetensors_load(data)
    tensor = loaded_tensors["tensor"]
    if "device_id" not in loaded_tensors.keys():
        return tensor
    device_id_tensor = loaded_tensors["device_id"]
    device_id = int(device_id_tensor.item())
    device = torch.device(
        'cuda', device_id) if device_id >= 0 else torch.device('cpu')
    return tensor.to(device)

def safetensors_tensor_to_bytes(tensor_value: torch.Tensor):
    tensors = {"tensor": tensor_value}
    device_id = tensor_value.device.index if tensor_value.device.type == 'cuda' else -1
    if device_id != -1:
        device_tensor = torch.tensor(device_id, dtype=torch.int32)
        tensors['device_id'] = device_tensor
    return safetensors_save(tensors)

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
    def from_file(file_path: str) -> 'MooncakeStoreConfig':
        """Load the config from a JSON file."""
        with open(file_path) as fin:
            config = json.load(fin)
        return MooncakeStoreConfig(
            local_hostname=config.get("local_hostname"),
            metadata_server=config.get("metadata_server"),
            global_segment_size=config.get("global_segment_size",
                                           DEFAULT_GLOBAL_SEGMENT_SIZE),
            local_buffer_size=config.get("local_buffer_size",
                                         DEFAULT_LOCAL_BUFFER_SIZE),
            protocol=config.get("protocol", "tcp"),
            device_name=config.get("device_name", ""),
            master_server_address=config.get("master_server_address"),
        )

    @staticmethod
    def load_from_env() -> 'MooncakeStoreConfig':
        """Load config from a file specified in the environment variable."""
        config_file_path = os.getenv('MOONCAKE_CONFIG_PATH')
        if config_file_path is None:
            raise ValueError(
                "The environment variable 'MOONCAKE_CONFIG_PATH' is not set.")
        return MooncakeStoreConfig.from_file(config_file_path)


class MooncakeStore:

    def __init__(
        self,
        page_tensor_shape
    ):
        try:
            from mooncake.store import MooncakeDistributedStore
        except ImportError as e:
            raise ImportError(
                "Please install mooncake by following the instructions at "
                "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "  # noqa: E501
                "to run vLLM with MooncakeConnector.") from e

        try:
            self.store = MooncakeDistributedStore()
            self.config = MooncakeStoreConfig.load_from_env()
            logger.info("Mooncake Configuration loaded successfully.")

            self.store.setup(self.config.local_hostname,
                             self.config.metadata_server,
                             self.config.global_segment_size,
                             self.config.local_buffer_size,
                             self.config.protocol,
                             self.config.device_name,
                             self.config.master_server_address)
            logger.info("Connect to Mooncake store successfully.")
            self.warmup()
            logger.info("Mooncake store warmup successfully.")
            self.page_tensor_shape = page_tensor_shape

        except ValueError as e:
            logger.error("Configuration loading failed: %s", e)
            raise
        except Exception as exc:
            logger.error(
                "An error occurred while loading the configuration: %s", exc)
            raise

    def warmup(self):
        warmup_key = "sglang_mooncake_store_warmup_key" + uuid.uuid4().hex
        # 10 MB
        warmup_value = bytes(10 * 1024 * 1024)
        self.store.put(warmup_key, warmup_value)
        assert self.store.is_exist(warmup_key) == 1
        self.store.get(warmup_key)
        self.store.remove(warmup_key)

    def close(self):
        # MooncakeDistributedStore will automatically call the destructor, so
        # it is unnecessary to close it manually.
        pass

    def remove(
        self,
        key: str,
    ) -> None:
        if key is not None:
            if self.is_exist(key):
                self.store.remove(key)

    def put(
        self,
        key: str,
        value: Optional[torch.Tensor]
    ) -> None:
        # A message queue needs to be introduced before making it asynchronous.
        if value is not None:
            self._put_impl(key, value)

    def batch_put(
        self,
        keys: List[str],
        values: List[torch.Tensor]
    ) -> None:
        if keys is None or values is None:
            return

        if len(keys) != len(values):
            return

        for i in range(len(keys)):
            if keys[i] is None or values[i] is None:
                return

        self._put_batch_impl(keys, values)

    def get(
        self,
        key: str,
    ) -> Optional[torch.Tensor]:
        # A message queue needs to be introduced before making it asynchronous.
        value = self._get_impl(key)
        return value

    def batch_get(
        self,
        keys: List[str],
    ) -> Optional[List[torch.Tensor]]:
        if keys is None:
            return None

        for key in keys:
            if key is None :
                return None

        return self._get_batch_impl(keys)

    def is_exist(self,
                 key: str
    ) -> bool:
        if key is not None:
            return self.store.is_exist(key) == 1
        return False

    def _put_batch_impl(
        self,
        keys: List[str],
        values: List[torch.Tensor]
    ) -> None:
        value_bytes = [tensor_to_bytes(value) for value in values]
        batches = {}
        for i in range(len(keys)):
            batches[keys[i]] = value_bytes[i]
        try:
            self.store.put_batch(batches)
        except TypeError as err:
            logger.error("Failed to put value into Mooncake Store: %s", err)
            raise TypeError("Mooncake Store Put Type Error.") from err

    def _put_impl(
        self,
        key: str,
        value: torch.Tensor
    ) -> None:
        """Put KVCache to Mooncake Store"""
        value_bytes = tensor_to_bytes(value)
        try:
            self.store.put(key, value_bytes)
        except TypeError as err:
            logger.error("Failed to put value into Mooncake Store: %s", err)
            raise TypeError("Mooncake Store Put Type Error.") from err

    def _get_batch_impl(
        self,
        keys: List[str]
    ) -> Optional[List[torch.Tensor]]:
        try:
            batch_data = self.store.get_batch(keys)
        except TypeError as err:
            logger.error("Failed to get value from Mooncake Store: %s", err)
            raise TypeError("Mooncake Store Get Type Error.") from err

        if batch_data:
            if len(batch_data) > 0:
                return [bytes_to_tensor(data, self.page_tensor_shape) for data in batch_data]
        return None

    def _get_impl(
        self,
        key: str,
    ) -> Optional[torch.Tensor]:
        """Get KVCache from Mooncake Store"""
        try:
            data = self.store.get(key)
        except TypeError as err:
            logger.error("Failed to get value from Mooncake Store: %s", err)
            raise TypeError("Mooncake Store Get Type Error.") from err

        if data:
            return bytes_to_tensor(data, self.page_tensor_shape)

        return None
