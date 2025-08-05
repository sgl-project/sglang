import hashlib
import json
import logging
import os
import uuid
from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np
import torch

from sglang.srt.distributed import get_tensor_model_parallel_rank
from sglang.srt.mem_cache.hicache_storage import HiCacheStorage

DEFAULT_GLOBAL_SEGMENT_SIZE = 4 * 1024 * 1024 * 1024  # 4 GiB
DEFAULT_LOCAL_BUFFER_SIZE = 128 * 1024 * 1024  # 128 MB

logger = logging.getLogger(__name__)


def get_hash_str_mooncake(current_page_ids: List, prefix_block_key: str):
    local_rank = get_tensor_model_parallel_rank()
    prefix_str = ""
    if prefix_block_key:
        if len(prefix_block_key):
            prefix_str = hashlib.sha256(prefix_block_key.encode()).hexdigest()
    current_token_ids_bytes = np.array(current_page_ids).tobytes()
    current_hash_object = hashlib.sha256(current_token_ids_bytes)
    current_hash_hex = current_hash_object.hexdigest()
    return f"{prefix_str}_{int(current_hash_hex[:16], 16)}_{local_rank}"


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
    def from_file() -> "MooncakeStoreConfig":
        """Load the config from a JSON file."""
        file_path = os.getenv("MOONCAKE_CONFIG_PATH")
        if file_path is None:
            raise ValueError(
                "The environment variable 'MOONCAKE_CONFIG_PATH' is not set."
            )
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
            device_name=config.get("device_name", "auto"),
            master_server_address=config.get("master_server_address"),
        )

    @staticmethod
    def load_from_env() -> "MooncakeStoreConfig":
        """Load config from a file specified in the environment variable.
        export MOONCAKE_MASTER=10.13.3.232:50051
        export MOONCAKE_PROTOCOL="rdma"
        export MOONCAKE_DEVICE="auto"
        export MOONCAKE_TE_META_DATA_SERVER="P2PHANDSHAKE"
        """
        # other required environment variables...
        if not os.getenv("MOONCAKE_MASTER"):
            raise ValueError("The environment variable 'MOONCAKE_MASTER' is not set.")
        return MooncakeStoreConfig(
            local_hostname=os.getenv("LOCAL_HOSTNAME", "localhost"),
            metadata_server=os.getenv("MOONCAKE_TE_META_DATA_SERVER", "P2PHANDSHAKE"),
            global_segment_size=int(
                os.getenv("MOONCAKE_GLOBAL_SEGMENT_SIZE", DEFAULT_GLOBAL_SEGMENT_SIZE)
            ),
            local_buffer_size=int(
                os.getenv("MOONCAKE_LOCAL_BUFFER_SIZE", DEFAULT_LOCAL_BUFFER_SIZE)
            ),
            protocol=os.getenv("MOONCAKE_PROTOCOL", "tcp"),
            device_name=os.getenv("MOONCAKE_DEVICE", "auto"),
            master_server_address=os.getenv("MOONCAKE_MASTER"),
        )

    def __post_init__(self):
        if self.device_name == "auto":
            os.environ["MC_MS_AUTO_DISC"] = "1"
            os.environ["MC_MS_FILTERS"] = (
                "mlx5_bond_0, mlx5_bond_1, mlx5_bond_2, mlx5_bond_3"
            )


class MooncakeStore(HiCacheStorage):
    def __init__(self):
        try:
            from mooncake.store import MooncakeDistributedStore
        except ImportError as e:
            raise ImportError(
                "Please install mooncake by following the instructions at "
                "https://kvcache-ai.github.io/Mooncake/getting_started/build.html"
                "to run SGLang with MooncakeConnector."
            ) from e

        try:
            self.store = MooncakeDistributedStore()
            self.config = MooncakeStoreConfig.load_from_env()
            logger.info("Mooncake Configuration loaded from env successfully.")

            ret_code = self.store.setup(
                self.config.local_hostname,
                self.config.metadata_server,
                self.config.global_segment_size,
                self.config.local_buffer_size,
                self.config.protocol,
                self.config.device_name,
                self.config.master_server_address,
            )
            if ret_code:
                logger.error(f"failed to setup mooncake store, error code: {ret_code}")

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

    def register_buffer(self, buffer: torch.Tensor) -> None:
        try:
            buffer_ptr = buffer.data_ptr()
            buffer_size = buffer.numel() * buffer.element_size()
            ret_code = self.store.register_buffer(buffer_ptr, buffer_size)
            if ret_code:
                logger.error(f"failed to register buffer, error code: {ret_code}")
        except TypeError as err:
            logger.error("Failed to register buffer to Mooncake Store: %s", err)
            raise TypeError("Mooncake Store Register Buffer Error.") from err

    def set(
        self,
        key,
        value: Optional[Any] = None,
        target_location: Optional[List[int]] = None,
        target_sizes: Optional[List[int]] = None,
    ) -> bool:
        assert len(key) == len(target_location) == len(target_sizes)
        if len(key) == 0:
            return

        for i in range(len(key)):
            if key[i] is None or target_location[i] is None or target_sizes[i] is None:
                return

        self._put_batch_zero_copy_impl(key, target_location, target_sizes)

    def batch_set(
        self,
        keys: List[str],
        value: Optional[Any] = None,
        target_location: Optional[List[int]] = None,
        target_sizes: Optional[List[int]] = None,
    ) -> bool:
        assert len(keys) == len(target_location) == len(target_sizes)
        if len(keys) == 0:
            return

        for i in range(len(keys)):
            if keys[i] is None or target_location[i] is None or target_sizes[i] is None:
                return

        self._put_batch_zero_copy_impl(keys, target_location, target_sizes)

    def get(
        self,
        key,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> torch.Tensor | None:
        assert len(key) == len(target_location) == len(target_sizes)
        if len(key) == 0:
            return

        for i in range(len(key)):
            if key[i] is None or target_location[i] is None or target_sizes[i] is None:
                return

        return self._get_batch_zero_copy_impl(key, target_location, target_sizes)

    def batch_get(
        self,
        keys: List[str],
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> torch.Tensor | None:
        assert len(keys) == len(target_location) == len(target_sizes)
        if len(keys) == 0:
            return

        for i in range(len(keys)):
            if keys[i] is None or target_location[i] is None or target_sizes[i] is None:
                return

        return self._get_batch_zero_copy_impl(keys, target_location, target_sizes)

    def exists(self, keys) -> bool | dict:
        _keys = []
        local_rank = torch.cuda.current_device()
        for key in keys:
            if key is None:
                return None
            # Since mooncake store is stored in layer by layer,
            # only the first layer is checked here.
            _keys.append(f"{key}_{local_rank}_k")
        result = {k: v for k, v in zip(keys, self.store.batch_is_exist(_keys))}
        return result

    def delete(self, key) -> None:
        raise (NotImplementedError)

    def close(self):
        # MooncakeDistributedStore will automatically call the destructor, so
        # it is unnecessary to close it manually.
        pass

    def clear(self) -> None:
        raise (NotImplementedError)

    def _put_batch_zero_copy_impl(
        self, key_strs: List[str], buffer_ptrs: List[int], buffer_sizes: List[int]
    ) -> None:
        try:
            self.store.batch_put_from(key_strs, buffer_ptrs, buffer_sizes)
        except TypeError as err:
            logger.error("Failed to put value to Mooncake Store: %s", err)
            raise TypeError("Mooncake Store Put Type Error.") from err

    def _get_batch_zero_copy_impl(
        self, key_strs: List[str], buffer_ptrs: List[int], buffer_sizes: List[int]
    ) -> None:
        try:
            self.store.batch_get_into(key_strs, buffer_ptrs, buffer_sizes)
        except TypeError as err:
            logger.error("Failed to get value from Mooncake Store: %s", err)
            raise TypeError("Mooncake Store Get Type Error.") from err
