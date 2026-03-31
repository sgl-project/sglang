# SPDX-License-Identifier: Apache-2.0

import logging
import os
from typing import Generator, List, Optional, Tuple
from urllib.parse import urlparse

import torch

from sglang.srt.connector.base_connector import BaseKVConnector
from sglang.srt.connector.utils import pull_files_from_db

logger = logging.getLogger(__name__)

DEFAULT_LOCAL_BUFFER_SIZE = 16 * 1024 * 1024  # 16 MB


class MooncakeStoreConnector(BaseKVConnector):
    """
    A KV connector backed by MooncakeDistributedStore.

    URL format:
        mooncake:///<model_name>

    The connector uses MooncakeStoreConfig (loaded from environment variables or
    config file) to set up the underlying MooncakeDistributedStore, and stores /
    retrieves torch.Tensor values serialized as raw bytes.
    """

    def __init__(self, url: str):
        try:
            from mooncake.store import MooncakeDistributedStore, ReplicateConfig
        except ImportError as e:
            raise ImportError(
                "Please install mooncake by following the instructions at "
                "https://kvcache-ai.github.io/Mooncake/getting_started/build.html "
                "to run SGLang with MooncakeStoreConnector."
            ) from e

        super().__init__(url)
        parsed_url = urlparse(url)
        self.model_name = parsed_url.path.lstrip("/")

        # Load mooncake config from env / file
        from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
            MooncakeStoreConfig,
        )

        self.config = MooncakeStoreConfig.load_from_env()

        # Build the store
        self.store = MooncakeDistributedStore()
        ret_code = self.store.setup(
            self.config.local_hostname,
            self.config.metadata_server,
            self.config.global_segment_size,
            DEFAULT_LOCAL_BUFFER_SIZE,
            self.config.protocol,
            self.config.device_name,
            self.config.master_server_address,
        )
        if ret_code:
            raise RuntimeError(
                f"Failed to setup Mooncake store, error code: {ret_code}"
            )

        self._rep_config = ReplicateConfig()
        self._rep_config.replica_num = 1

        preferred_segment = os.getenv("MOONCAKE_CONNECTOR_PREFERRED_SEGMENTS")
        if preferred_segment is not None:
            self._rep_config.preferred_segments = preferred_segment.split(",")

        logger.info("MooncakeStoreConnector initialized successfully.")

    # ------------------------------------------------------------------
    # BaseKVConnector interface
    # ------------------------------------------------------------------

    def batch_get_into(self, keys: List[str], tensors: List[torch.Tensor]) -> None:
        tensor_ptrs = [tensor.data_ptr() for tensor in tensors]
        tensor_sizes = [tensor.untyped_storage().nbytes() for tensor in tensors]

        for tensor in tensors:
            ret_code = self.store.register_buffer(
                tensor.data_ptr(), tensor.untyped_storage().nbytes()
            )
            if ret_code:
                raise RuntimeError(
                    f"Failed to register buffer to Mooncake Store, error code: {ret_code}"
                )

        results = self.store.batch_get_into(
            [f"{self.model_name}/{key}" for key in keys], tensor_ptrs, tensor_sizes
        )
        print(results)

    def get_into(self, key: str, tensor: torch.Tensor) -> None:
        tensor_size = tensor.untyped_storage().nbytes()
        tensor_ptr = tensor.data_ptr()

        ret_code = self.store.register_buffer(tensor_ptr, tensor_size)
        if ret_code:
            raise RuntimeError(
                f"Failed to register buffer to Mooncake Store, error code: {ret_code}"
            )

        self.store.batch_get_into(
            [f"{self.model_name}/{key}"], [tensor_ptr], [tensor_size]
        )

    def get(self, key: str) -> Optional[torch.Tensor]:
        raise NotImplementedError("Use batch_get_into() instead for performance.")

    def getstr(self, key: str) -> Optional[str]:
        data = self.store.get(key)
        if data is None:
            logger.error("Key %s not found in Mooncake store", key)
            return None
        return data.decode("utf-8")

    def set(self, key: str, tensor: torch.Tensor) -> None:
        raise NotImplementedError("Use batch_put_from() instead for performance.")

    def batch_put_from(self, keys: List[str], tensors: List[torch.Tensor]) -> None:
        tensor_ptrs = [tensor.data_ptr() for tensor in tensors]
        tensor_sizes = [tensor.untyped_storage().nbytes() for tensor in tensors]

        for tensor in tensors:
            ret_code = self.store.register_buffer(
                tensor.data_ptr(), tensor.untyped_storage().nbytes()
            )
            if ret_code:
                raise RuntimeError(
                    f"Failed to register buffer to Mooncake Store, error code: {ret_code}"
                )

        results = self.store.batch_put_from(
            keys, tensor_ptrs, tensor_sizes, self._rep_config
        )
        print(results)

    def setstr(self, key: str, obj: str) -> None:
        ret_code = self.store.put(key, obj.encode("utf-8"), self._rep_config)
        if ret_code != 0:
            raise RuntimeError(
                f"Failed to put string key '{key}' into Mooncake store, error code: {ret_code}"
            )

    def list(self, prefix: str) -> List[str]:
        """
        Mooncake store does not expose a native scan/list API, so we rely on
        a sentinel index key that stores a newline-separated list of keys
        written under that prefix.
        """
        # index_key = f"__index__{prefix}"
        # data = self.store.get(index_key)
        # if data is None:
        #     return []
        # content = data.decode("utf-8").strip()
        # if not content:
        #     return []
        # return content.split("\n")
        return [
            f"{prefix}{key}"
            for key in [
                "config.json",
                "vocab.json",
                "tokenizer_config.json",
                "model.safetensors.index.json",
                "generation_config.json",
                "tokenizer.json",
            ]
        ]

    def _register_key_in_index(self, key: str, prefix: str) -> None:
        """Maintain a simple index for list() support."""
        index_key = f"__index__{prefix}"
        existing = self.store.get(index_key)
        if existing is None:
            keys_list = [key]
        else:
            keys_list = existing.decode("utf-8").strip().split("\n")
            if key not in keys_list:
                keys_list.append(key)
        self.store.put(index_key, "\n".join(keys_list).encode("utf-8"))

    # ------------------------------------------------------------------
    # BaseConnector interface
    # ------------------------------------------------------------------

    def weight_iterator(
        self, rank: int = 0
    ) -> Generator[Tuple[str, torch.Tensor], None, None]:
        raise NotImplementedError(
            "MooncakeStoreConnector does not support iterating weights one by one. "
            "Please use a loading path that leverages `batch_get_into`."
        )

    def pull_files(
        self,
        allow_pattern: Optional[List[str]] = None,
        ignore_pattern: Optional[List[str]] = None,
    ) -> None:
        pull_files_from_db(self, self.model_name, allow_pattern, ignore_pattern)

    def close(self):
        # MooncakeDistributedStore cleans up via its destructor
        super().close()
