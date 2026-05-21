# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
from typing import Generator, List, Optional, Tuple
from urllib.parse import urlparse

import torch

from sglang.srt.connector.base_connector import BaseKVConnector
from sglang.srt.connector.utils import pull_files_from_db

logger = logging.getLogger(__name__)

DEFAULT_LOCAL_BUFFER_SIZE = 16 * 1024 * 1024  # 16 MB
FILE_INDEX_KEY_PREFIX = "__index__"
STANDALONE_CHUNK_SIZE = 8 * 1024 * 1024


def _tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().to("cpu").contiguous().view(torch.uint8).numpy().tobytes()


def _copy_bytes_into_tensor(data: bytes, tensor: torch.Tensor) -> None:
    tensor_size = tensor.untyped_storage().nbytes()
    if len(data) != tensor_size:
        raise RuntimeError(
            f"Expected {tensor_size} bytes for tensor load, got {len(data)}"
        )

    source = torch.frombuffer(bytearray(data), dtype=torch.uint8)
    target = tensor.view(torch.uint8).reshape(-1)
    if target.device.type == "cpu":
        target.copy_(source)
    else:
        target.copy_(source.to(target.device))


def _chunk_manifest_key(key: str) -> str:
    return f"{key}.__chunk_manifest__"


def _chunk_data_key(key: str, chunk_idx: int) -> str:
    return f"{key}.__chunk__.{chunk_idx}"


def _check_batch_get_results(
    keys: List[str], results: List[int], tensor_sizes: List[int]
) -> None:
    if len(results) != len(keys):
        raise RuntimeError(
            "Mooncake batch_get_into returned %d results for %d keys"
            % (len(results), len(keys))
        )

    failures = []
    for key, result, tensor_size in zip(keys, results, tensor_sizes):
        if result < 0:
            failures.append(f"{key}: error={result}")
        elif result != tensor_size:
            failures.append(f"{key}: expected {tensor_size} bytes, got {result}")

    if failures:
        raise RuntimeError(
            "Mooncake batch_get_into failed for some tensors: " + "; ".join(failures)
        )


def _check_batch_put_results(keys: List[str], results: List[int]) -> None:
    if len(results) != len(keys):
        raise RuntimeError(
            "Mooncake batch_put_from returned %d results for %d keys"
            % (len(results), len(keys))
        )

    failures = [
        f"{key}: error={result}" for key, result in zip(keys, results) if result != 0
    ]
    if failures:
        raise RuntimeError(
            "Mooncake batch_put_from failed for some tensors: " + "; ".join(failures)
        )


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
        if self.config.standalone_storage:
            if not self.config.client_server_address:
                raise ValueError(
                    "MOONCAKE_CLIENT must be set when MOONCAKE_STANDALONE_STORAGE is enabled."
                )
            ret_code = self.store.setup_dummy(
                self.config.global_segment_size,
                DEFAULT_LOCAL_BUFFER_SIZE,
                self.config.client_server_address,
            )
        else:
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

    def _key_exists(self, key: str) -> bool:
        ret_code = self.store.is_exist(key)
        if ret_code < 0:
            raise RuntimeError(
                f"Failed to query Mooncake key '{key}', error code: {ret_code}"
            )
        return ret_code == 1

    def batch_get_into(self, keys: List[str], tensors: List[torch.Tensor]) -> None:
        full_keys = [f"{self.model_name}/{key}" for key in keys]

        if self.config.standalone_storage:
            for full_key, tensor in zip(full_keys, tensors):
                if self._key_exists(full_key):
                    data = self.store.get(full_key)
                else:
                    manifest_key = _chunk_manifest_key(full_key)
                    if not self._key_exists(manifest_key):
                        raise RuntimeError(f"Mooncake key not found: {full_key}")

                    manifest = json.loads(self.store.get(manifest_key).decode("utf-8"))
                    data = b"".join(
                        self.store.get(_chunk_data_key(full_key, chunk_idx))
                        for chunk_idx in range(manifest["num_chunks"])
                    )
                _copy_bytes_into_tensor(data, tensor)
            return

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

        results = self.store.batch_get_into(full_keys, tensor_ptrs, tensor_sizes)
        _check_batch_get_results(full_keys, results, tensor_sizes)

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
        if not self._key_exists(key):
            logger.error("Key %s not found in Mooncake store", key)
            return None

        data = self.store.get(key)
        return data.decode("utf-8")

    def set(self, key: str, tensor: torch.Tensor) -> None:
        raise NotImplementedError("Use batch_put_from() instead for performance.")

    def batch_put_from(self, keys: List[str], tensors: List[torch.Tensor]) -> None:
        if self.config.standalone_storage:
            failures = []
            for key, tensor in zip(keys, tensors):
                data = _tensor_to_bytes(tensor)
                if len(data) <= STANDALONE_CHUNK_SIZE:
                    ret_code = self.store.put(key, data, self._rep_config)
                    if ret_code != 0:
                        failures.append(f"{key}: error={ret_code}")
                    continue

                num_chunks = (
                    len(data) + STANDALONE_CHUNK_SIZE - 1
                ) // STANDALONE_CHUNK_SIZE
                for chunk_idx in range(num_chunks):
                    start = chunk_idx * STANDALONE_CHUNK_SIZE
                    end = start + STANDALONE_CHUNK_SIZE
                    ret_code = self.store.put(
                        _chunk_data_key(key, chunk_idx),
                        data[start:end],
                        self._rep_config,
                    )
                    if ret_code != 0:
                        failures.append(
                            f"{_chunk_data_key(key, chunk_idx)}: error={ret_code}"
                        )
                        break
                else:
                    ret_code = self.store.put(
                        _chunk_manifest_key(key),
                        json.dumps(
                            {"num_chunks": num_chunks, "size": len(data)}
                        ).encode("utf-8"),
                        self._rep_config,
                    )
                    if ret_code != 0:
                        failures.append(f"{_chunk_manifest_key(key)}: error={ret_code}")
            if failures:
                raise RuntimeError(
                    "Mooncake put failed for some tensors: " + "; ".join(failures)
                )
            return

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
        _check_batch_put_results(keys, results)

    def setstr(self, key: str, obj: str) -> None:
        ret_code = self.store.put(key, obj.encode("utf-8"), self._rep_config)
        if ret_code != 0:
            raise RuntimeError(
                f"Failed to put string key '{key}' into Mooncake store, error code: {ret_code}"
            )

        prefix, _, _ = key.rpartition("/")
        self._register_key_in_index(key, f"{prefix}/")

    def list(self, prefix: str) -> List[str]:
        """
        Mooncake store does not expose a native scan/list API, so we rely on
        a sentinel index key that stores a newline-separated list of keys
        written under that prefix.
        """
        index_key = f"{FILE_INDEX_KEY_PREFIX}{prefix}"
        if not self._key_exists(index_key):
            return []

        data = self.store.get(index_key)
        content = data.decode("utf-8").strip()
        if not content:
            return []
        return content.split("\n")

    def _register_key_in_index(self, key: str, prefix: str) -> None:
        """Maintain a simple index for list() support."""
        index_key = f"{FILE_INDEX_KEY_PREFIX}{prefix}"
        if not self._key_exists(index_key):
            keys_list = [key]
        else:
            existing = self.store.get(index_key)
            keys_list = existing.decode("utf-8").strip().split("\n")
            if key not in keys_list:
                keys_list.append(key)
        ret_code = self.store.put(
            index_key, "\n".join(keys_list).encode("utf-8"), self._rep_config
        )
        if ret_code != 0:
            raise RuntimeError(
                f"Failed to update Mooncake file index '{index_key}', error code: {ret_code}"
            )

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
