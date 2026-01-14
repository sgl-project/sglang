import logging
import os
import time
import uuid
from typing import Any, List, Optional, Union

import torch

from sglang.srt.mem_cache.hicache_storage import HiCacheStorage, HiCacheStorageConfig

from .nixl_utils import NixlBackendSelection, NixlFileManager, NixlRegistration

try:
    from nixl._api import nixl_agent, nixl_agent_config
except ImportError as e:
    raise ImportError(
        "Please install NIXL by following the instructions at "
        "https://github.com/ai-dynamo/nixl/blob/main/README.md "
        "to use HiCacheNixl storage backend."
    ) from e

logger = logging.getLogger(__name__)


class HiCacheNixl(HiCacheStorage):
    """HiCacheNixl provides high-performance storage using NIXL plugins."""

    def __init__(
        self,
        storage_config: HiCacheStorageConfig,
        file_path: str = "/tmp/hicache_storage",
        plugin: str = "auto",
    ):
        """Initialize NIXL storage connector."""
        # Might be better to be unified across HiCache backends and moved to HiCacheController
        file_path = os.getenv("SGLANG_HICACHE_NIXL_BACKEND_STORAGE_DIR", file_path)
        self.file_manager = (
            NixlFileManager(file_path)
            if plugin not in NixlBackendSelection.OBJ_PLUGINS
            else None
        )

        # Initialize suffix based on storage config
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

        agent_config = nixl_agent_config(backends=[])
        self.agent_name = f"hicache_nixl_{str(uuid.uuid4())}"
        self.agent = nixl_agent(self.agent_name, agent_config)

        self.backend_selector = NixlBackendSelection(plugin)
        if not self.backend_selector.create_backend(self.agent):
            raise RuntimeError("Failed to create NIXL backend")

        self.registration = NixlRegistration(self.agent)

    def _get_suffixed_key(self, key: str) -> str:
        return key + self.config_suffix

    def register_buffers(
        self, buffers: Union[torch.Tensor, List[torch.Tensor], List[tuple]]
    ) -> Optional[Any]:
        """Register tensor(s) or target locations in host memory (list of addr,len tuples) with NIXL."""
        if isinstance(buffers[0], tuple):
            tuples = [(x[0], x[1], 0, "") for x in buffers]
            return self.registration._register_memory(tuples, "DRAM")
        else:
            return self.registration._register_memory(buffers)

    def register_files(
        self, file_paths: List[str], open_file: Optional[bool] = True
    ) -> Optional[Any]:
        """Register files with NIXL."""
        tuples = self.file_manager.files_to_nixl_tuples(file_paths)
        return self.registration._register_memory(tuples, "FILE")

    def register_objects(
        self, keys: List[str], sizes: Optional[List[int]] = None
    ) -> Optional[Any]:
        """Register objects with NIXL."""
        if not keys:
            return None
        tuples = [(0, 0, key, "") for key in keys]
        return self.registration._register_memory(tuples, "OBJ")

    def _execute_transfer(
        self,
        buffers: Optional[List[torch.Tensor | tuple]],
        keys: List[str],
        direction: str,
    ) -> bool:
        if len(buffers) != len(keys):
            logger.error("Mismatch between number of tensors/buffers and files/objects")
            return False

        # Registering file and object keys per transfer, to be updated when
        # pre-registration for file and object is added to HiCache.
        if self.backend_selector.mem_type == "FILE":
            tuples = self.file_manager.files_to_nixl_tuples(keys)
            if not tuples or not self.registration._register_memory(tuples, "FILE"):
                logger.error("Failed to prepare files for transfer")
                return False
        else:  # mem_type == "OBJ"
            tuples = [(0, 0, key, "") for key in keys]
            if not tuples or not self.registration._register_memory(tuples, "OBJ"):
                logger.error("Failed to register objects")
                return False

        # Prepare transfer descriptors
        if isinstance(buffers[0], torch.Tensor):
            tensor_sizes = [
                tensor.element_size() * tensor.numel() for tensor in buffers
            ]
            storage_tuples = [(x[0], s, x[2]) for x, s in zip(tuples, tensor_sizes)]
            host_descs = self.agent.get_xfer_descs(buffers)
        elif isinstance(buffers[0], tuple):
            storage_tuples = [(x[0], y[1], x[2]) for x, y in zip(tuples, buffers)]
            host_descs = self.agent.get_xfer_descs(
                [(x[0], x[1], 0) for x in buffers], "DRAM"
            )
        else:
            return False

        storage_descs = self.agent.get_xfer_descs(
            storage_tuples, self.backend_selector.mem_type
        )

        if (host_descs is None) or (storage_descs is None):
            logger.error("Failed to get transfer descriptors")
            return False

        # Initialize transfer, default assumption that tensor was registered
        try:
            xfer_req = self.agent.initialize_xfer(
                direction, host_descs, storage_descs, self.agent_name
            )
        except Exception:
            # Check if it was due to missing pre-registration
            if not self.register_buffers(buffers):
                logger.error("Failed to register tensors/buffers")
                return False

            try:
                xfer_req = self.agent.initialize_xfer(
                    direction, host_descs, storage_descs, self.agent_name
                )
            except Exception as e:
                logger.error(f"Failed to create transfer request: {e}")
                return False

        # Execute transfer and wait for its completion
        try:
            state = self.agent.transfer(xfer_req)
            while state != "DONE":
                state = self.agent.check_xfer_state(xfer_req)
                if state == "ERR":
                    self.agent.release_xfer_handle(xfer_req)
                    logger.error("Transfer failed")
                    return False
                time.sleep(0.0001)  # Can be changed to os.sched_yield() or parametrized

            self.agent.release_xfer_handle(xfer_req)
            return True

        except Exception as e:
            logger.error(f"Failed to execute transfer: {e}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            return False

    def get(
        self,
        key: str,
        target_location: Optional[torch.Tensor | int] = None,
        target_sizes: Optional[int] = None,
    ) -> torch.Tensor | None:
        # To be removed, being compatible with the current API
        if target_location is None:
            return None
        if target_sizes:
            result = self.batch_get([key], [target_location], [target_sizes])
        else:
            result = self.batch_get([key], [target_location])
        return result[0] if result else None

    def batch_get(
        self,
        keys: List[str],
        target_locations: Optional[List[torch.Tensor | int]] = None,
        target_sizes: Optional[List[int]] = None,
    ) -> List[torch.Tensor | None]:
        if not keys:
            return []

        # To be removed, being compatible with the current API
        if not target_locations:
            return [None] * len(keys)

        if target_sizes and (len(target_sizes) != len(target_locations)):
            logger.error("Mismatch between number of target_locations and target_sizes")
            return [None] * len(keys)
        if target_sizes:
            dest = list(zip(target_locations, target_sizes))
        else:
            dest = target_locations

        # Add suffix to keys
        suffixed_keys = [self._get_suffixed_key(key) for key in keys]

        if self.backend_selector.mem_type == "FILE":
            file_paths = [self.file_manager.get_file_path(key) for key in suffixed_keys]
            success = self._execute_transfer(dest, file_paths, "READ")
        else:
            success = self._execute_transfer(dest, suffixed_keys, "READ")
        return target_locations if success and not target_sizes else [None] * len(keys)

    def set(
        self,
        key: str,
        value: Optional[torch.Tensor] = None,
        target_location: Optional[int] = None,
        target_sizes: Optional[int] = None,
    ) -> bool:
        if target_location and target_sizes:
            return self.batch_set([key], None, [target_location], [target_sizes])
        else:
            return self.batch_set([key], [value])

    def batch_set(
        self,
        keys: List[str],
        values: Optional[List[torch.Tensor]] = None,
        target_locations: Optional[List[int]] = None,
        target_sizes: Optional[List[int]] = None,
    ) -> bool:
        if not keys or (not values and (not target_locations or not target_sizes)):
            logger.error("Keys or values were not passed")
            return False

        if not values:
            values = list(zip(target_locations, target_sizes))

        # Add suffix to keys
        suffixed_keys = [self._get_suffixed_key(key) for key in keys]

        if self.backend_selector.mem_type == "FILE":
            file_paths = []
            for key in suffixed_keys:
                file_path = self.file_manager.get_file_path(key)
                # New file per set, to be updated when partial writes is added to HiCache
                if not self.file_manager.create_file(file_path):
                    logger.error(f"Failed to create file {file_path}")
                    return False
                file_paths.append(file_path)
            return self._execute_transfer(values, file_paths, "WRITE")
        else:  # mem_type == "OBJ"
            return self._execute_transfer(values, suffixed_keys, "WRITE")

    def exists(self, key: str) -> bool:
        # Add suffix to key
        suffixed_key = self._get_suffixed_key(key)

        tuples = self.registration.create_query_tuples(
            suffixed_key,
            self.backend_selector.mem_type,
            self.file_manager if self.backend_selector.mem_type == "FILE" else None,
        )
        if not tuples:
            return False

        query_res = self.agent.query_memory(
            tuples,
            self.backend_selector.backend_name,
            mem_type=self.backend_selector.mem_type,
        )
        return query_res[0] is not None  # can be expanded to multiple keys
