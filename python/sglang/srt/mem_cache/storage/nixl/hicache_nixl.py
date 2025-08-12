import hashlib
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from sglang.srt.mem_cache.hicache_storage import HiCacheStorage

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

    def __init__(self, file_path: str = "/tmp/hicache_storage", plugin: str = "auto"):
        """Initialize NIXL storage connector."""
        self.file_manager = (
            NixlFileManager(file_path)
            if plugin not in NixlBackendSelection.OBJ_PLUGINS
            else None
        )

        agent_config = nixl_agent_config(backends=[])
        self.agent_name = f"hicache_nixl_{str(uuid.uuid4())}"
        self.agent = nixl_agent(self.agent_name, agent_config)

        self.backend_selector = NixlBackendSelection(plugin)
        if not self.backend_selector.create_backend(self.agent):
            raise RuntimeError("Failed to create NIXL backend")

        self.registration = NixlRegistration(self.agent)

    def register_buffers(
        self, buffers: Union[torch.Tensor, List[torch.Tensor]]
    ) -> Optional[Any]:
        """Register tensor(s) with NIXL. Can be extended to tuple mode of registration."""
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
        self, tensors: List[torch.Tensor], keys: List[str], direction: str
    ) -> bool:
        if len(tensors) != len(keys):
            logger.error("Mismatch between number of tensors and files/objects")
            return False

        # Registering file and object keys per transfer, to be updated when
        # pre-registration for file and object is added to HiCache.
        if self.backend_selector.mem_type == "FILE":
            tuples = self.file_manager.files_to_nixl_tuples(keys)
            if not tuples or not self.registration._register_memory(tuples, "FILE"):
                logger.error("Failed to prepare files for transfer")
                return False
        else: # mem_type == "OBJ"
            tuples = [(0, 0, key, "") for key in keys]
            if not tuples or not self.registration._register_memory(tuples, "OBJ"):
                logger.error("Failed to register objects")
                return False

        # Prepare transfer tuples based on tensor sizes
        tensor_sizes = [tensor.element_size() * tensor.numel() for tensor in tensors]
        transfer_tuples = [(x[0], s, x[2]) for x, s in zip(tuples, tensor_sizes)]

        # Get transfer descriptors
        tensor_descs = self.agent.get_xfer_descs(tensors)
        storage_descs = self.agent.get_xfer_descs(transfer_tuples, self.backend_selector.mem_type)

        if (tensor_descs is None) or (storage_descs is None):
            logger.error("Failed to get transfer descriptors")
            return False

        # Initialize transfer, default assumption that tensor was registered
        try:
            xfer_req = self.agent.initialize_xfer(direction, tensor_descs, storage_descs, self.agent_name)
        except:
            # Check if it was due to missing pre-registration
            if not self.register_buffers(tensors):
                logger.error("Failed to register tensors")
                return False

            try:
                xfer_req = self.agent.initialize_xfer(direction, tensor_descs, storage_descs, self.agent_name)
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

    def set(self, key: str, value: torch.Tensor) -> bool:
        return self.batch_set([key], [value])

    def batch_set(self, keys: List[str], values: List[torch.Tensor]) -> bool:
        if not keys:
            return True

        if self.backend_selector.mem_type == "FILE":
            file_paths = []
            for key in keys:
                file_path = self.file_manager.get_file_path(key)
                # New file per set, to be updated when partial writes is added to HiCache
                if not self.file_manager.create_file(file_path):
                    logger.error(f"Failed to create file {file_path}")
                    return False
                file_paths.append(file_path)
            return self._execute_transfer(values, file_paths, "WRITE")
        else: # mem_type == "OBJ"
            return self._execute_transfer(values, keys, "WRITE")

    def get(
        self, key: str, dst_tensor: Optional[torch.Tensor] = None
    ) -> torch.Tensor | None:
        if dst_tensor is None:  # To be removed, being compatible with the current API
            return None
        result = self.batch_get([key], [dst_tensor])
        return result[0] if result else None

    def batch_get(
        self, keys: List[str], dst_tensors: List[torch.Tensor]
    ) -> List[Optional[torch.Tensor]]:
        if not keys:
            return []

        if self.backend_selector.mem_type == "FILE":
            file_paths = [self.file_manager.get_file_path(key) for key in keys]
            success = self._execute_transfer(dst_tensors, file_paths, "READ")
        else:
            success = self._execute_transfer(dst_tensors, keys, "READ")
        return dst_tensors if success else [None] * len(keys)

    def exists(self, key: str) -> bool:
        tuples = self.registration.create_query_tuples(
            key,
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
