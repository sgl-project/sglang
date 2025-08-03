import hashlib
import logging
import os
import time
import uuid
from typing import Dict, List, Optional, Tuple, Union

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

    def _execute_transfer(
        self, tensors: List[torch.Tensor], keys: List[str], direction: str
    ) -> bool:
        if len(tensors) != len(keys):
            logger.error("Mismatch between number of tensors and files/objects")
            return False

        if not self.registration.register_buffers(tensors):
            logger.error("Failed to register tensors")
            return False

        # Get transfer tuples based on backend type
        tensor_sizes = [tensor.element_size() * tensor.numel() for tensor in tensors]
        if self.backend_selector.mem_type == "FILE":
            file_tuples = self.file_manager.files_to_nixl_tuples(keys)
            if not file_tuples or not self.registration.register_files(file_tuples):
                logger.error("Failed to prepare files for transfer")
                return False
            transfer_tuples = [
                (x[0], s, x[2]) for x, s in zip(file_tuples, tensor_sizes)
            ]
        else:
            if not self.registration.register_objects(keys, tensors):
                logger.error("Failed to register objects")
                return False
            transfer_tuples = [(0, s, key) for s, key in zip(tensor_sizes, keys)]

        try:
            # Get transfer descriptors
            if (tensor_descs := self.agent.get_xfer_descs(tensors)) is None or (
                file_descs := self.agent.get_xfer_descs(
                    transfer_tuples, self.backend_selector.mem_type
                )
            ) is None:
                logger.error("Failed to get transfer descriptors")
                return False

            # Initialize and execute transfer
            if (
                xfer_req := self.agent.initialize_xfer(
                    direction, tensor_descs, file_descs, self.agent_name
                )
            ) is None:
                logger.error("Failed to create transfer request")
                return False

            state = self.agent.transfer(xfer_req)
            while state != "DONE":
                state = self.agent.check_xfer_state(xfer_req)
                if state == "ERR":
                    logger.error("Transfer failed")
                    return False
            time.sleep(0.0001)  # Can be changed to os.sched_yield() or parametrized
            return True

        except Exception as e:
            logger.error(f"Failed to execute transfer: {e}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            return False

    def batch_set(self, keys: List[str], values: List[torch.Tensor]) -> bool:
        if not keys:
            return True

        if self.backend_selector.mem_type == "FILE":
            file_paths = []
            for key in keys:
                tensor_path = self.file_manager.get_file_path(key)
                if not self.file_manager.create_file(tensor_path):
                    logger.error(f"Failed to create file {tensor_path}")
                    return False
                file_paths.append(tensor_path)
            return self._execute_transfer(values, file_paths, "WRITE")
        else:
            return self._execute_transfer(values, keys, "WRITE")

    def set(self, key: str, value: torch.Tensor) -> bool:
        return self.batch_set([key], [value])

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
