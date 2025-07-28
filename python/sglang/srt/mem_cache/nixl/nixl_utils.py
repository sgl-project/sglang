import os
import logging
import torch
from typing import Dict, Tuple, Optional, List, Union

logger = logging.getLogger(__name__)


class NixlBackendSelection:
    """Handles NIXL backend selection and creation."""

    # File-based plugins
    FILE_PLUGINS = {"3FS", "POSIX", "GDS_MT", "GDS"}
    # Object-based plugins (add more as needed)
    OBJ_PLUGINS = {"OBJ"}  # Based on Amazon S3 SDK
    # Priority order for auto selection
    AUTO_PRIORITY = ["3FS", "POSIX", "GDS_MT", "GDS"]

    def __init__(self, plugin: str = "auto"):
        """Initialize backend selection.
        Args:
            plugin: Plugin to use (default "auto" selects best available).
                   Can be a file plugin (3FS, POSIX, GDS, GDS_MT) or
                   an object plugin (OBJ).
        """
        self.plugin = plugin
        self.backend_name = None
        self.mem_type = None

    def create_backend(self, agent) -> bool:
        """Create the appropriate NIXL backend based on configuration."""
        try:
            plugin_list = agent.get_plugin_list()
            logger.debug(f"Available NIXL plugins: {plugin_list}")

            # Select backend based on plugin setting or auto priority
            self.backend_name = next((p for p in self.AUTO_PRIORITY if p in plugin_list), "POSIX") if self.plugin == "auto" else self.plugin

            if self.backend_name not in plugin_list:
                logger.error(f"Backend {self.backend_name} not available in plugins: {plugin_list}")
                return False

            # Create backend and set memory type
            agent.create_backend(self.backend_name)
            self.mem_type = "OBJ" if self.backend_name in self.OBJ_PLUGINS else "FILE"
            logger.debug(f"Created NIXL backend: {self.backend_name} with memory type: {self.mem_type}")
            return True

        except Exception as e:
            logger.error(f"Failed to create NIXL backend: {e}")
            return False


class NixlRegistration:
    """Handles NIXL memory registration."""

    def __init__(self, agent):
        self.agent = agent

    def create_query_tuples(self, key: str, mem_type: str, file_manager=None) -> List[Tuple]:
        """Create NIXL tuples for querying memory.
        Args:
            key: Key to query (file path for FILE or object key for OBJ)
            mem_type: Memory type ("FILE" or "OBJ")
            file_manager: Optional NixlFileManager for FILE memory type
        Returns:
            List of NIXL tuples for querying
        """
        if mem_type == "FILE":
            if file_manager is None:
                logger.error("file_manager required for FILE memory type")
                return []
            return [(0, 0, 0, file_manager.get_file_path(key))]
        else:  # OBJ
            return [(0, 0, key)]

    def create_transfer_tuples(self, keys: List[str], tensors: Optional[List[torch.Tensor]] = None, mem_type: str = "FILE", file_manager=None) -> List[Tuple[int, int, Union[int, str]]]:
        """Create NIXL tuples for transfer operations.
        Args:
            keys: List of keys (file paths for FILE, object keys for OBJ)
            tensors: Optional list of tensors to get sizes from
            mem_type: Memory type ("FILE" or "OBJ")
            file_manager: Optional file manager for FILE operations
        Returns:
            List of (addr, len, id) tuples for transfer operations
        """
        if mem_type == "FILE":
            if not file_manager:
                logger.error("file_manager required for FILE operations")
                return []
            file_tuples = file_manager.files_to_nixl_tuples(keys)
            if not file_tuples:
                return []
            # Extract (offset, length, fd) from file tuples
            return [(x[0], tensor.element_size() * tensor.numel() if tensors else x[1], x[2])
                   for x, tensor in zip(file_tuples, tensors or [None] * len(file_tuples))]
        else:  # OBJ
            # Create object tuples with proper sizes
            return [(0, tensor.element_size() * tensor.numel() if tensor else 0, key)
                   for key, tensor in zip(keys, tensors or [None] * len(keys))]

    def _register_memory(self, items: Union[List[tuple], List[torch.Tensor]], mem_type: str, desc: str) -> Optional[any]:
        """Common registration logic for files, objects, and buffers.
        Args:
            items: List of tuples or tensors to register
            mem_type: Memory type ("FILE", "OBJ", "DRAM", "VRAM")
            desc: Description for logging
        """
        try:
            if not items:
                return None

            reg_descs = self.agent.get_reg_descs(items, mem_type)
            if reg_descs is None:
                logger.error("Failed to create registration descriptors")
                return None

            registered_memory = self.agent.register_memory(reg_descs)
            if registered_memory:
                return registered_memory
            else:
                logger.error("Failed to register with NIXL")
                return None

        except Exception as e:
            logger.error(f"Failed to register {desc}: {e}")
            return None

    def register_buffers(self, buffers: Union[torch.Tensor, List[torch.Tensor]]) -> Optional[any]:
        """Register tensors/buffers with NIXL."""
        if isinstance(buffers, torch.Tensor):
            buffers = [buffers]

        if not buffers:
            return None

        # Determine memory type based on tensor device
        mem_type = "VRAM" if buffers[0].device.type == 'cuda' else "DRAM"
        return self._register_memory(buffers, mem_type, "buffers")

    def register_files(self, tuples: List[tuple]) -> Optional[any]:
        """Register files with NIXL using (0, 0, fd, file_path) tuples."""
        return self._register_memory(tuples, "FILE", "files")

    def register_objects(self, keys: List[str], tensors: Optional[List[torch.Tensor]] = None) -> Optional[any]:
        """Register objects with NIXL."""
        if not keys:
            return None

        # Create object tuples with proper sizes
        tuples = [(0, tensor.element_size() * tensor.numel() if tensor else 0, key)
                 for key, tensor in zip(keys, tensors or [None] * len(keys))]
        return self._register_memory(tuples, "OBJ", "objects")


class NixlFileManager:
    """Handles file system operations for NIXL."""

    def __init__(self, base_dir: str):
        """
        Initialize file manager.
        Args:
            base_dir: Base directory for storing tensor files
        """
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        logger.debug(f"Initialized file manager with base directory: {base_dir}")

    def get_file_path(self, key: str) -> str:
        """Get full file path for a given key."""
        return os.path.join(self.base_dir, key)

    def create_file(self, file_path: str) -> bool:
        """Create a file if it doesn't exist."""
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            if not os.path.exists(file_path):
                with open(file_path, 'wb') as f:
                    pass  # Create empty file
            return True
        except Exception as e:
            logger.error(f"Failed to create file {file_path}: {e}")
            return False

    def open_file(self, file_path: str) -> Optional[int]:
        """Open a file and return its file descriptor."""
        try:
            fd = os.open(file_path, os.O_RDWR)
            return fd
        except Exception as e:
            logger.error(f"Failed to open file {file_path}: {e}")
            return None

    def close_file(self, fd: int) -> bool:
        """Close a file descriptor."""
        try:
            os.close(fd)
            return True
        except Exception as e:
            logger.error(f"Failed to close file descriptor {fd}: {e}")
            return False

    def files_to_nixl_tuples(self, file_paths: List[str], open_file: bool = True) -> List[Tuple[int, int, int, str]]:
        """Create NIXL tuples (offset, length, fd, file_path) for given files."""
        if not open_file:
            return [(0, 0, 0, path) for path in file_paths]

        tuples = []
        for path in file_paths:
            if (fd := self.open_file(path)) is None:
                # Clean up on failure
                [self.close_file(t[2]) for t in tuples]
                return []
            tuples.append((0, 0, fd, path))
        return tuples
