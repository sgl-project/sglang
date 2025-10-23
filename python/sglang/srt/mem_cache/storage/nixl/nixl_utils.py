import logging
import os
from typing import Any, List, Optional, Tuple, Union

import torch

logger = logging.getLogger(__name__)


class NixlBackendSelection:
    """Handles NIXL backend selection and creation."""

    # Priority order for File-based plugins in case of auto selection
    FILE_PLUGINS = ["3FS", "POSIX", "GDS_MT", "GDS"]
    # Priority order for File-based plugins in case of auto selection (add more as needed)
    OBJ_PLUGINS = ["OBJ"]  # Based on Amazon S3 SDK

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

    def set_bucket(self, bucket_name: str) -> None:
        """Set AWS bucket name in environment variable."""
        os.environ["AWS_DEFAULT_BUCKET"] = bucket_name
        logger.debug(f"Set AWS bucket name to: {bucket_name}")

    def create_backend(self, agent) -> bool:
        """Create the appropriate NIXL backend based on configuration."""
        try:
            plugin_list = agent.get_plugin_list()
            logger.debug(f"Available NIXL plugins: {plugin_list}")

            # Handle explicit plugin selection or auto priority
            if self.plugin == "auto":
                # Try all file plugins first
                for plugin in self.FILE_PLUGINS:
                    if plugin in plugin_list:
                        self.backend_name = plugin
                        break
                # If no file plugin found, try object plugins
                if not self.backend_name:
                    for plugin in self.OBJ_PLUGINS:
                        if plugin in plugin_list:
                            self.backend_name = plugin
                            break
            else:
                # Use explicitly requested plugin
                self.backend_name = self.plugin

            if self.backend_name not in plugin_list:
                logger.error(
                    f"Backend {self.backend_name} not available in plugins: {plugin_list}"
                )
                return False

            # Create backend and set memory type
            if self.backend_name in self.OBJ_PLUGINS:
                bucket = os.environ.get("AWS_DEFAULT_BUCKET")
                if not bucket:
                    logger.error(
                        "AWS_DEFAULT_BUCKET environment variable must be set for object storage"
                    )
                    return False
                agent.create_backend(self.backend_name, {"bucket": bucket})
            else:
                agent.create_backend(self.backend_name)

            self.mem_type = "OBJ" if self.backend_name in self.OBJ_PLUGINS else "FILE"
            logger.debug(
                f"Created NIXL backend: {self.backend_name} with memory type: {self.mem_type}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to create NIXL backend: {e}")
            return False


class NixlRegistration:
    """Handles NIXL memory registration."""

    def __init__(self, agent):
        self.agent = agent

    def create_query_tuples(
        self, key: str, mem_type: str, file_manager=None
    ) -> List[Tuple]:
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

    def _register_memory(
        self,
        items: Union[List[tuple], torch.Tensor, List[torch.Tensor]],
        mem_type: Optional[str] = None,
    ) -> Optional[Any]:
        """Common registration logic for files, objects, and buffers.
        Args:
            items: List of tuples or tensors to register
            mem_type: Memory type ("FILE", "OBJ") or None for tensor or list of tensors
        """
        if isinstance(items, list) and not items:
            return None

        reg_descs = self.agent.get_reg_descs(items, mem_type)
        if reg_descs is None:
            logger.error("Failed to create registration descriptors")
            return None

        try:
            registered_memory = self.agent.register_memory(reg_descs)
            return registered_memory  # Could be None in case of error
        except Exception as e:
            if not mem_type:
                logger.error(f"Failed to register Tensors with NIXL: {e}")
            else:
                logger.error(
                    f"Failed to register memory of type {mem_type} with NIXL: {e}"
                )
            return None


class NixlFileManager:
    """Handles file system operations for NIXL."""

    def __init__(self, base_dir: str):
        """
        Initialize file manager.
        Args:
            base_dir: Base directory for storing tensor files
        """
        self.base_dir = base_dir
        if base_dir == "":
            logger.debug(f"Initialized file manager without a base directory")
        else:
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
                with open(file_path, "wb") as f:
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

    def files_to_nixl_tuples(
        self, file_paths: List[str]
    ) -> List[Tuple[int, int, int, str]]:
        """Create NIXL tuples (offset, length, fd, file_path) for given files."""
        tuples = []
        for path in file_paths:
            if (fd := self.open_file(path)) is None:
                # Clean up on failure
                for t in tuples:
                    self.close_file(t[2])
                return []
            tuples.append((0, 0, fd, path))
        return tuples
