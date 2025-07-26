import os
import logging
import torch
from typing import Dict, Tuple, Optional, List, Union

logger = logging.getLogger(__name__)


class NixlBackendSelection:
    """Handles NIXL backend selection and creation."""

    def __init__(self, file_plugin: str = "auto"):
        self.file_plugin = file_plugin

    def create_backend(self, agent) -> bool:
        """Create the appropriate NIXL backend based on configuration."""
        try:
            # Get available plugins
            plugin_list = agent.get_plugin_list()
            logger.debug(f"Available NIXL plugins: {plugin_list}")

            # Select backend based on file_plugin setting
            if self.file_plugin == "auto":
                if "3FS" in plugin_list:
                    backend_name = "3FS"
                elif "POSIX" in plugin_list:
                    backend_name = "POSIX"
                elif "GDS_MT" in plugin_list:
                    backend_name = "GDS_MT"
                elif "GDS" in plugin_list:
                    backend_name = "GDS"
                else:
                    logger.warning("No suitable NIXL backend found, using POSIX")
                    backend_name = "POSIX"
            else:
                backend_name = self.file_plugin

            # Create the selected backend
            if backend_name in plugin_list:
                agent.create_backend(backend_name)
                logger.debug(f"Created NIXL backend: {backend_name}")
                return True
            else:
                logger.error(f"Backend {backend_name} not available in plugins: {plugin_list}")
                return False

        except Exception as e:
            logger.error(f"Failed to create NIXL backend: {e}")
            return False


class NixlRegistration:
    """Handles NIXL memory registration."""

    def __init__(self, agent):
        self.agent = agent

    def register_buffers(self, buffers: Union[torch.Tensor, List[torch.Tensor]]) -> Optional[any]:
        """Register tensors/buffers with NIXL."""
        try:
            if isinstance(buffers, torch.Tensor):
                buffers = [buffers]

            if not buffers:
                logger.debug("No buffers to register")
                return None

            # Determine memory type based on tensor device
            memory_type = "VRAM" if buffers[0].device.type == 'cuda' else "DRAM"
            logger.debug(f"Registering {len(buffers)} buffers with memory type: {memory_type}")

            # Let NIXL handle tensor descriptors with explicit memory type
            reg_descs = self.agent.get_reg_descs(buffers, memory_type)
            if reg_descs is None:
                logger.error("Failed to create registration descriptors")
                return None

            registered_memory = self.agent.register_memory(reg_descs)
            if registered_memory:
                logger.debug(f"Registered {len(buffers)} buffers")
                return registered_memory
            else:
                logger.error("Failed to register with NIXL")
                return None

        except Exception as e:
            logger.error(f"Failed to register buffers: {e}")
            return None

    def register_files(self, tuples: List[tuple]) -> Optional[any]:
        """Register files with NIXL using (0, 0, fd, file_path) tuples."""
        try:
            if not tuples:
                logger.debug("No files to register")
                return None

            reg_descs = self.agent.get_reg_descs(tuples, "FILE")
            if reg_descs is None:
                logger.error("Failed to create registration descriptors")
                return None

            registered_memory = self.agent.register_memory(reg_descs)
            if registered_memory:
                logger.debug(f"Registered {len(tuples)} files")
                return registered_memory
            else:
                logger.error("Failed to register with NIXL")
                return None

        except Exception as e:
            logger.error(f"Failed to register files: {e}")
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
        """
        Create NIXL tuples (offset, length, fd, file_path) for given files.
        Args:
            file_paths: List of file pathstuples
        Returns:
            List of NIXL tuples or empty list if any operation fails.
        """
        try:
            tuples = []
            opened_fds = []

            if open_file:
                for file_path in file_paths:
                    # Open file and get file descriptor
                    fd = self.open_file(file_path)
                    if fd is None:
                        # Clean up already opened files
                        for fd in opened_fds:
                            self.close_file(fd)
                        return []

                    opened_fds.append(fd)
                    # Format: (address, length, device_id, meta_info) = (offset, length, fd, file_path)
                    # Can be customized to write multiple entries to the same file based on offset and length
                    tuples.append((0, 0, fd, file_path))
            else:
                tuples = [(0, 0, 0, file_path) for file_path in file_paths]

            return tuples

        except Exception as e:
            logger.error(f"Failed to create NIXL tuples: {e}")
            # Clean up opened files on exception
            for fd in opened_fds:
                try:
                    self.close_file(fd)
                except:
                    pass
            return []
