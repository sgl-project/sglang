import logging
import torch
from typing import List, Dict, Optional, Tuple
from .nixl_file_management import NixlFileManagement
import os

logger = logging.getLogger(__name__)

class NixlRegistration:
    """Handles NIXL memory and file registration."""

    def __init__(self, agent):
        self.agent = agent
        self.registered_tensors: Dict[int, any] = {}  # tensor_id -> nixl_descs

    def register_tensor(self, tensor: torch.Tensor) -> bool:
        """Register a single tensor with NIXL."""
        try:
            tensor_id = id(tensor)
            if tensor_id in self.registered_tensors:
                logger.debug(f"Tensor {tensor_id} already registered")
                return True

            tensor_rdesc = self.agent.register_memory(tensor)
            if tensor_rdesc:
                self.registered_tensors[tensor_id] = tensor_rdesc
                logger.debug(f"Registered tensor {tensor_id} with NIXL")
                return True
            else:
                logger.error(f"Failed to register tensor {tensor_id}")
                return False

        except Exception as e:
            logger.error(f"Failed to register tensor: {e}")
            return False

    def register_tensors_batch(self, tensors: List[torch.Tensor]) -> bool:
        """Register multiple tensors with NIXL in batch."""
        if not tensors:
            logger.debug("No tensors was passed to register_tensors_batch")
            return True

        try:
            unregistered_tensors = [x for x in tensors if id(x) not in self.registered_tensors]
            if not unregistered_tensors:
                logger.debug("All tensors already registered")
                return True

            tensors_rdesc = self.agent.register_memory(tensors)

            if tesnors_rdesc:
                for tensor in unregistered_tensors:
                    self.registered_tensors[id(tensor)] = tensors_rdesc # Cannot be used for indivisual deregister
                logger.debug(f"Batch registered {len(tensors)} tensors")
            else:
                logger.error("Failed to batch register tensors")
                return False

            return True

        except Exception as e:
            logger.error(f"Failed to batch register tensors: {e}")
            return False

    def register_file(self, file_path: str, file_manager: NixlFileManagement) -> bool:
        """Register a file with NIXL."""
        if not file_paths:
            return True

        try:
            # Open files and collect file descriptors
            if file_path not in file_manager.registered_files:
                # Create file if it doesn't exist
                if not os.path.exists(file_path):
                    if not file_manager.create_file(file_path):
                        logger.error(f"Failed to create file {file_path}")
                        return False

                # Open file
                fd = file_manager.open_file(file_path)
                if fd is None:
                    logger.error(f"Failed to open file {file_path}")
                    return False
            else:
                logger.debug(f"File {file_path} already registered")

            # Register file with NIXL
            if fd:
                logger.info(f"Registering 1 file with fd {fd}")
                # Use proper format: (0, buf_size, fd, "") with buf_size = 0 for unlimited
                file_desc = self.agent.register_memory([(0, 0, fd, "")], "FILE")
                if file_desc:
                    file_manager.registered_files[file_path] = (fd, file_desc)
                    logger.debug(f"Registered fd {fd} with NIXL")
                    return True
                else:
                    file_manager.close_file(fd)
                    logger.warning("Failed to register fd {fd} with NIXL")
                    return False

            return True

        except Exception as e:
            logger.error(f"Failed to batch register files with NIXL: {e}")
            # Clean up opened file descriptors on exception
            for fd in fds:
                try:
                    file_manager.close_file(fd)
                except:
                    pass
            return False

    def register_files_batch(self, file_paths: List[str], file_manager: NixlFileManagement) -> bool:
        """Register multiple files with NIXL in batch."""
        if not file_paths:
            return True

        try:
            # Open files and collect file descriptors
            fds = []
            unregistered_files = []

            for file_path in file_paths:
                if file_path not in file_manager.registered_files:
                    # Create file if it doesn't exist
                    if not os.path.exists(file_path):
                        if not file_manager.create_file(file_path):
                            logger.error(f"Failed to create file {file_path}")
                            return False

                    # Open file
                    fd = file_manager.open_file(file_path)
                    if fd is None:
                        logger.error(f"Failed to open file {file_path}")
                        return False

                    fds.append(fd)
                    unregistered_files.append(file_path)
                else:
                    logger.debug(f"File {file_path} already registered")

            # Register files with NIXL
            if fds:
                logger.info(f"Number of file descriptors: {len(fds)}")
                # Use proper format: (0, buf_size, fd, "") with buf_size = 0 for unlimited
                file_list = [(0, 0, fd, "") for fd in fds]
                logger.info(f"Fds for registration: {fds}")
                files_desc = self.agent.register_memory(file_list, "FILE")
                if files_desc:
                    for file_path, fd in zip(unregistered_files, fds):
                        file_manager.registered_files[file_path] = (fd, files_desc) # Cannot be used for indivisual deregister
                    logger.debug(f"Batch registered {len(fds)} files with NIXL")
                    return True
                else:
                    # Clean up opened file descriptors on failure
                    for fd in fds:
                        file_manager.close_file(fd)
                    logger.warning("Failed to batch register files with NIXL")
                    return False

            return True

        except Exception as e:
            logger.error(f"Failed to batch register files with NIXL: {e}")
            # Clean up opened file descriptors on exception
            for fd in fds:
                try:
                    file_manager.close_file(fd)
                except:
                    pass
            return False
