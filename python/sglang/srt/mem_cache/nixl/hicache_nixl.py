import hashlib
import logging
import os
import uuid
from typing import List, Optional, Dict, Tuple

import torch

from sglang.srt.mem_cache.hicache_storage import HiCacheStorage
from .nixl_registration import NixlRegistration
from .nixl_file_management import NixlFileManagement
from .nixl_backend_selection import NixlBackendSelection

logger = logging.getLogger(__name__)

class HiCacheNixl(HiCacheStorage):
    """
    HiCacheNixl provides high-performance storage using NIXL file plugins.
    It leverages NIXL's efficient data transfer capabilities for storing and retrieving KV cache.
    Similar to HiCacheFile but with NIXL optimizations for GPU memory and different file plugins.
    """

    def __init__(self, file_path: str = "/tmp/hicache_nixl", file_plugin: str = "auto"):
        """
        Initialize NIXL storage connector.
        
        Args:
            file_path: Path to the storage directory for NIXL file operations
            file_plugin: Plugin to use ("auto", "GDS_MT", "3FS", "POSIX"). 
                        "auto" selects the best available plugin automatically.
        """
        try:
            from nixl._api import nixl_agent, nixl_agent_config
        except ImportError as e:
            raise ImportError(
                "Please install NIXL by following the instructions at "
                "https://github.com/ai-dynamo/nixl/blob/main/README.md "
                "to use HiCacheNixl storage backend."
            ) from e
        
        # Initialize file management
        self.file_manager = NixlFileManagement(file_path)
        if not self.file_manager.ensure_directory_exists():
            raise Exception("Failed to create storage directory")
        
        # Initialize NIXL agent
        agent_config = nixl_agent_config(backends=[])
        self.agent = nixl_agent(str(uuid.uuid4()), agent_config)
        
        # Create backend with user-specified or automatic plugin selection
        self.backend_selector = NixlBackendSelection(file_plugin)
        if not self.backend_selector.create_backend(self.agent):
            raise Exception("Failed to create NIXL backend")
        
        # Initialize registration manager
        self.registration = NixlRegistration(self.agent)


    def _ensure_file_opened(self, file_path: str) -> bool:
        """Ensure a file is opened and registered with NIXL, opening it if necessary."""
        if file_path not in self.file_manager.registered_files:
            return self._register_files_batch([file_path])
        return True

    def _handle_new_file_creation(self, file_path: str) -> bool:
        """Handle newly created files by opening and registering them with NIXL."""
        try:
            # Create the file if it doesn't exist
            if not self.file_manager.create_file(file_path):
                return False
            
            # Register the new file
            return self._register_files_batch([file_path])
        except Exception as e:
            logger.error(f"Failed to handle new file creation for {file_path}: {e}")
            return False

    def _is_gpu_tensor(self, tensor: torch.Tensor) -> bool:
        """Check if tensor is on GPU memory."""
        return tensor.device.type == 'cuda'

    def _get_file_path(self, key: str) -> str:
        """Get the file path for a given key."""
        return self.file_manager.get_file_path(key)

    def _register_tensor(self, tensor: torch.Tensor) -> bool:
        """Register a tensor with NIXL for I/O operations."""
        return self.registration.register_tensor(tensor)

    def _register_file(self, file_path: str) -> bool:
        """Register a file with NIXL for I/O operations using OS file descriptor."""
        return self.registration.register_files_batch([file_path], self.file_manager)

    def _register_tensors_batch(self, tensors: List[torch.Tensor]) -> bool:
        """Register multiple tensors with NIXL in a single batch operation."""
        return self.registration.register_tensors_batch(tensors)

    def _register_files_batch(self, file_paths: List[str]) -> bool:
        """Register multiple files with NIXL in a single batch operation."""
        return self.registration.register_files_batch(file_paths, self.file_manager)

    def get(
        self, key: str, dst_tensor: torch.Tensor
    ) -> torch.Tensor | None:
        """
        Retrieve the value associated with the given key from NIXL storage.
        Returns None if the key does not exist.
        
        Args:
            key: Storage key
            dst_tensor: Destination tensor buffer for the transfer. Required for pure NIXL operation.
        """
        logger.debug(f"=== GET OPERATION ===")
        logger.debug(f"Key: {key}")
        logger.debug(f"Destination tensor shape: {dst_tensor.shape}, dtype: {dst_tensor.dtype}")
        logger.debug(f"Destination tensor ID: {id(dst_tensor)}")
        
        # Use batch_get for consistency
        result = self.batch_get([key], [dst_tensor])
        retrieved = result[0] if result else None
        logger.debug(f"Get operation result: {retrieved is not None}")
        return retrieved

    def batch_get(
        self,
        keys: List[str],
        dst_tensors: List[torch.Tensor],
    ) -> List[torch.Tensor | None]:
        """
        Retrieve values for multiple keys from NIXL storage using batch operations.
        Returns a list of tensors or None for each key.
        
        Args:
            keys: List of storage keys
            dst_tensors: List of destination tensor buffers for the transfers
        """
        if not keys:
            return []
        
        # Collect all valid file paths and tensors for batch processing
        valid_tensors = []
        unregistered_tensors = []
        valid_file_paths = []
        valid_indices = []
        
        for i, key in enumerate(keys):
            tensor_path = self._get_file_path(key)
            
            if not os.path.exists(tensor_path):
                continue
            
            try:
                # Ensure file is opened (already pre-opened at init)
                if not self._ensure_file_opened(tensor_path):
                    logger.error(f"Failed to ensure file {tensor_path} is opened")
                    continue
                
                # Collect for batch operation
                valid_tensors.append(dst_tensors[i])
                valid_file_paths.append(tensor_path)
                valid_indices.append(i)
                
                # Check if tensor needs registration
                if id(dst_tensors[i]) not in self.registration.registered_tensors:
                    unregistered_tensors.append(dst_tensors[i])
                    
            except Exception as e:
                logger.error(f"Failed to prepare batch get for key {key}: {e}")
                continue
        
        # Batch register only unregistered tensors
        if unregistered_tensors:
            try:
                # Batch register unregistered tensors
                if not self._register_tensors_batch(unregistered_tensors):
                    logger.error("Failed to batch register tensors")
                    return dst_tensors
            except Exception as e:
                logger.error(f"Failed to register tensors: {e}")
                return dst_tensors
        
        # Perform batch transfer if we have valid operations
        if valid_tensors:
            logger.debug(f"=== BATCH GET TRANSFER ===")
            logger.debug(f"Number of valid tensors: {len(valid_tensors)}")
            logger.debug(f"Number of valid file paths: {len(valid_file_paths)}")
            
            try:
                # Process transfers individually since NIXL batch transfer is complex
                for i, (tensor, file_path) in enumerate(zip(valid_tensors, valid_file_paths)):
                    logger.debug(f"--- Processing transfer {i+1}/{len(valid_tensors)} ---")
                    logger.debug(f"Tensor ID: {id(tensor)}")
                    logger.debug(f"File path: {file_path}")
                    
                    # Debug: Check if tensor is registered
                    if id(tensor) not in self.registration.registered_tensors:
                        logger.error(f"Tensor {id(tensor)} not found in registered_tensors")
                        continue
                    
                    # Debug: Check if file is registered
                    if file_path not in self.file_manager.registered_files:
                        logger.error(f"File {file_path} not found in registered_files")
                        continue
                    
                    # Debug: Log transfer details
                    logger.debug(f"Transfer: tensor_id={id(tensor)}, file_path={file_path}")
                    logger.debug(f"Registered tensors: {list(self.registration.registered_tensors.keys())}")
                    logger.debug(f"Registered files: {list(self.file_manager.registered_files.keys())}")
                    
                    # Convert registration descriptors to transfer descriptors using .trim()
                    try:
                        src_descs = self.file_manager.registered_files[file_path][1].trim()
                        logger.debug(f"Source transfer descriptors created successfully: {src_descs}")
                        
                        dst_descs = self.registration.registered_tensors[id(tensor)].trim()
                        logger.debug(f"Destination transfer descriptors created successfully: {dst_descs}")
                        
                        # Create transfer request with transfer descriptors
                        logger.debug("About to call initialize_xfer...")
                        xfer_req = self.agent.initialize_xfer(
                            "READ",
                            src_descs,
                            dst_descs,
                            str(self.agent)  # peer_name (agent name for local transfers)
                        )
                        logger.debug(f"Transfer request created successfully: {xfer_req}")
                        
                    except Exception as e:
                        logger.error(f"Failed to create transfer descriptors or initialize transfer: {e}")
                        logger.error(f"Error type: {type(e)}")
                        continue
                    
                    logger.debug(f"Transfer request created: {xfer_req}")
                    
                    # Submit transfer
                    state = self.agent.transfer(xfer_req)
                    logger.debug(f"Transfer state: {state}")
                    
                    # Wait for transfer completion
                    while True:
                        state = self.agent.check_xfer_state(xfer_req)
                        logger.debug(f"Transfer state check: {state}")
                        if state == "completed":
                            logger.debug(f"Transfer {i+1} completed successfully")
                            break
                        elif state == "failed":
                            logger.error(f"Batch transfer failed for tensor {id(tensor)}")
                            break
                        # Continue waiting for completion
                    
            except Exception as e:
                logger.error(f"Failed to execute batch get: {e}")
        
        # Return dst_tensors directly - they contain the loaded data after transfers
        return dst_tensors

    def set(self, key: str, value: torch.Tensor, overwrite: bool = False) -> bool:
        """
        Store the value associated with the given key in NIXL storage.
        Returns True if the operation was successful, False otherwise.
        
        Args:
            key: Storage key
            value: Tensor to store
            overwrite: If True, overwrite existing key. If False, skip if key exists.
        """
        logger.debug(f"=== SET OPERATION ===")
        logger.debug(f"Key: {key}")
        logger.debug(f"Tensor shape: {value.shape}, dtype: {value.dtype}")
        logger.debug(f"Tensor ID: {id(value)}")
        logger.debug(f"Overwrite: {overwrite}")
        
        # Use batch_set for consistency
        result = self.batch_set([key], [value], overwrite)
        logger.debug(f"Set operation result: {result}")
        return result

    def batch_set(self, keys: List[str], values: List[torch.Tensor], overwrite: bool = False) -> bool:
        """
        Store multiple key-value pairs in NIXL storage using batch operations.
        Returns True if all operations were successful, False otherwise.
        
        Args:
            keys: List of storage keys
            values: List of tensors to store
            overwrite: If True, overwrite existing keys. If False, skip if key exists.
        """
        if not keys:
            return True
        
        # Collect all valid tensors and file paths for batch processing
        valid_tensors = []
        unregistered_tensors = []
        valid_file_paths = []
        
        for key, value in zip(keys, values):
            tensor_path = self._get_file_path(key)
            
            # Check if key exists and handle overwrite logic
            if os.path.exists(tensor_path):
                if not overwrite:
                    logger.debug(f"Key {key} already exists. Skipped.")
                    continue
                else:
                    logger.debug(f"Overwriting existing key {key}")
            
            try:
                # Handle file (create if new, ensure opened if existing)
                if not os.path.exists(tensor_path):
                    if not self._handle_new_file_creation(tensor_path):
                        logger.error(f"Failed to create and register file {tensor_path}")
                        return False
                else:
                    if not self._ensure_file_opened(tensor_path):
                        logger.error(f"Failed to ensure file {tensor_path} is opened")
                        return False
                
                # Collect for batch operation
                valid_tensors.append(value)
                valid_file_paths.append(tensor_path)
                
                # Check if tensor needs registration
                if id(value) not in self.registration.registered_tensors:
                    unregistered_tensors.append(value)
                
            except Exception as e:
                logger.error(f"Failed to prepare batch set for key {key}: {e}")
                return False
        
        # Batch register only unregistered tensors
        if unregistered_tensors:
            try:
                # Batch register unregistered tensors
                if not self._register_tensors_batch(unregistered_tensors):
                    logger.error("Failed to batch register tensors")
                    return False
            except Exception as e:
                logger.error(f"Failed to register tensors: {e}")
                return False
        
        # Perform batch transfer if we have valid operations
        if valid_tensors:
            logger.debug(f"=== BATCH SET TRANSFER ===")
            logger.debug(f"Number of valid tensors: {len(valid_tensors)}")
            logger.debug(f"Number of valid file paths: {len(valid_file_paths)}")
            
            try:
                # Process transfers individually since NIXL batch transfer is complex
                for i, (tensor, file_path) in enumerate(zip(valid_tensors, valid_file_paths)):
                    logger.debug(f"--- Processing transfer {i+1}/{len(valid_tensors)} ---")
                    logger.debug(f"Tensor ID: {id(tensor)}")
                    logger.debug(f"File path: {file_path}")
                    
                    # Debug: Check if tensor is registered
                    if id(tensor) not in self.registration.registered_tensors:
                        logger.error(f"Tensor {id(tensor)} not found in registered_tensors")
                        return False
                    
                    # Debug: Check if file is registered
                    if file_path not in self.file_manager.registered_files:
                        logger.error(f"File {file_path} not found in registered_files")
                        return False
                    
                    # Debug: Log transfer details
                    logger.debug(f"Transfer: tensor_id={id(tensor)}, file_path={file_path}")
                    logger.debug(f"Registered tensors: {list(self.registration.registered_tensors.keys())}")
                    logger.debug(f"Registered files: {list(self.file_manager.registered_files.keys())}")
                    
                    # Convert registration descriptors to transfer descriptors using .trim()
                    try:
                        src_descs = self.registration.registered_tensors[id(tensor)].trim()
                        logger.debug(f"Source transfer descriptors created successfully: {src_descs}")
                        
                        dst_descs = self.file_manager.registered_files[file_path][1].trim()
                        logger.debug(f"Destination transfer descriptors created successfully: {dst_descs}")
                        
                        # Create transfer request with transfer descriptors
                        logger.debug("About to call initialize_xfer...")
                        xfer_req = self.agent.initialize_xfer(
                            "WRITE",
                            src_descs,
                            dst_descs,
                            str(self.agent)  # peer_name (agent name for local transfers)
                        )
                        logger.debug(f"Transfer request created successfully: {xfer_req}")
                        
                    except Exception as e:
                        logger.error(f"Failed to create transfer descriptors or initialize transfer: {e}")
                        logger.error(f"Error type: {type(e)}")
                        return False
                    
                    logger.debug(f"Transfer request created: {xfer_req}")
                    
                    # Submit transfer
                    state = self.agent.transfer(xfer_req)
                    logger.debug(f"Transfer state: {state}")
                    
                    # Wait for transfer completion
                    while True:
                        state = self.agent.check_xfer_state(xfer_req)
                        logger.debug(f"Transfer state check: {state}")
                        if state == "completed":
                            logger.debug(f"Transfer {i+1} completed successfully")
                            break
                        elif state == "failed":
                            logger.error(f"Batch transfer failed for tensor {id(tensor)}")
                            return False
                        # Continue waiting for completion
                    
            except Exception as e:
                logger.error(f"Failed to execute batch set: {e}")
                return False
        
        return True

    def exists(self, key: str) -> bool:
        """
        Check if the key exists in NIXL storage.
        Returns True if the key exists, False otherwise.
        """
        tensor_path = self._get_file_path(key)
        return os.path.exists(tensor_path)

    def delete(self, key: str) -> None:
        """
        Delete the key from NIXL storage.
        """
        tensor_path = self._get_file_path(key)
        try:
            if os.path.exists(tensor_path):
                os.remove(tensor_path)
                # Clean up registered file if it exists
                if tensor_path in self.file_manager.registered_files:
                    fd, file_descs = self.file_manager.registered_files[tensor_path]
                    self.agent.deregister_memory(file_descs)
                    os.close(fd)
                    del self.file_manager.registered_files[tensor_path]
                logger.debug(f"Successfully deleted key {key} from NIXL storage")
        except Exception as e:
            logger.error(f"Failed to delete key {key} from NIXL storage: {e}")

    def clear(self) -> None:
        """
        Clear all entries in NIXL storage.
        """
        try:
            # Clean up all files
            self.file_manager.cleanup_files()
            
            # Clean up all registrations
            self.registration.cleanup_registrations()
            
        except Exception as e:
            logger.error(f"Failed to clear NIXL storage: {e}")

    def register(self, tensor: torch.Tensor) -> bool:
        """
        Register a tensor with NIXL for optimized operations.
        Returns True if registration was successful, False otherwise.
        """
        return self.registration.register_tensor(tensor)

    def deregister(self, tensor: torch.Tensor) -> bool:
        """
        Deregister a tensor from NIXL.
        Returns True if deregistration was successful, False otherwise.
        """
        return self.registration.deregister_tensor(tensor)

    def __del__(self):
        """Cleanup when the object is destroyed."""
        try:
            self.clear()
        except Exception as e:
            logger.error(f"Failed to cleanup NIXL resources: {e}") 
