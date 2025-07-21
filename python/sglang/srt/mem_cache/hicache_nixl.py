import hashlib
import logging
import os
import uuid
from typing import List, Optional, Dict, Tuple

import torch

from sglang.srt.mem_cache.hicache_storage import HiCacheStorage

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
        
        self.file_path = file_path
        self.file_plugin = file_plugin
        if not os.path.exists(self.file_path):
            os.makedirs(self.file_path)
            logger.info(f"Created HiCacheNixl storage directory at {self.file_path}")
        
        # Initialize NIXL agent
        agent_config = nixl_agent_config(backends=[])
        self.agent = nixl_agent(str(uuid.uuid4()), agent_config)
        
        # Create backend with user-specified or automatic plugin selection
        self._create_backend()
        
        # Track registered tensors and file descriptors
        self.registered_tensors = {}  # tensor_id -> tensor_descs
        self.registered_files = {}    # file_path -> (fd, file_descs)
        
        # Pre-open all existing files for better performance
        self._pre_open_existing_files()

    def _pre_open_existing_files(self):
        """Pre-open all existing files in the storage directory for better performance."""
        try:
            if not os.path.exists(self.file_path):
                return
            
            existing_files = []
            for filename in os.listdir(self.file_path):
                if filename.endswith('.bin'):
                    file_path = os.path.join(self.file_path, filename)
                    if os.path.isfile(file_path):
                        existing_files.append(file_path)
            
            if existing_files:
                logger.info(f"Pre-opening {len(existing_files)} existing files for NIXL operations")
                self._register_files_batch(existing_files)
                
        except Exception as e:
            logger.warning(f"Failed to pre-open existing files: {e}")

    def _refresh_pre_opened_files(self):
        """Refresh the list of pre-opened files by scanning the storage directory."""
        try:
            # Get current registered files
            current_files = set(self.registered_files.keys())
            
            # Scan for all .bin files in storage directory
            existing_files = set()
            if os.path.exists(self.file_path):
                for filename in os.listdir(self.file_path):
                    if filename.endswith('.bin'):
                        file_path = os.path.join(self.file_path, filename)
                        if os.path.isfile(file_path):
                            existing_files.add(file_path)
            
            # Find new files that need to be opened
            new_files = existing_files - current_files
            if new_files:
                logger.info(f"Opening {len(new_files)} new files for NIXL operations")
                self._register_files_batch(list(new_files))
                
        except Exception as e:
            logger.warning(f"Failed to refresh pre-opened files: {e}")

    def _ensure_file_opened(self, file_path: str) -> bool:
        """Ensure a file is opened and registered with NIXL, opening it if necessary."""
        if file_path not in self.registered_files:
            return self._register_files_batch([file_path])
        return True

    def _handle_new_file_creation(self, file_path: str) -> bool:
        """Handle newly created files by opening and registering them with NIXL."""
        try:
            # Create the file if it doesn't exist
            if not os.path.exists(file_path):
                # Create parent directory if needed
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                # Touch the file to create it
                open(file_path, 'a').close()
            
            # Register the new file
            return self._register_files_batch([file_path])
        except Exception as e:
            logger.error(f"Failed to handle new file creation for {file_path}: {e}")
            return False

    def _create_backend(self):
        """Create NIXL backend with user-specified or automatic plugin selection."""
        try:
            plugin_list = self.agent.get_plugin_list()
            logger.debug(f"Available NIXL plugins: {plugin_list}")
            
            if self.file_plugin == "auto":
                # Automatic selection with priority order: GDS_MT -> 3FS -> POSIX
                if "GDS_MT" in plugin_list:
                    self.agent.create_backend("GDS_MT")
                    self.backend_name = "GDS_MT"
                    logger.info("Auto-selected GDS_MT backend")
                elif "3FS" in plugin_list:
                    self.agent.create_backend("3FS")
                    self.backend_name = "3FS"
                    logger.info("Auto-selected 3FS backend")
                else:
                    self.backend_name = "POSIX"
                    logger.info("Auto-selected POSIX backend (no NIXL backend created)")
            else:
                # User-specified plugin
                if self.file_plugin in plugin_list:
                    self.agent.create_backend(self.file_plugin)
                    self.backend_name = self.file_plugin
                    logger.info(f"Created {self.file_plugin} backend as requested")
                else:
                    logger.warning(f"Requested plugin '{self.file_plugin}' not available. Available plugins: {plugin_list}")
                    self.backend_name = "POSIX"
                    logger.info("Falling back to POSIX backend")
        except Exception as e:
            logger.warning(f"Failed to create NIXL backend: {e}")
            self.backend_name = "POSIX"

    def _is_gpu_tensor(self, tensor: torch.Tensor) -> bool:
        """Check if tensor is on GPU memory."""
        return tensor.device.type == 'cuda'

    def _get_file_path(self, key: str) -> str:
        """Get the file path for a given key."""
        return os.path.join(self.file_path, f"{key}.bin")

    def _register_tensor(self, tensor: torch.Tensor) -> bool:
        """Register a tensor with NIXL for I/O operations."""
        return self._register_tensors_batch([tensor])

    def _register_file(self, file_path: str) -> bool:
        """Register a file with NIXL for I/O operations using OS file descriptor."""
        return self._register_files_batch([file_path])

    def _register_tensors_batch(self, tensors: List[torch.Tensor]) -> bool:
        """Register multiple tensors with NIXL in a single batch operation."""
        try:
            # Separate GPU and CPU tensors for batch registration
            gpu_tensor_addrs = []
            cpu_tensor_addrs = []
            gpu_tensor_ids = []
            cpu_tensor_ids = []
            
            for tensor in tensors:
                tensor_id = id(tensor)
                if tensor_id in self.registered_tensors:
                    continue  # Already registered
                
                if self._is_gpu_tensor(tensor):
                    tensor_ptr = tensor.data_ptr()
                    tensor_size = tensor.numel() * tensor.element_size()
                    gpu_id = tensor.device.index if tensor.device.index is not None else 0
                    
                    gpu_tensor_addrs.append((tensor_ptr, tensor_size, gpu_id, "gpu_tensor"))
                    gpu_tensor_ids.append(tensor_id)
                else:
                    tensor_ptr = tensor.data_ptr()
                    tensor_size = tensor.numel() * tensor.element_size()
                    
                    cpu_tensor_addrs.append((tensor_ptr, tensor_size, 0, "cpu_tensor"))
                    cpu_tensor_ids.append(tensor_id)
            
            # Batch register GPU tensors
            if gpu_tensor_addrs:
                gpu_tensor_descs = self.agent.register_memory(gpu_tensor_addrs, "VRAM", is_sorted=False)
                if gpu_tensor_descs:
                    for tensor_id, tensor_desc in zip(gpu_tensor_ids, gpu_tensor_descs):
                        self.registered_tensors[tensor_id] = tensor_desc
                    logger.debug(f"Batch registered {len(gpu_tensor_addrs)} GPU tensors with NIXL")
                else:
                    logger.warning("Failed to batch register GPU tensors with NIXL")
                    return False
            
            # Batch register CPU tensors
            if cpu_tensor_addrs:
                cpu_tensor_descs = self.agent.register_memory(cpu_tensor_addrs, "DRAM", is_sorted=False)
                if cpu_tensor_descs:
                    for tensor_id, tensor_desc in zip(cpu_tensor_ids, cpu_tensor_descs):
                        self.registered_tensors[tensor_id] = tensor_desc
                    logger.debug(f"Batch registered {len(cpu_tensor_addrs)} CPU tensors with NIXL")
                else:
                    logger.warning("Failed to batch register CPU tensors with NIXL")
                    return False
            
            return True
                    
        except Exception as e:
            logger.error(f"Failed to batch register tensors with NIXL: {e}")
            return False

    def _register_files_batch(self, file_paths: List[str]) -> bool:
        """Register multiple files with NIXL in a single batch operation."""
        try:
            # Collect unregistered files
            unregistered_files = []
            fds = []
            
            for file_path in file_paths:
                if file_path not in self.registered_files:
                    try:
                        # Open file and get OS file descriptor
                        fd = os.open(file_path, os.O_RDWR | os.O_CREAT)
                        unregistered_files.append(file_path)
                        fds.append(fd)
                    except Exception as e:
                        logger.error(f"Failed to open file {file_path}: {e}")
                        return False
            
            # Batch register file descriptors with NIXL
            if fds:
                file_descs_list = self.agent.register_file_fd(fds)
                if file_descs_list and len(file_descs_list) == len(fds):
                    for file_path, fd, file_descs in zip(unregistered_files, fds, file_descs_list):
                        self.registered_files[file_path] = (fd, file_descs)
                    logger.debug(f"Batch registered {len(fds)} files with NIXL")
                    return True
                else:
                    # Clean up opened file descriptors on failure
                    for fd in fds:
                        os.close(fd)
                    logger.warning("Failed to batch register files with NIXL")
                    return False
            
            return True
                    
        except Exception as e:
            logger.error(f"Failed to batch register files with NIXL: {e}")
            # Clean up opened file descriptors on exception
            for fd in fds:
                try:
                    os.close(fd)
                except:
                    pass
            return False

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
        if dst_tensor is None:
            logger.error("NIXL get() requires dst_tensor for pure NIXL datapath")
            return None
            
        tensor_path = self._get_file_path(key)
        
        if not os.path.exists(tensor_path):
            return None
        
        try:
            # Register file if not already registered
            if not self._ensure_file_opened(tensor_path):
                logger.error(f"Failed to register file {tensor_path}")
                return None
            
            # Register destination tensor (only if not already registered)
            if id(dst_tensor) not in self.registered_tensors:
                if not self._register_tensors_batch([dst_tensor]):
                    logger.error(f"Failed to register destination tensor for key {key}")
                    return None
            
            # Use NIXL transfer with registered buffers
            xfer_req = self.agent.initialize_xfer(
                src_file=self.registered_files[tensor_path][1],  # file_descs
                dst_tensor=self.registered_tensors[id(dst_tensor)],
                xfer_type="read"
            )
            self.agent.transfer(xfer_req)
            
            # Wait for transfer completion
            while True:
                state = self.agent.check_xfer_state(xfer_req)
                if state == "completed":
                    break
                elif state == "failed":
                    logger.error(f"Transfer failed for key {key}")
                    return None
                # Continue waiting for completion
            
            return dst_tensor
                
        except Exception as e:
            logger.error(f"Failed to get key {key} from NIXL storage: {e}")
            return None

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
                if id(dst_tensors[i]) not in self.registered_tensors:
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
            try:
                # Collect registered descriptors for batch transfer
                valid_tensor_descs = []
                valid_file_descs = []
                
                for tensor, file_path in zip(valid_tensors, valid_file_paths):
                    valid_tensor_descs.append(self.registered_tensors[id(tensor)])
                    valid_file_descs.append(self.registered_files[file_path][1])  # file_descs
                
                # Create batch transfer with all tensors and file descriptors at once
                xfer_req = self.agent.initialize_xfer(
                    src_files=valid_file_descs,  # List of file descriptors
                    dst_tensors=valid_tensor_descs,   # List of tensor descriptors
                    xfer_type="read"
                )
                
                # Submit batch transfer
                self.agent.transfer(xfer_req)
                
                # Wait for batch transfer completion
                while True:
                    state = self.agent.check_xfer_state(xfer_req)
                    if state == "completed":
                        break
                    elif state == "failed":
                        logger.error(f"Batch transfer failed")
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
        tensor_path = self._get_file_path(key)
        
        # Check if key exists and handle overwrite logic
        if os.path.exists(tensor_path):
            if not overwrite:
                logger.debug(f"Key {key} already exists. Skipped (use overwrite=True to overwrite).")
                return True
            else:
                logger.debug(f"Overwriting existing key {key}")
        
        try:
            # Register source tensor (only if not already registered)
            if id(value) not in self.registered_tensors:
                if not self._register_tensors_batch([value]):
                    logger.error(f"Failed to register source tensor for key {key}")
                    return False
            
            # Handle file (create if new, ensure opened if existing)
            if not os.path.exists(tensor_path):
                if not self._handle_new_file_creation(tensor_path):
                    logger.error(f"Failed to create and register file {tensor_path}")
                    return False
            else:
                if not self._ensure_file_opened(tensor_path):
                    logger.error(f"Failed to register file {tensor_path}")
                    return False
            
            # Use NIXL transfer with registered buffers
            xfer_req = self.agent.initialize_xfer(
                src_tensor=self.registered_tensors[id(value)],
                dst_file=self.registered_files[tensor_path][1],  # file_descs
                xfer_type="write"
            )
            self.agent.transfer(xfer_req)
            
            # Wait for transfer completion
            while True:
                state = self.agent.check_xfer_state(xfer_req)
                if state == "completed":
                    break
                elif state == "failed":
                    logger.error(f"Transfer failed for key {key}")
                    return False
                # Continue waiting for completion
            
            return True
        except Exception as e:
            logger.error(f"Failed to set key {key} in NIXL storage: {e}")
            return False

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
                if id(value) not in self.registered_tensors:
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
            try:
                # Collect registered descriptors for batch transfer
                valid_tensor_descs = []
                valid_file_descs = []
                
                for tensor, file_path in zip(valid_tensors, valid_file_paths):
                    valid_tensor_descs.append(self.registered_tensors[id(tensor)])
                    valid_file_descs.append(self.registered_files[file_path][1])  # file_descs
                
                # Create batch transfer with all tensors and file descriptors at once
                xfer_req = self.agent.initialize_xfer(
                    src_tensors=valid_tensor_descs,    # List of tensor descriptors
                    dst_files=valid_file_descs,   # List of file descriptors
                    xfer_type="write"
                )
                
                # Submit batch transfer
                self.agent.transfer(xfer_req)
                
                # Wait for batch transfer completion
                while True:
                    state = self.agent.check_xfer_state(xfer_req)
                    if state == "completed":
                        break
                    elif state == "failed":
                        logger.error(f"Batch transfer failed")
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
                if tensor_path in self.registered_files:
                    fd, file_descs = self.registered_files[tensor_path]
                    self.agent.deregister_file(file_descs)
                    os.close(fd)
                    del self.registered_files[tensor_path]
                logger.debug(f"Successfully deleted key {key} from NIXL storage")
        except Exception as e:
            logger.error(f"Failed to delete key {key} from NIXL storage: {e}")

    def clear(self) -> None:
        """
        Clear all entries in NIXL storage.
        """
        try:
            for filename in os.listdir(self.file_path):
                file_path = os.path.join(self.file_path, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    # Clean up registered file if it exists
                    if file_path in self.registered_files:
                        fd, file_descs = self.registered_files[file_path]
                        self.agent.deregister_file(file_descs)
                        os.close(fd)
                        del self.registered_files[file_path]
            logger.info("Successfully cleared all entries in HiCacheNixl storage")
        except Exception as e:
            logger.error(f"Failed to clear HiCacheNixl storage: {e}")

    def register(self, tensor: torch.Tensor) -> bool:
        """
        Register a tensor with NIXL for optimized operations.
        Returns True if registration was successful, False otherwise.
        """
        return self._register_tensor(tensor)

    def deregister(self, tensor: torch.Tensor) -> bool:
        """
        Deregister a tensor from NIXL.
        Returns True if deregistration was successful, False otherwise.
        """
        try:
            tensor_id = id(tensor)
            if tensor_id in self.registered_tensors:
                tensor_descs = self.registered_tensors[tensor_id]
                self.agent.deregister_memory(tensor_descs)
                del self.registered_tensors[tensor_id]
                logger.debug("Successfully deregistered tensor from NIXL")
                return True
            else:
                logger.debug("Tensor not registered with NIXL")
                return True
        except Exception as e:
            logger.error(f"Failed to deregister tensor from NIXL: {e}")
            return False

    def __del__(self):
        """Cleanup when the object is destroyed."""
        try:
            # Deregister all registered tensors
            for tensor_descs in self.registered_tensors.values():
                self.agent.deregister_memory(tensor_descs)
            self.registered_tensors.clear()
            
            # Deregister all registered files and close fds
            for fd, file_descs in self.registered_files.values():
                self.agent.deregister_file(file_descs)
                os.close(fd)
            self.registered_files.clear()
            
        except Exception as e:
            logger.warning(f"Failed to cleanup NIXL resources: {e}") 