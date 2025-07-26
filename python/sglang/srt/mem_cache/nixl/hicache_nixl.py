import hashlib
import logging
import os
import uuid
from typing import List, Optional, Union, Dict, Tuple

import torch

from sglang.srt.mem_cache.hicache_storage import HiCacheStorage
from .nixl_utils import NixlRegistration, NixlFileManager, NixlBackendSelection

# Import NIXL API
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
    """
    HiCacheNixl provides high-performance storage using NIXL file plugins.
    """
    def __init__(self, file_path: str = "/tmp/hicache_nixl", file_plugin: str = "auto"):
        """
        Initialize NIXL storage connector.
        Args:
            file_path: Path to the storage directory for NIXL file operations
            file_plugin: Plugin to use ("auto", "3FS", "POSIX", "GDS", "GDS_MT"). 
                        "auto" selects the best available plugin automatically.
        """
        # Initialize file management
        self.file_manager = NixlFileManager(file_path)
        
        # Initialize NIXL agent
        agent_config = nixl_agent_config(backends=[])
        self.agent_name = f"hicache_nixl_{str(uuid.uuid4())}"
        self.agent = nixl_agent(self.agent_name, agent_config)
        
        # Create backend with user-specified or automatic plugin selection
        self.backend_selector = NixlBackendSelection(file_plugin)
        if not self.backend_selector.create_backend(self.agent):
            raise Exception("Failed to create NIXL backend")
        
        # Initialize registration manager
        self.registration = NixlRegistration(self.agent)


    def _is_gpu_tensor(self, tensor: torch.Tensor) -> bool:
        """Check if tensor is on GPU memory."""
        return tensor.device.type == 'cuda'



    def _execute_transfer(
        self,
        tensors: List[torch.Tensor],
        file_paths: List[str],
        direction: str,  # "READ" or "WRITE"
    ) -> bool:
        """Execute a NIXL transfer operation."""
        reg_mem = self.registration.register_buffers(tensors)
        if not reg_mem:
            logger.error("Failed to register tensors")
            return False
  
        file_info = [(path, 0, 0) for path in file_paths]  
        
        file_tuples = self.file_manager.create_nixl_tuples(file_info)
        if not file_tuples:
            logger.error("Failed to create NIXL tuples for files")
            return False
        
        # Register files with NIXL
        reg_mem = self.registration.register_files(file_tuples)
        if not reg_mem:
            logger.error("Failed to register files")
            return False
        
        try:
            # Step 3: Create transfer descriptors
            # Determine memory type based on tensor device
            memory_type = "VRAM" if tensors[0].device.type == 'cuda' else "DRAM"
            
            # For tensors, we can use the tensors directly with explicit memory type
            tensor_descs = self.agent.get_xfer_descs(tensors, memory_type)
            if tensor_descs is None:
                logger.error("Failed to get tensor transfer descriptors")
                return False
            
            # Calculate tensor sizes in bytes for file tuple lengths
            tensor_sizes = [tensor.element_size() * tensor.numel() for tensor in tensors]
            
            # For files, convert 4-element tuples to 3-element tuples for transfer
            # Use tensor sizes to ensure file tuple lengths match tensor sizes
            transfer_tuples = self.file_manager.get_nixl_tuples_for_transfer(file_tuples, tensor_sizes)
            if not transfer_tuples:
                logger.error("Failed to convert file tuples for transfer")
                return False
            
            file_descs = self.agent.get_xfer_descs(transfer_tuples, "FILE")
            if file_descs is None:
                logger.error("Failed to get file transfer descriptors")
                return False
            
            # For file plugins: src is always tensors, dst is always files
            # Direction determines whether we're reading from or writing to files
            src_descs = tensor_descs  # Source is always tensors
            dst_descs = file_descs     # Destination is always files
            
            # Step 4: Create transfer request
            xfer_req = self.agent.initialize_xfer(
                direction,
                src_descs,
                dst_descs,
                self.agent_name
            )

            if xfer_req is None:
                logger.error("Failed to initialize transfer")
                return False
            
            state = self.agent.transfer(xfer_req)
            
            while state != "DONE":
                state = self.agent.check_xfer_state(xfer_req)
                if state == "ERR":
                    logger.error("Transfer failed")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to execute transfer: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False

    def batch_set(self, keys: List[str], values: List[torch.Tensor]) -> bool:
        if not keys:
            return True
        
        # Create files if needed
        file_paths = []
        for key in keys:
            tensor_path = self.file_manager.get_file_path(key)
            if not self.file_manager.create_file(tensor_path):
                logger.error(f"Failed to create file {tensor_path}")
                return False
            file_paths.append(tensor_path)
        
        # Execute WRITE transfer
        return self._execute_transfer(values, file_paths, "WRITE")

    def set(self, key: str, value: torch.Tensor) -> bool:
        return self.batch_set([key], [value])

    def get(
        self, key: str, dst_tensor: torch.Tensor
    ) -> torch.Tensor | None:
        result = self.batch_get([key], [dst_tensor])
        retrieved = result[0] if result else None
        return retrieved
    
    def batch_get(self, keys: List[str], dst_tensors: List[torch.Tensor]) -> List[Optional[torch.Tensor]]:
        if not keys:
            return []
        
        # Get file paths
        file_paths = [self.file_manager.get_file_path(key) for key in keys]
        
        # Execute READ transfer
        success = self._execute_transfer(dst_tensors, file_paths, "READ")
        return dst_tensors if success else [None] * len(keys)
    
   
    # Modify this to use new NIXL queryMem API 
    def exists(self, key: str) -> bool:
        """
        Check if the key exists in NIXL storage.
        Returns True if the key exists, False otherwise.
        """
        tensor_path = self.file_manager.get_file_path(key)
        return os.path.exists(tensor_path)