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
            
            # Determine memory type based on tensor device
            if tensor.is_cuda:
                mem_type = "VRAM"
                device_id = tensor.device.index
            else:
                mem_type = "DRAM"
                device_id = 0
            
            # Create registration descriptor
            tensor_addr = (tensor.data_ptr(), tensor.numel() * tensor.element_size(), device_id, "")
            
            # Register with NIXL
            tensor_descs = self.agent.register_memory([tensor_addr], mem_type, is_sorted=False)
            if tensor_descs:
                self.registered_tensors[tensor_id] = tensor_descs
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
            return True
        
        try:
            # Filter out already registered tensors
            unregistered_tensors = []
            for tensor in tensors:
                if id(tensor) not in self.registered_tensors:
                    unregistered_tensors.append(tensor)
            
            if not unregistered_tensors:
                logger.debug("All tensors already registered")
                return True
            
            # Group tensors by memory type
            dram_tensors = []
            vram_tensors = []
            
            for tensor in unregistered_tensors:
                if tensor.is_cuda:
                    vram_tensors.append(tensor)
                else:
                    dram_tensors.append(tensor)
            
            # Register DRAM tensors
            if dram_tensors:
                dram_addrs = []
                for tensor in dram_tensors:
                    dram_addrs.append((tensor.data_ptr(), tensor.numel() * tensor.element_size(), 0, ""))
                
                dram_descs = self.agent.register_memory(dram_addrs, "DRAM", is_sorted=False)
                if dram_descs:
                    for tensor in dram_tensors:
                        self.registered_tensors[id(tensor)] = dram_descs
                    logger.debug(f"Batch registered {len(dram_tensors)} DRAM tensors")
                else:
                    logger.error("Failed to batch register DRAM tensors")
                    return False
            
            # Register VRAM tensors
            if vram_tensors:
                vram_addrs = []
                for tensor in vram_tensors:
                    vram_addrs.append((tensor.data_ptr(), tensor.numel() * tensor.element_size(), tensor.device.index, ""))
                
                vram_descs = self.agent.register_memory(vram_addrs, "VRAM", is_sorted=False)
                if vram_descs:
                    for tensor in vram_tensors:
                        self.registered_tensors[id(tensor)] = vram_descs
                    logger.debug(f"Batch registered {len(vram_tensors)} VRAM tensors")
                else:
                    logger.error("Failed to batch register VRAM tensors")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to batch register tensors: {e}")
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
                logger.info(f"File list for registration: {file_list}")
                logger.info(f"About to call register_memory with file_list type: {type(file_list)}")
                logger.info(f"File list length: {len(file_list)}")
                file_descs_list = self.agent.register_memory(file_list, "FILE", is_sorted=False)
                if file_descs_list:
                    for file_path, fd in zip(unregistered_files, fds):
                        file_manager.registered_files[file_path] = (fd, file_descs_list)
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
    
    def deregister_tensor(self, tensor: torch.Tensor) -> bool:
        """Deregister a tensor from NIXL."""
        try:
            tensor_id = id(tensor)
            if tensor_id not in self.registered_tensors:
                logger.debug(f"Tensor {tensor_id} not registered")
                return True
            
            tensor_descs = self.registered_tensors[tensor_id]
            try:
                self.agent.deregister_memory(tensor_descs)
                logger.debug(f"Deregistered tensor {tensor_id}")
            except Exception as e:
                logger.warning(f"Failed to deregister tensor {tensor_id} from NIXL (may already be deregistered): {e}")
            finally:
                # Always remove from our tracking, even if NIXL deregistration failed
                del self.registered_tensors[tensor_id]
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to deregister tensor: {e}")
            return False
    
    def deregister_tensors_batch(self, tensors: List[torch.Tensor]) -> bool:
        """Deregister multiple tensors from NIXL in batch."""
        if not tensors:
            return True
        
        try:
            # Group tensors by their descriptors to avoid double deregistration
            descriptors_to_deregister = []
            tensors_to_remove = []
            
            for tensor in tensors:
                tensor_id = id(tensor)
                if tensor_id in self.registered_tensors:
                    desc = self.registered_tensors[tensor_id]
                    # Check if we already have this descriptor
                    if desc not in descriptors_to_deregister:
                        descriptors_to_deregister.append(desc)
                    tensors_to_remove.append(tensor_id)
            
            # Deregister each unique descriptor only once
            for desc in descriptors_to_deregister:
                try:
                    self.agent.deregister_memory(desc)
                    logger.debug(f"Successfully deregistered descriptor in batch")
                except Exception as e:
                    logger.warning(f"Failed to deregister descriptor in batch (may already be deregistered): {e}")
            
            # Remove all tensors from tracking
            for tensor_id in tensors_to_remove:
                if tensor_id in self.registered_tensors:
                    del self.registered_tensors[tensor_id]
            
            return True
        except Exception as e:
            logger.error(f"Failed to batch deregister tensors: {e}")
            return False
    
    def cleanup_registrations(self):
        """Clean up all registrations."""
        try:
            # Deregister all tensors
            tensor_ids = list(self.registered_tensors.keys())
            for tensor_id in tensor_ids:
                try:
                    tensor_descs = self.registered_tensors[tensor_id]
                    self.agent.deregister_memory(tensor_descs)
                    logger.debug(f"Successfully deregistered tensor {tensor_id}")
                except Exception as e:
                    logger.warning(f"Failed to deregister tensor {tensor_id} from NIXL (may already be deregistered): {e}")
                finally:
                    # Always remove from our tracking, even if NIXL deregistration failed
                    if tensor_id in self.registered_tensors:
                        del self.registered_tensors[tensor_id]
            
            self.registered_tensors.clear()
            logger.debug("Cleaned up all tensor registrations")
            
        except Exception as e:
            logger.error(f"Failed to cleanup registrations: {e}") 