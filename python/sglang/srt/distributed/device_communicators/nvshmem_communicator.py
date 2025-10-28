"""
NVSHMEM Communicator for symmetric memory operations.
Simple, efficient implementation following torchrun initialization pattern.
"""

import logging
import os
from typing import Optional, Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

logger = logging.getLogger(__name__)

# Check NVSHMEM availability
nvshmem_is_available = False
if torch.cuda.is_available():
    try:
        from cuda.core.experimental import Device
        import nvshmem.core
        import nvshmem.bindings
        import numpy as np
        nvshmem_is_available = True
    except ImportError:
        nvshmem_is_available = False

def get_nvshmem_comm():
    """Get NVSHMEM communicator from world group if available"""
    try:
        from sglang.srt.distributed import get_world_group
        world_group = get_world_group()
        nvshmem_comm = getattr(world_group, 'nvshmem_comm', None)
        return nvshmem_comm if (nvshmem_comm and not nvshmem_comm.disabled) else None
    except Exception:
        return None


class NvshmemCommunicator:
    """Simple NVSHMEM communicator following torchrun pattern"""
    
    def __init__(self, group: ProcessGroup, device: Union[int, str, torch.device]):
        self.disabled = not nvshmem_is_available
        if self.disabled:
            logger.warning("NVSHMEM not available")
            return
            
        self.group = group
        self.world_size = dist.get_world_size(self.group)
        self.rank = dist.get_rank(self.group)
        self.local_rank = self.rank % 8
        
        # Track created tensors for unified cleanup
        self._created_tensors = []
        
        self._init_nvshmem()
        
    
    def _init_nvshmem(self):
        """Initialize NVSHMEM using UniqueID broadcast pattern"""
        # Set CUDA device
        torch.cuda.set_device(self.local_rank)
        
        # Create cuda.core Device
        dev = Device(self.local_rank)
        dev.set_current()
        # Create and broadcast unique ID
        uid = nvshmem.core.get_unique_id(empty=(self.rank != 0))
        uid_bytes = uid._data.view(np.uint8).copy()
        uid_tensor = torch.from_numpy(uid_bytes).cuda()
        dist.broadcast(uid_tensor, src=0, group=self.group)
        dist.barrier(group=self.group)
        uid._data[:] = uid_tensor.cpu().numpy().view(uid._data.dtype)
        
        # Initialize NVSHMEM
        nvshmem.core.init(
            device=dev, 
            uid=uid, 
            rank=self.rank, 
            nranks=self.world_size, 
            initializer_method="uid"
        )
        
        logger.info(f"NVSHMEM initialized: rank={self.rank}, world_size={self.world_size}")
    
    def create_symmetric_tensor(self, shape, dtype=torch.float32):
        """Create symmetric tensor and track for cleanup"""
        if self.disabled:
            return torch.zeros(shape, dtype=dtype, device=f"cuda:{self.local_rank}")
        
        tensor = nvshmem.core.tensor(shape, dtype=dtype)
        self._created_tensors.append(tensor)
        return tensor
    
    def create_multicast_pointer(self, tensor):
        """Create multicast pointer"""
        if self.disabled:
            return None
        local_ptr = tensor.data_ptr()
        return nvshmem.bindings.mc_ptr(nvshmem.core.Teams.TEAM_NODE, local_ptr)
    
    def cleanup(self):
        """Cleanup NVSHMEM tensors and finalize"""
        if not self.disabled:
            try:
                # Free all tracked tensors
                for tensor in self._created_tensors:
                    try:
                        nvshmem.core.free_tensor(tensor)
                    except Exception as e:
                        logger.warning(f"Failed to free NVSHMEM tensor: {e}")
                
                self._created_tensors.clear()
                
                # Synchronization barrier before finalization
                if dist.is_initialized():
                    dist.barrier(device_ids=[self.local_rank], group=self.group)
                
                # Finalize NVSHMEM
                nvshmem.core.finalize()
                logger.info("NVSHMEM tensors freed and finalized")
                
            except Exception as e:
                logger.warning(f"NVSHMEM cleanup failed: {e}")
    
    def get_tensor_count(self):
        """Get number of tracked tensors for debugging"""
        return len(self._created_tensors) if not self.disabled else 0
