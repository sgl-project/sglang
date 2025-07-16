import os
import torch
from threading import RLock
from typing import Tuple, Dict


class DiskKVCache:
    """
    Thread-safe disk-based KV cache using memory-mapped files
    Implements persistent storage for KV cache with layer-level locking
    """

    def __init__(self,
                 size: int,
                 page_size: int,
                 dtype: torch.dtype,
                 head_num: int,
                 head_dim: int,
                 layer_num: int,
                 cache_dir: str = "./kv_cache"):
        """
        Initialize disk cache with memory-mapped files

        Args:
            size: Total cache capacity in tokens
            page_size: Size of each memory page
            dtype: Data type for storage (e.g., torch.float16)
            head_num: Number of attention heads
            head_dim: Dimension of each attention head
            layer_num: Number of transformer layers
            cache_dir: Directory for cache files
        """
        self.size = size
        self.page_size = page_size
        self.dtype = dtype
        self.head_num = head_num
        self.head_dim = head_dim
        self.layer_num = layer_num
        self.cache_dir = cache_dir
        self._file_locks: Dict[int, RLock] = {}  # Per-layer file locks

        os.makedirs(cache_dir, exist_ok=True)

        # Initialize memory-mapped buffers
        self.k_buffers = []
        self.v_buffers = []
        for i in range(layer_num):
            self._file_locks[i] = RLock()
            self._initialize_layer_file(i)

    def _initialize_layer_file(self, layer_id: int):
        """Create and initialize memory-mapped file for single layer"""
        file_path = os.path.join(self.cache_dir, f"layer_{layer_id}.bin")

        # Calculate required file size
        element_size = torch.tensor([], dtype=self.dtype).element_size()
        file_size = (self.size + self.page_size) * self.head_num * self.head_dim * element_size

        # Pre-allocate file space
        with open(file_path, "wb") as f:
            f.truncate(file_size)

        # Create memory-mapped tensors
        k_buffer = torch.from_file(
            file_path,
            dtype=self.dtype,
            size=(self.size + self.page_size) * self.head_num * self.head_dim,
            shared=True  # Enable multi-process sharing
        ).view(self.size + self.page_size, self.head_num, self.head_dim)

        self.k_buffers.append(k_buffer)
        self.v_buffers.append(k_buffer.clone())

    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get KV buffers for specified layer with lock protection

        Args:
            layer_id: Layer index to retrieve

        Returns:
            Tuple of (key_buffer, value_buffer) tensors
        """
        with self._file_locks[layer_id]:
            return self.k_buffers[layer_id], self.v_buffers[layer_id]

    def cleanup(self):
        """Clean up all cache resources"""
        for lock in self._file_locks.values():
            with lock:  # Ensure all operations complete
                pass
        try:
            import shutil
            shutil.rmtree(self.cache_dir)
        except Exception as e:
            print(f"Cache cleanup error: {str(e)}")
