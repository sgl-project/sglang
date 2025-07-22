import os
import torch
from threading import RLock
from typing import Tuple, Dict

GB = 1024 * 1024 * 1024
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
                 max_capacity_gb: int = 10,
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
        self.max_capacity_gb = max_capacity_gb
        self.available_layers = set()

        # check the capacity
        self._check_capacity()

        os.makedirs(cache_dir, exist_ok=True)

        # Initialize memory-mapped buffers
        self.k_buffers = []
        self.v_buffers = []
        for i in range(layer_num):
            self._file_locks[i] = RLock()
            self._initialize_layer_file(i)

    def _check_capacity(self):
        """Check if the maximum capacity limit is exceeded"""
        if self.max_capacity_gb is None:
            return

        total_elements = (self.size + self.page_size) * self.head_num * self.head_dim
        element_size = torch.tensor([], dtype=self.dtype).element_size()
        total_size_gb = (total_elements * element_size * self.layer_num * 2) / GB

        if total_size_gb > self.max_capacity_gb:
            raise ValueError(
                f"Requested cache size {total_size_gb:.2f}GB "
                f"exceeds maximum capacity {self.max_capacity_gb}GB"
            )

    def _initialize_layer_file(self, layer_id: int):
        """Create and initialize memory-mapped file for single layer"""
        file_path = os.path.join(self.cache_dir, f"layer_{layer_id}.bin")

        # Calculate required file size
        element_size = torch.tensor([], dtype=self.dtype).element_size()
        file_size = (self.size + self.page_size) * self.head_num * self.head_dim * element_size * 2

        # Pre-allocate file space
        with open(file_path, "wb") as f:
            f.truncate(file_size)

        # Create memory-mapped tensors
        cache_buffer = torch.from_file(
            file_path,
            dtype=self.dtype,
            size=(self.size + self.page_size) * self.head_num * self.head_dim * 2,
            shared=True  # Enable multi-process sharing
        ).view(2, self.size + self.page_size, self.head_num, self.head_dim)

        self.k_buffers.append(cache_buffer[0])
        self.v_buffers.append(cache_buffer[1])

    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get KV buffers for specified layer with lock protection

        Args:
            layer_id: Layer index to retrieve

        Returns:
            Tuple of (key_buffer, value_buffer) tensors
        """
        with self._file_locks[layer_id]:
            if layer_id not in self.available_layers:
                raise RuntimeError(f"layer {layer_id}'s kvcache is not set")
            else:
                return self.k_buffers[layer_id], self.v_buffers[layer_id]

    def set_kv_buffer(self, layer_id: int, loc: int, k_cache: torch.Tensor, v_cache: torch.Tensor):
        with self._file_locks[layer_id]:
            self.available_layers.add(layer_id)
            self.k_buffers[layer_id][loc] = k_cache.cpu()
            self.v_buffers[layer_id][loc] = v_cache.cpu()

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
