import os
import torch
from typing import Tuple
import atexit

GB = 1024 ** 3


class DiskKVCache:

    def __init__(self, size: int, page_size: int, dtype: torch.dtype,
                 head_num: int, head_dim: int, layer_num: int,
                 cache_dir: str = "./kv_cache"):
        self.size = size
        self.page_size = page_size
        self.dtype = dtype
        self.head_num = head_num
        self.head_dim = head_dim
        self.layer_num = layer_num
        self.cache_dir = cache_dir

        os.makedirs(cache_dir, exist_ok=True)

        self.k_files = []
        self.v_files = []
        self.k_buffers = []
        self.v_buffers = []

        total_elements = (size + page_size) * head_num * head_dim
        element_size = torch.tensor([], dtype=dtype).element_size()
        file_size = total_elements * element_size

        for i in range(layer_num):
            k_filepath = os.path.join(cache_dir, f"k_{i}.bin")
            v_filepath = os.path.join(cache_dir, f"v_{i}.bin")

            with open(k_filepath, "wb") as f:
                f.truncate(file_size)
            with open(v_filepath, "wb") as f:
                f.truncate(file_size)

            k_buffer = torch.from_file(
                k_filepath,
                dtype=dtype,
                size=total_elements,
                shared=True
            ).view(size + page_size, head_num, head_dim)

            v_buffer = torch.from_file(
                v_filepath,
                dtype=dtype,
                size=total_elements,
                shared=True
            ).view(size + page_size, head_num, head_dim)

            self.k_files.append(k_filepath)
            self.v_files.append(v_filepath)
            self.k_buffers.append(k_buffer)
            self.v_buffers.append(v_buffer)

        atexit.register(self.cleanup)

    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer_id < 0 or layer_id >= self.layer_num:
            raise ValueError(f"Invalid layer_id: {layer_id}")
        return self.k_buffers[layer_id], self.v_buffers[layer_id]

    def cleanup(self):
        try:
            import shutil
            shutil.rmtree(self.cache_dir)
        except:
            pass
