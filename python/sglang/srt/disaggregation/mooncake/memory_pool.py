import os

import torch
from torch.cuda.memory import CUDAPluggableAllocator


class CustomAllocator:
    _instances = {}

    @classmethod
    def get_allocator(cls, device):
        if device not in cls._instances:
            so_path = os.environ.get(
                "SGLANG_MOONCAKE_ALLOCATOR_SO_PATH", "/data/numa0/zbz1/hook.so"
            )
            allocator = CUDAPluggableAllocator(so_path, "my_malloc", "my_free")
            cls._instances[device] = allocator
        return cls._instances[device]
