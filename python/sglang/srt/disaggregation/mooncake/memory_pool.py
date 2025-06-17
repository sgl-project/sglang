import os
import threading
from importlib import resources
from typing import Dict, Final, Optional

import torch
from torch.cuda.memory import CUDAPluggableAllocator


# TODO(shangming): move this class into mooncake's package for more general use cases
class MooncakeNVLinkAllocator:
    _instances: Dict[torch.device, CUDAPluggableAllocator] = {}
    _lock: Final = threading.Lock()

    @classmethod
    def _get_so_path(cls) -> str:
        """Dynamically locate hook.so in the mooncake package installation"""
        try:
            # Attempt to locate package resource
            with resources.path("mooncake", "hook.so") as so_path:
                if so_path.exists():
                    return str(so_path)
        except (ImportError, FileNotFoundError, TypeError):
            pass

        # Fallback strategy: check in package location via import metadata
        try:
            import mooncake

            base_path = os.path.dirname(os.path.abspath(mooncake.__file__))
            so_path = os.path.join(base_path, "hook.so")
            if os.path.exists(so_path):
                return so_path
        except (ImportError, FileNotFoundError, TypeError):
            raise ImportError(
                "SGLANG_MOONCAKE_CUSTOM_MEM_POOL require mooncake-transfer-engine >= 0.3.3.post2."
            )

    @classmethod
    def get_allocator(cls, device: torch.device) -> CUDAPluggableAllocator:
        with cls._lock:
            if device not in cls._instances:
                so_path = cls._get_so_path()
                cls._instances[device] = CUDAPluggableAllocator(
                    so_path, "mc_nvlink_malloc", "mc_nvlink_free"
                )
            return cls._instances[device]
