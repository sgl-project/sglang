"""OOT (Out-Of-Tree) device configuration registry.

Allows third-party plugins to register custom device configurations
without modifying SGLang upstream code. This enables non-CUDA devices
(NPU, XPU, MLU, etc.) to plug in via a single registration call.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import torch

logger = logging.getLogger(__name__)

_oot_device_config: Optional["OOTDeviceConfig"] = None


@dataclass
class OOTDeviceConfig:
    """Plugin-registered device configuration.

    Attributes:
        device_type: Device type string, e.g. "cuda", "npu", "mlu"
        torch_device_module: The torch device module (e.g. torch.cuda, torch.npu)
        dist_backend: Distributed backend name, e.g. "nccl", "hccl", "flagcx"
        empty_cache_fn: Optional function to clear device cache
        device_count_fn: Optional function to get device count
    """

    device_type: str
    dist_backend: str = "nccl"
    empty_cache_fn: Optional[Callable[[], None]] = None
    device_count_fn: Optional[Callable[[], int]] = None

    def get_device(self, local_rank: int) -> torch.device:
        """Return a torch.device for the given local rank."""
        return torch.device(f"{self.device_type}:{local_rank}")

    def empty_cache(self) -> None:
        """Clear device memory cache."""
        if self.empty_cache_fn is not None:
            self.empty_cache_fn()
        else:
            # Fallback: try torch.{device_type}.empty_cache()
            mod = getattr(torch, self.device_type, None)
            if mod is not None and hasattr(mod, "empty_cache"):
                mod.empty_cache()

    def get_device_count(self) -> int:
        """Return the number of available devices."""
        if self.device_count_fn is not None:
            return self.device_count_fn()
        mod = getattr(torch, self.device_type, None)
        if mod is not None and hasattr(mod, "device_count"):
            return mod.device_count()
        return 0


def register_oot_device(config: OOTDeviceConfig) -> None:
    """Register an OOT device configuration (called by plugins)."""
    global _oot_device_config
    _oot_device_config = config
    logger.info(
        f"Registered OOT device: type={config.device_type}, "
        f"dist_backend={config.dist_backend}"
    )


def get_oot_device_config() -> Optional[OOTDeviceConfig]:
    """Return the registered OOT device config, or None."""
    return _oot_device_config
