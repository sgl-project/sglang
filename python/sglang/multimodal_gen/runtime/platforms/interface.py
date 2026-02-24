# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/platforms/interface.py
"""
Multimodal Gen Platform Interface - Re-exports from unified platform.

This file re-exports the Platform base class and enumerations from
sglang.platforms.interface for backward compatibility. The local Platform
class extends the unified one with multimodal_gen-specific methods.

New code should import directly from sglang.platforms.interface.
"""

from __future__ import annotations

import random
from functools import lru_cache
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import torch

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import resolve_obj_by_qualname

# Re-export from unified platform interface
from sglang.platforms.interface import (
    AttentionBackendEnum,
    CpuArchEnum,
    DeviceCapability,
)
from sglang.platforms.interface import Platform as UnifiedPlatform
from sglang.platforms.interface import (
    PlatformEnum,
    UnspecifiedPlatform,
)

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
        AttentionImpl,
    )

logger = init_logger(__name__)


class Platform(UnifiedPlatform):
    """
    Multimodal Gen Platform extension.

    Extends the unified Platform class with multimodal_gen-specific methods.
    This class maintains backward compatibility for existing code that imports
    from this module.
    """

    _enum: PlatformEnum
    device_name: str
    device_type: str
    device: torch.device | None = None  # Dummy attribute for compatibility

    dispatch_key: str = "CPU"
    simple_compile_backend: str = "inductor"

    # Use tuple (immutable) instead of list to avoid shared mutable state
    supported_quantization: ClassVar[tuple[str, ...]] = ()

    @lru_cache(maxsize=1)
    def is_cuda(self) -> bool:
        return self.is_cuda_static()

    @lru_cache(maxsize=1)
    def is_npu(self) -> bool:
        return self._enum == PlatformEnum.NPU

    @lru_cache(maxsize=1)
    def is_rocm(self) -> bool:
        return self.is_rocm_static()

    @lru_cache(maxsize=1)
    def is_tpu(self) -> bool:
        return self._enum == PlatformEnum.TPU

    @lru_cache(maxsize=1)
    def is_cpu(self) -> bool:
        return self._enum == PlatformEnum.CPU

    @classmethod
    @lru_cache(maxsize=1)
    def is_blackwell(cls):
        if not cls.is_cuda_static():
            return False
        return torch.cuda.get_device_capability()[0] == 10

    @classmethod
    @lru_cache(maxsize=1)
    def is_hopper(cls):
        if not cls.is_cuda_static():
            return False
        return torch.cuda.get_device_capability() == (9, 0)

    @classmethod
    @lru_cache(maxsize=1)
    def is_sm120(cls):
        if not cls.is_cuda_static():
            return False
        return torch.cuda.get_device_capability()[0] == 12

    @classmethod
    def is_cuda_static(cls) -> bool:
        return getattr(cls, "_enum", None) == PlatformEnum.CUDA

    @classmethod
    def is_rocm_static(cls) -> bool:
        return getattr(cls, "_enum", None) == PlatformEnum.ROCM

    @lru_cache(maxsize=1)
    def is_hpu(self) -> bool:
        return hasattr(torch, "hpu") and torch.hpu.is_available()

    @lru_cache(maxsize=1)
    def is_xpu(self) -> bool:
        return hasattr(torch, "xpu") and torch.xpu.is_available()

    @lru_cache(maxsize=1)
    def is_npu(self) -> bool:
        return hasattr(torch, "npu") and torch.npu.is_available()

    def is_out_of_tree(self) -> bool:
        return self._enum == PlatformEnum.OOT

    @lru_cache(maxsize=1)
    def is_cuda_alike(self) -> bool:
        """Returns True for CUDA-compatible platforms (CUDA, ROCm)."""
        # Note: MUSA is NOT included here - it uses mccl, not nccl
        return self._enum in (PlatformEnum.CUDA, PlatformEnum.ROCM)

    @lru_cache(maxsize=1)
    def is_mps(self) -> bool:
        return self._enum == PlatformEnum.MPS

    @lru_cache(maxsize=1)
    def is_musa(self):
        return self._enum == PlatformEnum.MUSA

    @lru_cache(maxsize=1)
    def is_hip(self) -> bool:
        return self.is_rocm()

    @classmethod
    @lru_cache(maxsize=1)
    def is_amp_supported(cls) -> bool:
        return True

    @classmethod
    def get_local_torch_device(cls) -> torch.device:
        raise NotImplementedError

    @classmethod
    def get_attn_backend_cls_str(
        cls,
        selected_backend: AttentionBackendEnum | None,
        head_size: int,
        dtype: torch.dtype,
    ) -> str:
        """Get the attention backend class of a device."""
        return ""

    @classmethod
    def get_device_capability(
        cls,
        device_id: int = 0,
    ) -> DeviceCapability | None:
        """Stateless version of :func:`torch.cuda.get_device_capability`."""
        return None

    @classmethod
    def has_device_capability(
        cls,
        capability: tuple[int, int] | int,
        device_id: int = 0,
    ) -> bool:
        """
        Test whether this platform is compatible with a device capability.

        The ``capability`` argument can either be:

        - A tuple ``(major, minor)``.
        - An integer ``<major><minor>``. (See :meth:`DeviceCapability.to_int`)
        """
        current_capability = cls.get_device_capability(device_id=device_id)
        if current_capability is None:
            return False

        if isinstance(capability, tuple):
            return current_capability >= capability

        return current_capability.to_int() >= capability

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        """Get the name of a device."""
        raise NotImplementedError

    @classmethod
    def get_device_uuid(cls, device_id: int = 0) -> str:
        """Get the uuid of a device, e.g. the PCI bus ID."""
        raise NotImplementedError

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        """Get the total memory of a device in bytes."""
        raise NotImplementedError

    @lru_cache(maxsize=1)
    def get_device(self, local_rank: int) -> torch.device:
        if self.is_cuda() or self.is_rocm():
            return torch.device("cuda", local_rank)
        elif self.is_npu():
            return torch.device("npu", local_rank)
        elif self.is_musa():
            return torch.device("musa", local_rank)
        elif self.is_mps():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    @lru_cache(maxsize=1)
    def get_torch_distributed_backend_str(self) -> str:
        """
        Get the torch.distributed backend string.

        Returns:
            "mccl" for MUSA, "nccl" for CUDA/ROCm, "gloo" for CPU/MPS/others
        """
        # Check MUSA first - it's CUDA-like but uses mccl
        if self.is_musa():
            return "mccl"
        elif self.is_cuda_alike():
            return "nccl"
        elif self.is_mps():
            return "gloo"
        else:
            return "gloo"

    @classmethod
    def is_async_output_supported(cls, enforce_eager: bool | None) -> bool:
        """
        Check if the current platform supports async output.
        """
        raise NotImplementedError

    @classmethod
    def inference_mode(cls):
        """A device-specific wrapper of `torch.inference_mode`.

        This wrapper is recommended because some hardware backends such as TPU
        do not support `torch.inference_mode`. In such a case, they will fall
        back to `torch.no_grad` by overriding this method.
        """
        return torch.inference_mode(mode=True)

    @classmethod
    def seed_everything(cls, seed: int | None = None) -> None:
        """
        Set the seed of each random module.
        `torch.manual_seed` will set seed on all devices.
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

    @classmethod
    def verify_model_arch(cls, model_arch: str) -> None:
        """
        Verify whether the current platform supports the specified model
        architecture.
        """
        pass

    @classmethod
    def verify_quantization(cls, quant: str) -> None:
        """
        Verify whether the quantization is supported by the current platform.
        """
        if cls.supported_quantization and quant not in cls.supported_quantization:
            raise ValueError(
                f"{quant} quantization is currently not supported in "
                f"{cls.device_name}."
            )

    @classmethod
    def get_current_memory_usage(
        cls, device: torch.types.Device | None = None
    ) -> float:
        """
        Return the memory usage in bytes.
        """
        raise NotImplementedError

    @classmethod
    def get_available_gpu_memory(
        cls,
        device_id: int = 0,
        distributed: bool = False,
        empty_cache: bool = True,
        cpu_group: Any = None,
    ) -> float:
        """
        Return the available memory in GiB.
        """
        raise NotImplementedError

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        """
        Get device specific communicator class for distributed communication.
        """
        return "sglang.multimodal_gen.runtime.distributed.device_communicators.base_device_communicator.DeviceCommunicatorBase"  # noqa

    @classmethod
    def get_cpu_architecture(cls) -> CpuArchEnum:
        """Get the CPU architecture of the current platform."""
        return CpuArchEnum.UNSPECIFIED

    @classmethod
    def enable_dit_layerwise_offload_for_wan_by_default(cls) -> bool:
        """Whether to enable DIT layerwise offload by default on the current platform."""
        return True

    def get_attn_backend(self, *args, **kwargs) -> AttentionImpl:
        attention_cls_str = self.get_attn_backend_cls_str(*args, **kwargs)
        return resolve_obj_by_qualname(attention_cls_str)


# Re-export UnspecifiedPlatform from unified module (no local redefinition needed)
__all__ = [
    "Platform",
    "UnifiedPlatform",
    "UnspecifiedPlatform",
    "PlatformEnum",
    "DeviceCapability",
    "AttentionBackendEnum",
    "CpuArchEnum",
]
