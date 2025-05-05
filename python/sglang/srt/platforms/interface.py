# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/vllm-project/vllm/blob/v0.8.2/vllm/platforms/interface.py

import enum
import platform
import random
from platform import uname
from typing import TYPE_CHECKING, Any, NamedTuple, Optional, Tuple, Union

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.configs.model_config import ModelConfig

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


def in_wsl() -> bool:
    # Reference: https://github.com/microsoft/WSL/issues/4071
    return "microsoft" in " ".join(uname()).lower()


class PlatformEnum(enum.Enum):
    CUDA = enum.auto()
    ROCM = enum.auto()
    HPU = enum.auto()
    XPU = enum.auto()
    CPU = enum.auto()
    OOT = enum.auto()
    UNSPECIFIED = enum.auto()


class CpuArchEnum(enum.Enum):
    X86 = enum.auto()
    ARM = enum.auto()
    POWERPC = enum.auto()
    OTHER = enum.auto()
    UNKNOWN = enum.auto()


class DeviceCapability(NamedTuple):
    major: int
    minor: int

    def as_version_str(self) -> str:
        return f"{self.major}.{self.minor}"

    def to_int(self) -> int:
        """
        Express device capability as an integer ``<major><minor>``.

        It is assumed that the minor version is always a single digit.
        """
        assert 0 <= self.minor < 10
        return self.major * 10 + self.minor


class Platform:
    _enum: PlatformEnum

    # Real device name of current platform.
    device_name: str

    # For specifying torch device for cuda alike platform's capability.
    device_type: str

    #  The torch.distributed backend on current platform
    torch_distributed_backend: str

    # The torch.compile backend for compiling simple and
    # standalone functions. The default value is "inductor" to keep
    # the same behavior as PyTorch.
    torch_compile_backend: str = "inductor"

    supported_quantization: list[str] = []

    supported_speculative_algorithm: list[str] = []

    # Use first element as default dtype
    supported_dtype: list[str] = []

    # Use first element as default backend
    supported_attntion_backend: list[str] = []

    # Use first element as default backend
    supported_sampling_backend: list[str] = []

    # Use first element as default backend
    supported_lora_backend: list[str] = []

    def is_cuda(self) -> bool:
        return self._enum == PlatformEnum.CUDA

    def is_rocm(self) -> bool:
        return self._enum == PlatformEnum.ROCM

    def is_hpu(self) -> bool:
        return self._enum == PlatformEnum.HPU

    def is_xpu(self) -> bool:
        return self._enum == PlatformEnum.XPU

    def is_cpu(self) -> bool:
        return self._enum == PlatformEnum.CPU

    def is_out_of_tree(self) -> bool:
        return self._enum == PlatformEnum.OOT

    def is_cuda_alike(self) -> bool:
        """Stateless version of :func:`torch.cuda.is_available`."""
        return self._enum in (PlatformEnum.CUDA, PlatformEnum.ROCM)

    @classmethod
    def get_device_capability(
        cls,
        device_id: int = 0,
    ) -> Optional[DeviceCapability]:
        """Stateless version of :func:`torch.cuda.get_device_capability`."""
        return None

    @classmethod
    def has_device_capability(
        cls,
        capability: Union[Tuple[int, int], int],
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
    def get_device_module(cls) -> Any:
        """Get `torch.device_module` like `torch.cuda` of current platform."""
        raise NotImplementedError

    @classmethod
    def get_device_sku(cls, device_id: int = 0) -> str:
        """Get the SKU name of a device."""
        raise NotImplementedError

    @classmethod
    def get_device_uuid(cls, device_id: int = 0) -> str:
        """Get the uuid of a device, e.g. the PCI bus ID."""
        raise NotImplementedError

    @classmethod
    def get_device_core_count(cls, device_id: int = 0) -> str:
        """Get the core count of a device, e.g. SMs of CUDA, CUs of ROCM."""
        raise NotImplementedError

    @classmethod
    def get_device_count(cls) -> int:
        """Get device count on current platform"""
        raise NotImplementedError

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0, distributed=False) -> float:
        """
        Get total memory for device_type:device_id device in gigabytes.
        """
        raise NotImplementedError

    @classmethod
    def get_device_available_memory(
        cls, device_id: int = 0, distributed=False, empty_cache=True
    ) -> float:
        """
        Get available memory for device_type:device_id device in gigabytes.
        When distributed is True, the available memory is the minimum available memory of all GPUs.
        """
        raise NotImplementedError

    @classmethod
    def supports_overlap_scheduler(cls) -> bool:
        """
        Check if the current platform supports overlap scheduler
        """
        raise NotImplementedError

    @classmethod
    def seed_everything(cls, seed: Optional[int] = None) -> None:
        """
        Set the seed of each random module.
        `torch.manual_seed` will set seed on all devices.

        Loosely based on: https://github.com/Lightning-AI/pytorch-lightning/blob/2.4.0/src/lightning/fabric/utilities/seed.py#L20
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

    @classmethod
    def check_and_update_server_args(cls, server_args: ServerArgs) -> None:
        """
        Check and update the server arguments for the current platform.

        It can raise an exception if the configuration is not compatible with
        the current platform, or it can update the configuration to make it
        compatible with the current platform.

        The config is passed by reference, so it can be modified in place.
        """
        pass

    @classmethod
    def check_and_update_model_dtype(cls, model_config: ModelConfig, dtype: str) -> str:
        """
        Check and update the model's dtype for the current platform.
        """
        if cls.supported_dtype and dtype not in cls.supported_dtype:
            logger.warning(
                f"dtype {dtype} is currently not supported in "
                f"{cls.device_name}. use {cls.supported_dtype[0]} instead"
            )
            return cls.supported_dtype[0]
        return dtype

    @classmethod
    def check_and_update_attntion_backend(
        cls, model_config: ModelConfig, backend: str
    ) -> str:
        """
        Check and update the attntion backend for the current platform.
        """
        raise NotImplementedError

    @classmethod
    def check_and_update_sampling_backend(cls, backend: str) -> str:
        """
        Check and update the sampling backend for the current platform.
        """
        raise NotImplementedError

    @classmethod
    def check_and_update_lora_backend(cls, backend: str) -> str:
        """
        Check and update the lora backend for the current platform.
        """
        raise NotImplementedError

    @classmethod
    def verify_model_arch(cls, model_arch: str) -> None:
        """
        Verify whether the current platform supports the specified model
        architecture.

        - This will raise an Error or Warning based on the model support on
        the current platform.
        - By default all models are considered supported.
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
    def verify_speculative_algorithm(cls, algo: str) -> None:
        """
        Verify whether the speculative algorithm is supported by the current platform.
        """
        if (
            cls.supported_speculative_algorithm
            and algo not in cls.supported_speculative_algorithm
        ):
            raise ValueError(
                f"speculative algorithm {algo} is currently not supported in "
                f"{cls.device_name}."
            )

    @classmethod
    def get_cpu_architecture(cls) -> CpuArchEnum:
        """
        Determine the CPU architecture of the current system.
        Returns CpuArchEnum indicating the architecture type.
        """
        machine = platform.machine().lower()

        if machine in ("x86_64", "amd64", "i386", "i686"):
            return CpuArchEnum.X86
        elif machine.startswith("arm") or machine.startswith("aarch"):
            return CpuArchEnum.ARM
        elif machine.startswith("ppc"):
            return CpuArchEnum.POWERPC

        return CpuArchEnum.OTHER if machine else CpuArchEnum.UNKNOWN

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        """Checks whether pin memory is available on the current platform."""
        if in_wsl():
            # Pinning memory in WSL is not supported.
            # https://docs.nvidia.com/cuda/wsl-user-guide/index.html#known-limitations-for-linux-cuda-applications
            logger.warning(
                "Using 'pin_memory=False' as WSL is detected. "
                "This may slow down the performance."
            )
            return False
        return True

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        """
        Get device specific communicator class for distributed communication.
        """
        raise NotImplementedError

    @classmethod
    def supports_fp8(cls) -> bool:
        return False

    @classmethod
    def fp8_dtype(cls) -> torch.dtype:
        """
        Returns the preferred FP8 type on the current platform.
        """
        return torch.float8_e4m3fn

    @classmethod
    def fp8_min_max(cls) -> Tuple[float, float]:
        """
        Returns the preferred FP8 max value on the current platform.
        """
        fp8_max = torch.finfo(cls.fp8_dtype()).max
        return (-fp8_max, fp8_max)

    @classmethod
    def is_triton_avaliable(cls) -> bool:
        raise NotImplementedError

    @classmethod
    def init_environments(cls) -> None:
        """
        Init environments on current platform.

        - Init platform specific env vars.
        - Init platform specific patches.
        """
        pass


class UnspecifiedPlatform(Platform):
    _enum = PlatformEnum.UNSPECIFIED
    device_type = ""
