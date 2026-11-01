from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from sglang.kernels.arch._cuda import CudaArch
    from sglang.kernels.arch._hip import HipArch


class DeviceArch:
    """Facts every `torch.cuda`-style device reports; each platform subclasses it.

    Read from the current device on first access: all GPUs of a node are assumed
    to be the same model.
    """

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"

    @cached_property
    def props(self):
        return torch.cuda.get_device_properties()

    @cached_property
    def name(self) -> str:
        return self.props.name

    @cached_property
    def num_sms(self) -> int:
        """SMs on NVIDIA, CUs on AMD."""
        return self.props.multi_processor_count

    @cached_property
    def warp_size(self) -> int:
        return self.props.warp_size

    @cached_property
    def l2_cache_bytes(self) -> int:
        return self.props.L2_cache_size

    @cached_property
    def smem_per_block_optin(self) -> int:
        """Largest dynamic shared memory a block can opt into."""
        return self.props.shared_memory_per_block_optin

    @cached_property
    def smem_per_sm(self) -> int:
        return self.props.shared_memory_per_multiprocessor

    @cached_property
    def regs_per_sm(self) -> int:
        return self.props.regs_per_multiprocessor

    @cached_property
    def max_threads_per_sm(self) -> int:
        return self.props.max_threads_per_multi_processor

    @property
    def cuda(self) -> Optional[CudaArch]:
        return None

    @property
    def hip(self) -> Optional[HipArch]:
        return None

    @property
    def supports_pdl(self) -> bool:
        return False

    def is_sm(self, major: int, minor: Optional[int] = None) -> bool:
        """Whether this is an NVIDIA GPU of SM `major` (any minor) or SM `major.minor`."""
        return False

    def is_sm_at_least(self, major: int, minor: int = 0) -> bool:
        """Whether this is an NVIDIA GPU with compute capability >= (major, minor)."""
        return False
