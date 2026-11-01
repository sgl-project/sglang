from __future__ import annotations

from functools import cached_property
from typing import Optional, Tuple

import torch

from sglang.kernels.arch.base import DeviceArch


class CudaArch(DeviceArch):
    @property
    def cuda(self) -> CudaArch:
        return self

    @cached_property
    def major(self) -> int:
        return self.props.major

    @cached_property
    def minor(self) -> int:
        return self.props.minor

    @property
    def capability(self) -> Tuple[int, int]:
        return self.major, self.minor

    @property
    def sm_version(self) -> int:
        """`major * 10 + minor`, e.g. 90, 100, 120."""
        return self.major * 10 + self.minor

    @property
    def max_vec_bytes(self) -> int:
        """Widest global load/store the hardware has: 256-bit on Blackwell+, else 128-bit."""
        return 32 if self.major >= 10 else 16

    def is_sm(self, major: int, minor: Optional[int] = None) -> bool:
        return self.major == major and (minor is None or self.minor == minor)

    def is_sm_at_least(self, major: int, minor: int = 0) -> bool:
        return self.capability >= (major, minor)

    @property
    def supports_pdl(self) -> bool:
        return self.major >= 9

    @property
    def supports_cluster(self) -> bool:
        return self.major >= 9
