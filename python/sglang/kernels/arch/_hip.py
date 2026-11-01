from __future__ import annotations

from functools import cached_property

from sglang.kernels.arch.base import DeviceArch


class HipArch(DeviceArch):
    @property
    def hip(self) -> HipArch:
        return self

    @cached_property
    def gcn_arch(self) -> str:
        """Base gfx target, e.g. "gfx942" (feature suffixes such as ":sramecc+" stripped)."""
        return self.props.gcnArchName.split(":")[0]
