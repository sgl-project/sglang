"""Apple MPS support for SRT's in-tree platform layer."""

from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.platforms.device_mixin import (
    DeviceMixin,
    PlatformEnum,
)
from sglang.srt.platforms.interface import SRTPlatform


def _recommended_working_set_size() -> int:
    getter = getattr(torch.mps, "recommended_max_memory", None)
    if not callable(getter):
        raise RuntimeError(
            "The SGLang MPS backend requires "
            "torch.mps.recommended_max_memory() from Torch 2.13.x"
        )
    total = int(getter())
    if total <= 0:
        raise RuntimeError(
            "torch.mps.recommended_max_memory() returned a non-positive Metal "
            "working-set limit; refusing to size the model and KV pools from "
            "raw system RAM"
        )
    return total


class MpsDeviceMixin(DeviceMixin):
    """Device operations backed by ``torch.mps``."""

    _enum: PlatformEnum = PlatformEnum.MPS
    device_name: str = "mps"
    device_type: str = "mps"

    def get_device_total_memory(self, device_id: int = 0) -> int:
        # Metal's working-set limit, not total system RAM, bounds model memory.
        return _recommended_working_set_size()

    def get_current_memory_usage(self, device: Optional[torch.device] = None) -> float:
        # Driver allocation includes resident cached MPS buffers.
        return float(torch.mps.driver_allocated_memory())

    def get_device(self, local_rank: int = 0) -> torch.device:
        if local_rank not in (-1, 0):
            raise ValueError(f"MPS exposes one device; got local_rank={local_rank}")
        return torch.device("mps")

    def set_device(self, device: torch.device) -> None:
        if str(device).split(":", 1)[0] != "mps":
            raise ValueError(f"MPS platform cannot select device {device}")
        # MPS has no device-selection API.

    def get_device_name(self, device_id: int = 0) -> str:
        return str(torch.backends.mps.get_name())

    def get_device_uuid(self, device_id: int = 0) -> str:
        # PyTorch exposes no stable MPS UUID.
        return "mps:0"

    def get_device_capability(self, device_id: int = 0):
        return None

    def empty_cache(self) -> None:
        empty_cache = getattr(torch.mps, "empty_cache", None)
        if callable(empty_cache):
            empty_cache()

    def synchronize(self) -> None:
        synchronize = getattr(torch.mps, "synchronize", None)
        if callable(synchronize):
            synchronize()


class MpsSRTPlatform(MpsDeviceMixin, SRTPlatform):
    """Built-in SRT platform for Apple Silicon MPS."""

    def get_default_attention_backend(self) -> str:
        return "torch_native"
