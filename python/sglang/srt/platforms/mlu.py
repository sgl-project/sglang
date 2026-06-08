"""Cambricon MLU device operations for the SRT platform layer."""

from __future__ import annotations

import logging
from typing import Optional

import torch

from sglang.srt.platforms.device_mixin import DeviceCapability, DeviceMixin, PlatformEnum
from sglang.srt.platforms.interface import SRTPlatform

logger = logging.getLogger(__name__)


class MluDeviceMixin(DeviceMixin):
    """MLU implementation of the shared device operations."""

    _enum: PlatformEnum = PlatformEnum.MLU
    device_name: str = "mlu"
    device_type: str = "mlu"

    def get_device_total_memory(self, device_id: int = 0) -> int:
        return int(torch.mlu.get_device_properties(device_id).total_memory)

    def get_current_memory_usage(
        self, device: Optional["torch.device"] = None
    ) -> float:
        torch.mlu.reset_peak_memory_stats(device)
        return float(torch.mlu.max_memory_allocated(device))

    def get_device(self, local_rank: int = 0) -> "torch.device":
        return torch.device("mlu", local_rank)

    def set_device(self, device: "torch.device") -> None:
        torch.mlu.set_device(device)

    def get_device_name(self, device_id: int = 0) -> str:
        return str(torch.mlu.get_device_name(device_id))

    def get_device_uuid(self, device_id: int = 0) -> str:
        props = torch.mlu.get_device_properties(device_id)
        return str(getattr(props, "uuid", f"mlu:{device_id}"))

    def get_device_capability(self, device_id: int = 0) -> Optional[DeviceCapability]:
        return None

    def empty_cache(self) -> None:
        torch.mlu.empty_cache()

    def synchronize(self) -> None:
        torch.mlu.synchronize()

    def get_available_memory(self, device_id: int = 0) -> tuple[int, int]:
        current = torch.mlu.current_device()
        try:
            torch.mlu.set_device(device_id)
            free, total = torch.mlu.mem_get_info()
            return int(free), int(total)
        finally:
            torch.mlu.set_device(current)

    def is_pin_memory_available(self, device=None) -> bool:
        if device is not None and str(device) == "cpu":
            return False
        return True

    def get_torch_distributed_backend_str(self) -> str:
        return "cncl"

    @classmethod
    def seed_everything(cls, seed: int | None = None) -> None:
        if seed is not None:
            super().seed_everything(seed)
            torch.mlu.manual_seed_all(seed)


class MluSRTPlatform(MluDeviceMixin, SRTPlatform):
    """In-tree Cambricon MLU SRT platform."""

    def apply_server_args_defaults(self, server_args) -> None:
        for name in (
            "attention_backend",
            "prefill_attention_backend",
            "decode_attention_backend",
        ):
            value = getattr(server_args, name)
            if value is None:
                setattr(server_args, name, "mlu")
            elif value != "mlu":
                raise ValueError(
                    f"MLU currently supports only the 'mlu' attention backend; "
                    f"got {name}={value!r}."
                )
        if server_args.sampling_backend is None:
            server_args.sampling_backend = "pytorch"
        elif server_args.sampling_backend != "pytorch":
            raise ValueError(
                "MLU currently supports only the 'pytorch' sampling backend; "
                f"got sampling_backend={server_args.sampling_backend!r}."
            )
        if server_args.page_size is None:
            server_args.page_size = 16

        server_args.disable_custom_all_reduce = True
        if server_args.enable_hierarchical_cache:
            logger.warning("MLU does not support hierarchical cache; disabling it.")
            server_args.enable_hierarchical_cache = False

    def get_dispatch_key_name(self) -> str:
        return "mlu"

    def get_position_dtype(self) -> torch.dtype:
        return torch.int32
