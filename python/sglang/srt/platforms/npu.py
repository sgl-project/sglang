"""NPU (Huawei Ascend) device operations for the SRT platform layer."""

import logging
from typing import Optional

import torch

from sglang.srt.platforms.device_mixin import (
    DeviceCapability,
    DeviceMixin,
    PlatformEnum,
)
from sglang.srt.platforms.interface import SRTPlatform

logger = logging.getLogger(__name__)


class NpuDeviceMixin(DeviceMixin):
    """NPU implementation of the shared device operations.

    Requires ``torch_npu`` to be importable so that ``torch.npu`` is populated;
    the plugin is imported lazily in ``_ensure_torch_npu()``.
    """

    _enum: PlatformEnum = PlatformEnum.NPU
    device_name: str = "npu"
    device_type: str = "npu"

    @staticmethod
    def _ensure_torch_npu() -> None:
        # torch_npu registers the ``npu`` device with the torch dispatcher on
        # import. It's not a hard dependency of sglang, so we import here on
        # demand and give a friendly error if it's missing.
        if not hasattr(torch, "npu"):
            import torch_npu  # noqa: F401  # pylint: disable=import-outside-toplevel

    def get_device_total_memory(self, device_id: int = 0) -> int:
        self._ensure_torch_npu()
        return int(torch.npu.get_device_properties(device_id).total_memory)

    def get_current_memory_usage(
        self, device: Optional["torch.device"] = None
    ) -> float:
        self._ensure_torch_npu()
        return float(torch.npu.max_memory_allocated(device))

    def get_device(self, local_rank: int) -> "torch.device":
        return torch.device("npu", local_rank)

    def set_device(self, device: "torch.device") -> None:
        self._ensure_torch_npu()
        torch.npu.set_device(device)

    def get_device_name(self, device_id: int = 0) -> str:
        self._ensure_torch_npu()
        return str(torch.npu.get_device_name(device_id))

    def get_device_uuid(self, device_id: int = 0) -> str:
        # torch_npu's DeviceProperties does not expose a UUID; the closest
        # stable identifier is the device name plus the OS-assigned index.
        self._ensure_torch_npu()
        return f"{torch.npu.get_device_name(device_id)}:{device_id}"

    def get_device_capability(self, device_id: int = 0) -> DeviceCapability:
        # Ascend NPUs don't expose a (major, minor) capability tuple the way
        # NVIDIA / AMD do. Return a fixed (0, 0) so callers get a well-formed
        # value; consumers that need finer-grained gating should key off
        # ``torch.npu.get_device_name(device_id)`` (e.g. "Ascend910B2").
        return DeviceCapability(0, 0)

    def empty_cache(self) -> None:
        self._ensure_torch_npu()
        torch.npu.empty_cache()

    def synchronize(self) -> None:
        self._ensure_torch_npu()
        torch.npu.synchronize()

    def get_available_memory(self, device_id: int = 0) -> tuple[int, int]:
        """Return (free, total) device memory in bytes."""
        self._ensure_torch_npu()
        if not (hasattr(torch, "npu") and torch.npu.is_available()):
            return 0, 0
        return torch.npu.mem_get_info(device_id)

    def is_pin_memory_available(self, device=None) -> bool:
        if device is not None and str(device) == "cpu":
            return False
        return True

    # ``get_torch_distributed_backend_str`` intentionally not overridden.
    # ``_DEVICE_TO_DISTRIBUTED_BACKEND`` in device_mixin.py already maps
    # "npu" -> "hccl" (or "zbal" when SGLANG_ZBAL_LOCAL_MEM_SIZE > 0), and
    # the base ``DeviceMixin`` implementation looks up ``self.device_type``
    # in that table -- so plain inheritance gives the right answer.

    @classmethod
    def seed_everything(cls, seed: int | None = None) -> None:
        if seed is not None:
            super().seed_everything(seed)
            if hasattr(torch, "npu"):
                torch.npu.manual_seed_all(seed)


class NpuSRTPlatform(NpuDeviceMixin, SRTPlatform):
    """Default in-tree NPU SRT platform (Huawei Ascend 910B / A2)."""

    # In-tree NPU dispatch already flows through ``if _is_npu:`` guards in
    # ``kv_cache_configurator``, ``cuda_graph_setup`` and elsewhere -- see
    # ``hardware_backend/npu/`` for the concrete backend classes. Overriding
    # the ``get_*_cls`` factory methods here is not required for in-tree
    # dispatch (those factories are only consulted for out-of-tree platforms
    # via ``current_platform.is_out_of_tree()``); leaving them at the
    # ``NotImplementedError`` defaults matches ``CudaSRTPlatform``.

    def get_default_attention_backend(self) -> str:
        return "ascend"

    def get_dispatch_key_name(self) -> str:
        return "npu"

    def supports_fp8(self) -> bool:
        # Ascend 910B/A2 supports FP8 via the CANN toolkit + torch_npu ops,
        # matching the existing NPU quantization backends under
        # ``hardware_backend/npu/quantization/``.
        return True

    def support_cuda_graph(self) -> bool:
        # ``NPUGraphRunner`` in ``hardware_backend/npu/graph_runner/`` uses
        # torch_npu graph capture for the decode path.
        return True

    def support_piecewise_cuda_graph(self) -> bool:
        # Piecewise / prefill-side graph capture is not yet wired for NPU.
        return False
