"""NPU device operations for the SRT platform layer."""

from typing import Optional

import torch

from sglang.srt.platforms.device_mixin import (
    DeviceCapability,
    DeviceMixin,
    PlatformEnum,
)
from sglang.srt.platforms.interface import PlatformCapabilities, SRTPlatform


class NPUDeviceMixin(DeviceMixin):
    """NPU implementation of the shared device operations."""

    _enum: PlatformEnum = PlatformEnum.NPU
    device_name: str = "npu"
    device_type: str = "npu"

    def get_device_total_memory(self, device_id: int = 0) -> int:
        return int(torch.npu.get_device_properties(device_id).total_memory)

    def get_current_memory_usage(
        self, device: Optional["torch.device"] = None
    ) -> float:
        return float(torch.npu.max_memory_allocated(device))

    def get_device(self, local_rank: int) -> "torch.device":
        return torch.device("npu", local_rank)

    def set_device(self, device: "torch.device") -> None:
        torch.npu.set_device(device)

    def get_device_name(self, device_id: int = 0) -> str:
        return str(torch.npu.get_device_name(device_id))

    def get_device_uuid(self, device_id: int = 0) -> str:
        return str(torch.npu.get_device_properties(device_id).uuid)

    def get_device_capability(self, device_id: int = 0) -> DeviceCapability:
        # The return value of torch_npu.npu.get_device_capability() is configured
        # via the environment variable TORCH_NPU_DEVICE_CAPABILITY, which is only
        # used for compatibility with native PyTorch and does not represent the
        # actual capabilities of the NPU hardware
        return DeviceCapability(0, 0)

    def empty_cache(self) -> None:
        torch.npu.empty_cache()

    def synchronize(self) -> None:
        torch.npu.synchronize()

    def get_available_memory(self, device_id: int = 0) -> tuple[int, int]:
        return torch.npu.mem_get_info(device_id)

    def is_pin_memory_available(self, device=None) -> bool:
        if device is not None and str(device) == "cpu":
            return False
        return True

    @classmethod
    def seed_everything(cls, seed: int | None = None) -> None:
        if seed is not None:
            super().seed_everything(seed)
            if hasattr(torch, "npu"):
                torch.npu.manual_seed_all(seed)


class NPUSRTPlatform(NPUDeviceMixin, SRTPlatform):
    """Default in-tree NPU SRT platform."""

    capabilities = PlatformCapabilities(
        supports_triton=True,
        graph_capture=True,
    )

    def get_default_attention_backend(self) -> str:
        return "ascend"

    def get_dispatch_key_name(self) -> str:
        return "npu"

    def on_destroy_model_parallel(self) -> None:
        from sglang.srt.hardware_backend.npu.moe.mega_moe import (
            destroy_mega_moe_buffers,
        )

        destroy_mega_moe_buffers()

    def request_window_nbytes(
        self,
        *,
        num_slots: int,
        layers: int,
        page_size: int,
        capacity: int,
        workspace_rows: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        draft_ring_size: int,
        draft_ring_layers: int,
    ) -> Optional[int]:
        from sglang.srt.hardware_backend.npu.dsv4.dsv4_kv_layout import (
            NPUDSV4KVLayout,
        )
        from sglang.srt.hardware_backend.npu.dsv4.npu_request_window import (
            dsv41_request_window_spec,
        )
        from sglang.srt.hardware_backend.npu.utils import is_npu_arch35

        row_layout = (
            NPUDSV4KVLayout.V41_SWA_FP8 if is_npu_arch35() else NPUDSV4KVLayout.V41_BF16
        )
        # A V4.1 draft worker still uses the V4 sparse-attention op, so its ring
        # rows keep the V4 format; budget them separately from the target window.
        draft_row_layout = (
            NPUDSV4KVLayout.V4_FP8 if is_npu_arch35() else NPUDSV4KVLayout.V4_BF16
        )
        row_dim, row_dtype = row_layout.kv_info(qk_nope_head_dim, qk_rope_head_dim)
        spec = dsv41_request_window_spec(
            num_slots=num_slots,
            layers=layers,
            page_size=page_size,
            capacity=capacity,
            workspace_rows=workspace_rows,
            row_dim=row_dim,
            row_dtype=row_dtype,
        )
        nbytes = spec.nbytes
        if draft_ring_size and draft_ring_layers:
            # Share the pool's page formula so the budget matches the
            # allocation (whole pages plus one spare).
            draft_dim, draft_dtype = draft_row_layout.kv_info(
                qk_nope_head_dim, qk_rope_head_dim
            )
            ring_rows = (num_slots * draft_ring_size + page_size + 1) // page_size
            row_bytes = draft_dim * draft_dtype.itemsize
            nbytes += ring_rows * page_size * draft_ring_layers * row_bytes
        return nbytes
