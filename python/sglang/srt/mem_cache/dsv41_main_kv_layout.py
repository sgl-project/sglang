from __future__ import annotations

import msgspec
import torch

from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout


DSV41_MAIN_KV_LAYOUT = KVLayout.DSV41_MAIN_KV_E2M1_BLOCK16_ROPE_BF16_V1
DSV41_MAIN_KV_LAYOUT_VERSION = 1
DSV41_MAIN_KV_PAYLOAD_BYTES = 224
DSV41_MAIN_KV_SCALE_BYTES = 32
DSV41_MAIN_KV_ROPE_BYTES = 128
DSV41_MAIN_KV_BYTES_PER_SLOT = (
    DSV41_MAIN_KV_PAYLOAD_BYTES + DSV41_MAIN_KV_SCALE_BYTES + DSV41_MAIN_KV_ROPE_BYTES
)
DSV41_MAIN_KV_PAGE_SLOTS = (128, 256)


class MainKVLayoutSpec(msgspec.Struct, frozen=True):
    layout_id: KVLayout
    version: int
    page_slots: int
    payload_bytes_per_slot: int
    scale_bytes_per_slot: int
    rope_bytes_per_slot: int
    bytes_per_slot: int
    payload_offset: int
    scale_offset: int
    rope_offset: int
    page_bytes: int


def make_dsv41_packed_main_kv_spec(page_slots: int) -> MainKVLayoutSpec:
    if page_slots not in DSV41_MAIN_KV_PAGE_SLOTS:
        raise ValueError(
            "DSV4.1 packed Main KV supports 128 or 256 slots per page, "
            f"got {page_slots}"
        )

    scale_offset = DSV41_MAIN_KV_PAYLOAD_BYTES * page_slots
    rope_offset = (DSV41_MAIN_KV_PAYLOAD_BYTES + DSV41_MAIN_KV_SCALE_BYTES) * page_slots
    page_bytes = DSV41_MAIN_KV_BYTES_PER_SLOT * page_slots
    return MainKVLayoutSpec(
        layout_id=DSV41_MAIN_KV_LAYOUT,
        version=DSV41_MAIN_KV_LAYOUT_VERSION,
        page_slots=page_slots,
        payload_bytes_per_slot=DSV41_MAIN_KV_PAYLOAD_BYTES,
        scale_bytes_per_slot=DSV41_MAIN_KV_SCALE_BYTES,
        rope_bytes_per_slot=DSV41_MAIN_KV_ROPE_BYTES,
        bytes_per_slot=DSV41_MAIN_KV_BYTES_PER_SLOT,
        payload_offset=0,
        scale_offset=scale_offset,
        rope_offset=rope_offset,
        page_bytes=page_bytes,
    )


def validate_dsv41_packed_main_kv_spec(spec: MainKVLayoutSpec) -> None:
    expected = make_dsv41_packed_main_kv_spec(spec.page_slots)
    if spec != expected:
        raise ValueError(
            "packed Main KV spec does not match the canonical "
            f"{DSV41_MAIN_KV_LAYOUT.value} geometry"
        )


def resolve_dsv41_main_kv_layout_specs(
    option: str, full_page_size: int
) -> dict[int, MainKVLayoutSpec] | None:
    option = option.lower()
    if option in ("auto", "flashmla_fp8"):
        return None
    if option != "packed_fp4":
        raise ValueError(f"unknown DSV4.1 Main-KV layout {option!r}")
    return {
        1: make_dsv41_packed_main_kv_spec(full_page_size),
        2: make_dsv41_packed_main_kv_spec(full_page_size // 2),
    }


class PackedMainKVView(msgspec.Struct, frozen=True):
    storage: torch.Tensor
    spec: MainKVLayoutSpec

    def __post_init__(self) -> None:
        validate_dsv41_packed_main_kv_spec(self.spec)
        if self.storage.dtype is not torch.uint8:
            raise ValueError(
                f"packed Main KV storage must use torch.uint8, got {self.storage.dtype}"
            )
        if self.storage.ndim != 2:
            raise ValueError(
                "packed Main KV storage must be 2D [num_pages, page_bytes], "
                f"got shape {tuple(self.storage.shape)}"
            )
        if not self.storage.is_contiguous():
            raise ValueError("packed Main KV storage must be contiguous")
        if self.storage.shape[1] != self.spec.page_bytes:
            raise ValueError(
                "packed Main KV page width does not match its spec: "
                f"{self.storage.shape[1]} != {self.spec.page_bytes}"
            )

    def _byte_region(self, *, offset: int, bytes_per_slot: int) -> torch.Tensor:
        return self.storage.as_strided(
            size=(self.storage.shape[0], self.spec.page_slots, bytes_per_slot),
            stride=(self.storage.stride(0), bytes_per_slot, 1),
            storage_offset=self.storage.storage_offset() + offset,
        )

    @property
    def payload(self) -> torch.Tensor:
        return self._byte_region(
            offset=self.spec.payload_offset,
            bytes_per_slot=self.spec.payload_bytes_per_slot,
        )

    @property
    def scales(self) -> torch.Tensor:
        return self._byte_region(
            offset=self.spec.scale_offset,
            bytes_per_slot=self.spec.scale_bytes_per_slot,
        )

    @property
    def rope(self) -> torch.Tensor:
        return self._byte_region(
            offset=self.spec.rope_offset,
            bytes_per_slot=self.spec.rope_bytes_per_slot,
        ).view(torch.bfloat16)
