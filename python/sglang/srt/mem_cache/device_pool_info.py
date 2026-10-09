"""Device buffer facts consumed by host assembly and direct linker.

PagedLayerBufferInfo defines the per-layer paging contract. Concrete descriptors
own format validation and zero-copy views. Consumers own allocation, transfer
scheduling and the decision to pack drafts. Checkpoint buffers without token
pages are outside this contract.
"""

from __future__ import annotations

from enum import Enum
from typing import Protocol

import msgspec
import torch
from typing_extensions import Self

from sglang.srt.mem_cache.hicache_storage import PoolName


class PagedLayerBufferInfo(Protocol):
    """Common capability for buffers with one tensor per owning model layer.

    Binding validates descriptors before any page access. Frozen implementations
    retain tensor references without owning their allocation or copying data.
    """

    @property
    def buffers(self) -> tuple[torch.Tensor, ...]:
        """Physical tensors in the same order as DevicePoolInfo.layer_ids."""
        ...

    @property
    def page_size(self) -> int:
        """Original tokens covered by one page, before index compression."""
        ...

    @property
    def page_bytes(self) -> int:
        """Bytes per page of one layer, used for host sizing after validation."""
        ...

    def validate(self, *, layer_ids: tuple[int, ...] | None = None) -> None:
        """Binding/adapter entry check for tensor format and optional layer count.

        Raise ValueError with the failing field and actual value. Layer IDs are
        model layer numbers, including PP offsets, in physical-buffer order.
        """
        ...

    def page_views(self) -> tuple[torch.Tensor, ...]:
        """Linker transfer views: contiguous uint8 [page, page_bytes], zero-copy.

        Call after validate(). Return one view per owning layer. Exclude any
        trailing rows that do not form a complete page.
        """
        ...

    def packed_with(self, drafts: tuple[Self, ...]) -> Self:
        """Host packed-draft assembly of validated, same-format descriptors.

        Preserve target-then-draft order and tensor identity. Return a validated
        descriptor of the same type or raise ValueError for incompatible formats.
        Linker handles heterogeneous buffers separately and does not call this.
        """
        ...


class IndexKeyFormat(Enum):
    # Each key has 128 fp8 elements. A page stores keys followed by fp32 scales.
    DSA_FP8 = "dsa_fp8"

    def page_bytes(self, key_count: int) -> int:
        # Each physical format owns its sizing, including scales and padding.
        if self is IndexKeyFormat.DSA_FP8:
            key_data_bytes = key_count * 128
            scale_bytes = key_count * 4
            return key_data_bytes + scale_bytes
        raise ValueError(f"unsupported index key format {self}")


class MLABufferInfo(msgspec.Struct, frozen=True, kw_only=True):
    # Coverage in original tokens, independent of any compressed row count.
    page_size: int
    buffers: tuple[torch.Tensor, ...]

    @property
    def page_bytes(self) -> int:
        buffer = self.buffers[0]
        return self.page_size * buffer.shape[-1] * buffer.element_size()

    def validate(self, *, layer_ids: tuple[int, ...] | None = None) -> None:
        if layer_ids is not None and len(layer_ids) != len(self.buffers):
            raise ValueError(
                f"{len(layer_ids)} model layers must match {len(self.buffers)} buffers"
            )
        if self.page_size <= 0:
            raise ValueError(f"MLA page_size must be positive, got {self.page_size}")
        if not self.buffers:
            raise ValueError("MLA buffers must contain at least one tensor")
        first = self.buffers[0]
        for layer, buffer in enumerate(self.buffers):
            if (
                buffer.ndim != 3
                or buffer.shape[0] <= 0
                or buffer.shape[1] != 1
                or buffer.shape[2] < 1
            ):
                raise ValueError(
                    f"MLA buffer[{layer}]: expected [token > 0, 1, width > 0], "
                    f"got shape={tuple(buffer.shape)}"
                )
            if not buffer.is_contiguous():
                raise ValueError(
                    f"MLA buffer[{layer}]: expected contiguous tensor, "
                    f"got stride={buffer.stride()}"
                )
            if buffer.shape[1:] != first.shape[1:]:
                raise ValueError(
                    f"MLA buffer[{layer}]: expected row shape={tuple(first.shape[1:])}, "
                    f"got {tuple(buffer.shape[1:])}"
                )
            if buffer.dtype != first.dtype:
                raise ValueError(
                    f"MLA buffer[{layer}]: expected dtype={first.dtype}, got {buffer.dtype}"
                )
            if buffer.device != first.device:
                raise ValueError(
                    f"MLA buffer[{layer}]: expected device={first.device}, got {buffer.device}"
                )

    def page_views(self) -> tuple[torch.Tensor, ...]:
        """Zero-copy byte views of complete pages, after validate()."""
        pages = []
        for buffer in self.buffers:
            page_count = buffer.shape[0] // self.page_size
            if page_count == 0:
                raise ValueError("MLA buffer must contain at least one complete page")
            pages.append(
                buffer[: page_count * self.page_size]
                .view(torch.uint8)
                .reshape(page_count, -1)
            )
        return tuple(pages)

    def packed_with(self, drafts: tuple[MLABufferInfo, ...]) -> MLABufferInfo:
        if any(draft.page_size != self.page_size for draft in drafts):
            raise ValueError("packed MLA page coverage differs")
        buffers = self.buffers + tuple(
            buffer for draft in drafts for buffer in draft.buffers
        )
        packed = MLABufferInfo(page_size=self.page_size, buffers=buffers)
        packed.validate()
        return packed


class IndexKeyBufferInfo(msgspec.Struct, frozen=True, kw_only=True):
    page_size: int
    buffers: tuple[torch.Tensor, ...]
    compress_ratio: int
    format: IndexKeyFormat

    @property
    def page_bytes(self) -> int:
        return self.format.page_bytes(self.page_size // self.compress_ratio)

    def validate(self, *, layer_ids: tuple[int, ...] | None = None) -> None:
        buffers = self.buffers
        if layer_ids is not None and len(layer_ids) != len(buffers):
            raise ValueError(
                f"{len(layer_ids)} model layers must match {len(buffers)} buffers"
            )
        if (
            self.page_size <= 0
            or self.compress_ratio <= 0
            or self.page_size % self.compress_ratio
        ):
            raise ValueError(
                f"page coverage {self.page_size} must be divisible by "
                f"compression ratio {self.compress_ratio}"
            )
        if not isinstance(self.format, IndexKeyFormat):
            raise ValueError(f"unsupported index key format {self.format}")
        if not buffers:
            raise ValueError("index key input must contain at least one buffer")
        page_bytes = self.page_bytes
        for layer, buffer in enumerate(buffers):
            if (
                buffer.dtype != torch.uint8
                or buffer.ndim != 2
                or buffer.shape[0] == 0
                or buffer.shape[1] != page_bytes
                or not buffer.is_contiguous()
                or buffer.device != buffers[0].device
            ):
                raise ValueError(
                    f"index buffer[{layer}]: expected contiguous uint8 index pages of {page_bytes} bytes "
                    f"on {buffers[0].device}, got {tuple(buffer.shape)} "
                    f"{buffer.dtype} on {buffer.device}, stride={buffer.stride()}"
                )

    def page_views(self) -> tuple[torch.Tensor, ...]:
        return self.buffers

    def packed_with(self, drafts: tuple[IndexKeyBufferInfo, ...]) -> IndexKeyBufferInfo:
        if any(
            draft.page_size != self.page_size
            or draft.compress_ratio != self.compress_ratio
            or draft.format is not self.format
            for draft in drafts
        ):
            raise ValueError("packed index page format differs")
        buffers = self.buffers + tuple(
            buffer for draft in drafts for buffer in draft.buffers
        )
        packed = IndexKeyBufferInfo(
            page_size=self.page_size,
            buffers=buffers,
            compress_ratio=self.compress_ratio,
            format=self.format,
        )
        packed.validate()
        return packed


DeviceBufferInfo = MLABufferInfo | IndexKeyBufferInfo


class DevicePoolInfo(msgspec.Struct, frozen=True, kw_only=True):
    pool_name: PoolName
    indices_from_pool: PoolName
    # Model layer IDs, including the PP stage offset, in buffer-axis order.
    layer_ids: tuple[int, ...]
    buffer_info: DeviceBufferInfo
    # (reader model layer ID, owner model layer ID), including PP offsets.
    # Readers reuse owner buffers and are excluded from layer_ids.
    # Current transfer binding rejects nonempty mappings.
    shared_layer_to_owner: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        if (
            not self.layer_ids
            or len(set(self.layer_ids)) != len(self.layer_ids)
            or min(self.layer_ids) < 0
        ):
            raise ValueError(f"{self.pool_name}: invalid model layers {self.layer_ids}")
        readers = set()
        for reader, owner in self.shared_layer_to_owner:
            if (
                reader < 0
                or reader in readers
                or reader in self.layer_ids
                or owner not in self.layer_ids
            ):
                raise ValueError(
                    f"{self.pool_name}: invalid shared layer {reader} -> {owner}"
                )
            readers.add(reader)
