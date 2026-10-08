from __future__ import annotations

from enum import Enum

import msgspec
import torch

from sglang.srt.mem_cache.hicache_storage import PoolName


class IndexPageEncoding(Enum):
    # A page stores 128 fp8 keys per row followed by one fp32 scale per row.
    DSA_FP8 = "dsa_fp8"


class EncodedPageBuffers(msgspec.Struct, frozen=True, kw_only=True):
    buffers: tuple[torch.Tensor, ...]
    encoding: IndexPageEncoding


class MLABufferInfo(msgspec.Struct, frozen=True, kw_only=True):
    # Coverage in original tokens, independent of any compressed row count.
    page_size: int
    buffers: tuple[torch.Tensor, ...]
    compress_ratio: int = 1

    @property
    def layer_count(self) -> int:
        return len(self.buffers)

    def validate(self) -> None:
        if self.page_size <= 0 or self.compress_ratio != 1 or not self.buffers:
            raise ValueError("MLA host transfer requires uncompressed token rows")
        first = self.buffers[0]
        if first.ndim != 3 or first.shape[1] != 1 or first.shape[2] < 1:
            raise ValueError("MLA buffers require [token, 1, width] rows")
        for buffer in self.buffers:
            if (
                buffer.ndim != 3
                or buffer.shape[0] <= 0
                or buffer.shape[1:] != first.shape[1:]
                or buffer.dtype != first.dtype
                or buffer.device != first.device
                or not buffer.is_contiguous()
            ):
                raise ValueError("MLA layers must have matching packed rows")


class IndexKeyBufferInfo(msgspec.Struct, frozen=True, kw_only=True):
    page_size: int
    buffers: EncodedPageBuffers
    compress_ratio: int

    @property
    def page_bytes(self) -> int:
        return self.page_size // self.compress_ratio * (128 + 4)

    def validate(self) -> None:
        if (
            self.page_size <= 0
            or self.compress_ratio <= 0
            or self.page_size % self.compress_ratio
        ):
            raise ValueError(
                f"page coverage {self.page_size} must be divisible by "
                f"compression ratio {self.compress_ratio}"
            )
        if self.buffers.encoding is not IndexPageEncoding.DSA_FP8:
            raise ValueError(f"unsupported index page encoding {self.buffers.encoding}")
        buffers = self.buffers.buffers
        if not buffers:
            raise ValueError("index key input must contain at least one buffer")
        page_bytes = self.page_bytes
        for buffer in buffers:
            if (
                buffer.dtype != torch.uint8
                or buffer.ndim != 2
                or buffer.shape[0] == 0
                or buffer.shape[1] != page_bytes
                or not buffer.is_contiguous()
                or buffer.device != buffers[0].device
            ):
                raise ValueError(
                    f"expected contiguous uint8 index pages of {page_bytes} bytes "
                    f"on {buffers[0].device}, got {tuple(buffer.shape)} "
                    f"{buffer.dtype} on {buffer.device}"
                )


DeviceBufferInfo = MLABufferInfo | IndexKeyBufferInfo


class DevicePoolInfo(msgspec.Struct, frozen=True, kw_only=True):
    pool_name: PoolName
    indices_from_pool: PoolName
    # Model layer IDs, including the PP stage offset, in buffer-axis order.
    layer_ids: tuple[int, ...]
    buffer_info: DeviceBufferInfo
    shared_layer_to_owner: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.buffer_info, IndexKeyBufferInfo):
            buffers = self.buffer_info.buffers.buffers
        else:
            buffers = self.buffer_info.buffers
        if not self.layer_ids or len(self.layer_ids) != len(buffers):
            raise ValueError(
                f"{self.pool_name}: {len(self.layer_ids)} model layers must match "
                f"{len(buffers)} buffers"
            )
        if len(set(self.layer_ids)) != len(self.layer_ids) or min(self.layer_ids) < 0:
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
