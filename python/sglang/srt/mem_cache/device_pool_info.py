"""Device buffer facts consumed by host assembly.

The device provider enumerates persistent tensors. Host assembly owns capacity,
layer routing and transfers. This first consumer describes Mamba checkpoints.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum

import msgspec
import torch

from sglang.srt.mem_cache.hicache_storage import PoolName


class MambaStateBufferInfo(msgspec.Struct, frozen=True, kw_only=True):
    """Persistent checkpoint tensors. Layer and checkpoint axes come first.

    Extra buffers share checkpoint IDs but have no layer axis. Speculative
    intermediate tensors are work space and do not belong in host checkpoints.
    """

    temporal: torch.Tensor
    conv: tuple[torch.Tensor, ...]
    extra_checkpoint_buffers: tuple[torch.Tensor, ...] = ()

    def validate(self, *, layer_ids: tuple[int, ...] | None = None) -> None:
        shape = tuple(self.temporal.shape)
        if self.temporal.ndim < 2 or not self.conv:
            raise ValueError(
                f"Mamba temporal requires [layer, checkpoint, ...] and conv buffers, "
                f"got temporal.shape={shape}, conv count={len(self.conv)}"
            )
        if layer_ids is not None and len(layer_ids) != shape[0]:
            raise ValueError(
                f"Mamba layer_ids={layer_ids} must match temporal.shape[0]={shape[0]}"
            )
        for index, buffer in enumerate(self.conv):
            if buffer.shape[:2] != self.temporal.shape[:2]:
                raise ValueError(
                    f"Mamba conv[{index}].shape={tuple(buffer.shape)} must start "
                    f"with layer/checkpoint axes {shape[:2]}"
                )
            if buffer.dtype != self.conv[0].dtype:
                raise ValueError(
                    f"Mamba conv[{index}].dtype={buffer.dtype}, expected {self.conv[0].dtype}"
                )
        for index, buffer in enumerate(self.extra_checkpoint_buffers):
            if buffer.ndim < 1 or buffer.shape[0] != shape[1]:
                raise ValueError(
                    f"Mamba extra_checkpoint_buffers[{index}].shape={tuple(buffer.shape)} "
                    f"must start with checkpoint capacity {shape[1]}"
                )
        for buffer in (*self.conv, *self.extra_checkpoint_buffers):
            if buffer.device != self.temporal.device:
                raise ValueError(
                    f"Mamba buffer.device={buffer.device}, expected {self.temporal.device}"
                )


class KVBufferLayout(Enum):
    NHD = "nhd"
    PAGED_NHD = "paged_nhd"
    HND = "hnd"
    VECTORIZED_5D = "vectorized_5d"


class KVBuffers(msgspec.Struct, frozen=True, kw_only=True):
    """Per-layer K/V views and their existing device pointer allocations.

    NHD views may be strided. Page/head/vector axes stay in the native layout;
    this description neither flattens buffers nor enables new transfer backends.
    """

    k: tuple[torch.Tensor, ...]
    v: tuple[torch.Tensor, ...]
    # Some packed formats have no pointer tables; providers must not allocate them.
    k_data_ptrs: torch.Tensor | None
    v_data_ptrs: torch.Tensor | None
    layout: KVBufferLayout


class KVScaleLayout(Enum):
    TOKEN = "token"
    MXFP8_INTERLEAVED = "mxfp8_interleaved"


class KVScaleBuffers(msgspec.Struct, frozen=True, kw_only=True):
    k: tuple[torch.Tensor, ...]
    v: tuple[torch.Tensor, ...]
    layout: KVScaleLayout


class MHABufferInfo(msgspec.Struct, frozen=True, kw_only=True):
    """Physical MHA input for host construction; no capacity or allocation policy.

    Geometry is explicit for empty stages. head_num counts physical attention
    rows per allocator token, which can exceed the model's attention head count.
    Existing pointer tables and tensor strides must be preserved by consumers.
    """

    buffers: KVBuffers
    page_size: int
    head_num: int
    head_dim: int
    v_head_dim: int
    dtype: torch.dtype
    device: torch.device
    scales: KVScaleBuffers | None = None
    # Resolve current contiguous backing at transfer time for relocatable pools.
    transfer_buffers: Callable[[], tuple[torch.Tensor, torch.Tensor]] | None = None

    def validate(self, *, layer_ids: tuple[int, ...] | None = None) -> None:
        layers = len(self.buffers.k)
        if len(self.buffers.v) != layers or (
            layer_ids is not None and len(layer_ids) != layers
        ):
            raise ValueError(
                f"MHA layers: k={layers}, v={len(self.buffers.v)}, layer_ids={layer_ids}"
            )
        if min(self.page_size, self.head_num, self.head_dim, self.v_head_dim) <= 0:
            raise ValueError(
                f"Invalid MHA geometry: page_size={self.page_size}, "
                f"head_num={self.head_num}, head_dim={self.head_dim}, "
                f"v_head_dim={self.v_head_dim}"
            )
        for name in ("k", "v"):
            tensors = self.buffers.k if name == "k" else self.buffers.v
            for layer, tensor in enumerate(tensors):
                if tensor.dtype != self.dtype or tensor.device != self.device:
                    raise ValueError(
                        f"MHA {name}[{layer}]: dtype={tensor.dtype}, device={tensor.device}; "
                        f"expected dtype={self.dtype}, device={self.device}"
                    )
        if self.scales is not None and (
            len(self.scales.k) != layers or len(self.scales.v) != layers
        ):
            raise ValueError(
                f"MHA scales: k={len(self.scales.k)}, v={len(self.scales.v)}, "
                f"expected layers={layers}"
            )


class MLASplitBuffers(msgspec.Struct, frozen=True, kw_only=True):
    """Native NPU allocations; index slots have their own pool-local layer order."""

    k: torch.Tensor | None
    v: torch.Tensor | None
    index_k: torch.Tensor | None
    index_scales: torch.Tensor | None
    index_layer_indices: tuple[int, ...]
    k_width: int
    v_width: int
    index_width: int


class MLABufferInfo(msgspec.Struct, frozen=True, kw_only=True):
    """Persistent MLA rows or native split components, including empty stages.

    layer_indices address the pool's controller slots, not model layer IDs or
    reserved host rows. Existing pointer allocations are views, never rebuilt.
    """

    buffers: tuple[torch.Tensor, ...] | MLASplitBuffers
    data_ptrs: torch.Tensor | None
    layer_indices: tuple[int, ...]
    device_layer_num: int
    page_size: int
    kv_lora_rank: int
    qk_rope_head_dim: int
    kv_cache_dim: int
    dtype: torch.dtype
    device: torch.device
    packed_fp8: bool = False
    scales: tuple[torch.Tensor, ...] | None = None
    native_buffers: Callable[[], MLASplitBuffers] | None = None

    def validate(self, *, layer_ids: tuple[int, ...] | None = None) -> None:
        if layer_ids is not None and len(layer_ids) != len(self.layer_indices):
            raise ValueError(
                f"MLA model layers={layer_ids} differ from buffer slots={self.layer_indices}"
            )
        if len(set(self.layer_indices)) != len(self.layer_indices) or any(
            i < 0 or i >= self.device_layer_num for i in self.layer_indices
        ):
            raise ValueError(
                f"MLA slots={self.layer_indices} outside device_layer_num={self.device_layer_num}"
            )
        if isinstance(self.buffers, tuple):
            if len(self.buffers) != len(self.layer_indices):
                raise ValueError(
                    f"MLA buffers={len(self.buffers)}, slots={self.layer_indices}"
                )
            for i, tensor in zip(self.layer_indices, self.buffers):
                if (
                    tensor.ndim != 3
                    or tensor.shape[1] != 1
                    or tensor.dtype != self.dtype
                    or tensor.device != self.device
                ):
                    raise ValueError(
                        f"MLA slot={i}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
                        f"device={tensor.device}; expected [token, 1, width], "
                        f"dtype={self.dtype}, device={self.device}"
                    )
        elif self.buffers.index_k is not None and self.buffers.index_k.shape[0] != len(
            self.buffers.index_layer_indices
        ):
            raise ValueError(
                f"MLA index shape={tuple(self.buffers.index_k.shape)}, "
                f"index layers={self.buffers.index_layer_indices}"
            )


class PageRowBufferInfo(msgspec.Struct, frozen=True, kw_only=True):
    """Packed page rows: one ``row_bytes`` uint8 row per (layer, page).

    ``slab`` is ``[layers, num_pages, row_bytes]`` and ``rows[i]`` is ``slab[i]``;
    a row holds every component of that page, such as payloads and scales.
    """

    slab: torch.Tensor
    rows: tuple[torch.Tensor, ...]
    row_bytes: int
    page_size: int
    num_pages: int

    def validate(self, *, layer_ids: tuple[int, ...] | None = None) -> None:
        layers = len(self.rows)
        expected = (layers, self.num_pages, self.row_bytes)
        if (
            tuple(self.slab.shape) != expected
            or self.slab.dtype != torch.uint8
            or not self.slab.is_contiguous()
            or (layer_ids is not None and len(layer_ids) != layers)
        ):
            raise ValueError(
                f"Page rows: slab shape={tuple(self.slab.shape)}, "
                f"dtype={self.slab.dtype}, layer_ids={layer_ids}; "
                f"expected contiguous uint8 {expected}"
            )
        for layer, row in enumerate(self.rows):
            if row.data_ptr() != self.slab[layer].data_ptr():
                raise ValueError(f"Page rows: rows[{layer}] is not slab[{layer}]")


class DevicePoolInfo(msgspec.Struct, frozen=True, kw_only=True):
    pool_name: PoolName
    indices_from_pool: PoolName
    # Model layer IDs, including the PP stage offset, in buffer-axis order.
    layer_ids: tuple[int, ...]
    buffer_info: (
        MambaStateBufferInfo | MHABufferInfo | MLABufferInfo | PageRowBufferInfo
    )

    def __post_init__(self) -> None:
        # A PP stage may own no checkpoint layers. Buffer validation decides
        # whether an empty layer axis is valid for the physical format.
        if len(set(self.layer_ids)) != len(self.layer_ids) or any(
            layer < 0 for layer in self.layer_ids
        ):
            raise ValueError(f"{self.pool_name}: invalid model layers {self.layer_ids}")
