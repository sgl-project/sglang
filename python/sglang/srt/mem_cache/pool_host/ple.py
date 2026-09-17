"""Host mirror for the per-request PLE side states of a Qwen4-Exp MambaPool.

The short-conv window and the N-gram context are addressed by the request's
MambaPool slot, not by a KV token, so this pool is a MAMBA-derived sidecar: it
never allocates, and every transfer arrives with the MAMBA pool's host and
device indices.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any

import msgspec
import numpy as np
import torch

from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.srt.mem_cache.ple_state_pool import PLE_NGRAM_STATE_LAYER_ID
from sglang.srt.mem_cache.pool_host.base import HostKVCache, host_memory_budget_bytes
from sglang.srt.mem_cache.pool_host.common import (
    ALLOC_MEMORY_FUNCS,
    get_allocator_from_storage,
)
from sglang.srt.mem_cache.pool_host.state_transfer import (
    copy_state_slots_all_layers_lf_pf,
    copy_state_slots_pf_lf,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.pool_host.mamba import MambaPoolHost

logger = logging.getLogger(__name__)

_PAGE_FIRST_LAYOUTS = ("page_first", "page_first_direct")


class PleStateRegion(msgspec.Struct, frozen=True):
    """One group of same-shaped, same-dtype slot-indexed state tensors.

    `device_tensors` are `[slots, *state_shape]` views; the slot axis is
    normalized to dim 0 by `SlotIndexedState.iter_transfer_state_entries`.
    """

    field: str
    dtype: Any
    state_shape: tuple[int, ...]
    device_tensors: list[Any]
    layer_ids: list[int]

    @property
    def elem_count(self) -> int:
        return int(np.prod(self.state_shape)) if self.state_shape else 1

    @property
    def slot_bytes(self) -> int:
        return self.elem_count * self.dtype.itemsize


def collect_ple_state_regions(device_pool: MambaPool) -> list[PleStateRegion]:
    """Group a MambaPool's slot-sibling state tensors into transfer regions.

    The `SlotIndexedState` protocol excludes the `intermediate_*` spec-verify
    scratch, which is per-draft-token and must not be cached.
    """
    RegionKey = tuple[str, torch.dtype, tuple[int, ...]]
    grouped: dict[RegionKey, tuple[list[Any], list[int]]] = {}
    for sibling in device_pool._slot_siblings:
        for entry in sibling.iter_transfer_state_entries():
            field, tensor, _slice_axis, layer_id = entry
            if tensor.numel() == 0:
                continue
            key = (field, tensor.dtype, tuple(tensor.shape[1:]))
            tensors, layer_ids = grouped.setdefault(key, ([], []))
            tensors.append(tensor)
            layer_ids.append(layer_id)
    return [
        PleStateRegion(
            field=field,
            dtype=dtype,
            state_shape=state_shape,
            device_tensors=tensors,
            layer_ids=layer_ids,
        )
        for (field, dtype, state_shape), (tensors, layer_ids) in grouped.items()
    ]


class PleStatePoolHost(HostKVCache):
    """Host buffers for a MambaPool's PLE side states; slots mirror MAMBA's."""

    def __init__(
        self,
        device_pool: MambaPool,
        anchor_host: MambaPoolHost,
        layout: str,
        pin_memory: bool = True,
        device: str = "cpu",
        allocator_type: str = "default",
    ):
        if layout not in _PAGE_FIRST_LAYOUTS:
            raise ValueError(
                f"PleStatePoolHost requires a page-first layout, got {layout!r}."
            )
        self.regions = collect_ple_state_regions(device_pool)
        if not self.regions:
            raise ValueError(
                "PleStatePoolHost was built for a MambaPool with no slot-sibling state."
            )

        self.device_pool = device_pool
        self.pool_label = "ple"
        self.page_size = 1
        self.layout = layout
        self.pin_memory = pin_memory
        self.device = device
        self.allocator = get_allocator_from_storage(allocator_type)
        self.gpu_device = device_pool.device

        # Transfers arrive with MAMBA's host indices, so the slot count has to
        # be the anchor's.
        self.size = anchor_host.size
        self.page_num = self.size // self.page_size + 1
        self.start_layer = 0
        self.end_layer = self.layer_num

        self.dtype = self.regions[0].dtype
        self.size_per_token = self.get_size_per_token()

        requested_bytes = self.size * self.size_per_token
        available_bytes = host_memory_budget_bytes()
        if requested_bytes > available_bytes:
            raise ValueError(
                f"Not enough host memory for the PLE state host pool. Requesting "
                f"{requested_bytes / 1e9:.2f} GB but only have "
                f"{available_bytes / 1e9:.2f} GB free. Please reduce the size of "
                f"the hierarchical cache."
            )
        logger.info(
            "Allocating %.2f GB host memory for the PLE state pool "
            "(slots=%d, regions=%s, layout=%s).",
            requested_bytes / 1e9,
            self.size,
            [(r.field, len(r.device_tensors)) for r in self.regions],
            self.layout,
        )

        self.device_ptrs = [
            torch.tensor(
                [tensor.data_ptr() for tensor in region.device_tensors],
                dtype=torch.uint64,
                device=self.gpu_device,
            )
            for region in self.regions
        ]
        self.kv_buffer = self.init_kv_buffer()
        # Must be True: HostPoolGroup computes can_use_write_back_jit as AND of
        # all pools. This pool's own backup path routes by layout + io_backend.
        self.can_use_write_back_jit = True
        self.can_use_jit = False
        self.lock = threading.RLock()
        self.clear()

    # One transfer unit for the whole pool: the per-layer load loop maps a
    # global layer to a single local index, and the request-wide N-gram context
    # belongs to no model layer.
    layer_num = 1
    target_layer_num = 1

    @property
    def model_layer_ids(self) -> list[int]:
        """Model layers whose state this pool carries, N-gram sentinel aside."""
        return sorted(
            layer_id
            for region in self.regions
            for layer_id in region.layer_ids
            if layer_id != PLE_NGRAM_STATE_LAYER_ID
        )

    def get_size_per_token(self) -> int:
        return sum(
            region.slot_bytes * len(region.device_tensors) for region in self.regions
        )

    def get_ksize_per_token(self) -> int:
        return self.get_size_per_token()

    def init_kv_buffer(self):
        alloc_func = ALLOC_MEMORY_FUNCS[
            self.gpu_device.type
            if isinstance(self.gpu_device, torch.device)
            else str(self.gpu_device)
        ]
        self.region_buffers = []
        for region in self.regions:
            entry_count = len(region.device_tensors)
            # Per-slot row stride is slot_bytes * entry_count, the
            # src_layout_dim/dst_layout_dim the transfer helpers assume.
            dims = (self.size, entry_count)
            if self.layout == "page_first_direct":
                dims += (1,)
            dims += region.state_shape
            self.region_buffers.append(
                alloc_func(
                    dims,
                    dtype=region.dtype,
                    device=self.device,
                    pin_memory=self.pin_memory,
                    allocator=self.allocator,
                    registration_granularity_bytes=region.slot_bytes * entry_count,
                )
            )
        return list(self.region_buffers)

    def get_hybrid_pool_buffer(self):
        return list(self.region_buffers)

    def load_to_device_per_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend="kernel",
        *,
        is_draft: bool = False,
    ):
        """Load the whole PLE state; `layer_id` is the pool's only unit, 0."""
        if layer_id != 0:
            raise ValueError(
                f"PleStatePoolHost has a single transfer unit, got {layer_id=}"
            )
        for region_index, region in enumerate(self.regions):
            for entry_index, device_tensor in enumerate(region.device_tensors):
                copy_state_slots_pf_lf(
                    src=self.region_buffers[region_index],
                    dst=device_tensor,
                    src_indices=host_indices,
                    dst_indices=device_indices,
                    layer_id=entry_index,
                    num_layers=len(region.device_tensors),
                    io_backend=io_backend,
                )

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend="kernel"
    ):
        for region_index, region in enumerate(self.regions):
            copy_state_slots_all_layers_lf_pf(
                src_layers=region.device_tensors,
                dst=self.region_buffers[region_index],
                src_indices=device_indices,
                dst_indices=host_indices,
                num_layers=len(region.device_tensors),
                io_backend=io_backend,
                src_ptrs=self.device_ptrs[region_index],
            )

    def _iter_page_tensors(self, index: int):
        for buffer in self.region_buffers:
            yield buffer[index]

    @staticmethod
    def _flatten_tensor_bytes(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.contiguous().view(torch.uint8).reshape(-1)

    def get_data_page(self, index, flat: bool = True) -> torch.Tensor:
        data_page = torch.cat(
            [
                self._flatten_tensor_bytes(tensor)
                for tensor in self._iter_page_tensors(index)
            ]
        )
        return data_page.flatten() if flat else data_page

    def get_dummy_flat_data_page(self) -> torch.Tensor:
        return torch.zeros(
            self.page_size * self.size_per_token,
            dtype=torch.uint8,
            device=self.device,
            pin_memory=self.pin_memory,
        )

    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        flat_bytes = data_page.contiguous().view(torch.uint8).reshape(-1)
        start = 0
        for tensor in self._iter_page_tensors(index):
            num_bytes = tensor.numel() * tensor.element_size()
            tensor_bytes = flat_bytes[start : start + num_bytes]
            start += num_bytes
            tensor.copy_(tensor_bytes.view(dtype=tensor.dtype).reshape(tensor.shape))

    def get_page_buffer_meta(self, indices):
        """Per-page (pointer, size) pairs for zero-copy L3 I/O, region-major."""
        assert len(indices) % self.page_size == 0
        indices = indices.tolist()
        base_ptrs = [buffer.data_ptr() for buffer in self.region_buffers]
        row_bytes = [
            region.slot_bytes * len(region.device_tensors) for region in self.regions
        ]
        ptr_list = []
        element_size_list = []
        for i in range(0, len(indices), self.page_size):
            for region_index in range(len(self.regions)):
                ptr_list.append(
                    base_ptrs[region_index] + indices[i] * row_bytes[region_index]
                )
                element_size_list.append(self.page_size * row_bytes[region_index])
        return ptr_list, element_size_list

    def is_stride_page_aligned(self, page_size_bytes: int = 4096) -> bool:
        for buffer, region in zip(self.region_buffers, self.regions, strict=True):
            stride = region.slot_bytes * len(region.device_tensors)
            if buffer.data_ptr() % page_size_bytes != 0:
                return False
            if stride % page_size_bytes != 0:
                return False
        return True
