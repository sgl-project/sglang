# Adapted from NVIDIA TensorRT-LLM (https://github.com/NVIDIA/TensorRT-LLM)
"""Composite VA presenting a contiguous full-expert weight tensor per (layer, weight)."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch

from sglang.srt.cuda_vmm_utils import (
    VmmReservation,
    get_device_granularity,
    make_device_allocation_prop,
    tensor_from_pointer,
)
from sglang.srt.layers.moe.dwdp.layout import (
    EdgeInfo,
    LayerWeightSpecs,
    MnnvlHandleSet,
    PageAlignedLayout,
)
from sglang.srt.layers.moe.dwdp.page_pool import PagePool, compute_slot_sizes

logger = logging.getLogger(__name__)


class WeightBuffer:
    def __init__(
        self,
        layer_weight_specs: LayerWeightSpecs,
        handles: MnnvlHandleSet,
        local_start: int,
        local_end: int,
        dwdp_size: int,
        device_id: int,
    ):
        self._layer_weight_specs = layer_weight_specs
        self._handles = handles
        self._local_start = local_start
        self._local_end = local_end
        self._dwdp_size = dwdp_size
        self._device_id = device_id
        self._granularity = get_device_granularity(device_id)
        self._prop = make_device_allocation_prop(device_id, handle_types=None)
        self._pool_page_size = PagePool.DEFAULT_PAGE_SIZE_MULTIPLIER * self._granularity
        self._page_pool: Optional[PagePool] = None
        self._moe_layer_indices = sorted(layer_weight_specs.keys())
        self._layouts: Dict[int, Dict[str, PageAlignedLayout]] = {}
        self._tensors: Dict[int, Dict[str, torch.Tensor]] = {}
        self._remote_slices: Dict[
            int, Dict[str, List[Tuple[torch.Tensor, int, int]]]
        ] = {}
        self._reservations: Dict[int, List[VmmReservation]] = {}
        self._released = False

    @classmethod
    def create(
        cls,
        layer_weight_specs: LayerWeightSpecs,
        handles: MnnvlHandleSet,
        local_start: int,
        local_end: int,
        dwdp_size: int,
        device_id: int,
    ) -> WeightBuffer:
        buf = cls(
            layer_weight_specs, handles, local_start, local_end, dwdp_size, device_id
        )
        for li, ws in layer_weight_specs.items():
            buf._layouts[li] = {}
            for name, spec in ws.items():
                buf._layouts[li][name] = PageAlignedLayout.compute(
                    expert_bytes=spec.expert_bytes,
                    num_experts=spec.num_experts,
                    local_start=local_start,
                    local_end=local_end,
                    granularity=buf._granularity,
                    handle_phys_size=handles.get_size(li, name),
                    pool_granularity=buf._pool_page_size,
                )

        assignments = {li: buf.buffer_index_for_layer(li) for li in layer_weight_specs}
        slot_sizes = compute_slot_sizes(buf._layouts, assignments)
        buf._page_pool = PagePool.create(
            slot_sizes, device_id, page_size=buf._pool_page_size
        )

        for li in buf._moe_layer_indices:
            buf._setup_layer(li)

        logger.debug(
            f"WeightBuffer created for {len(buf._moe_layer_indices)} layers, "
            f"local [{local_start}, {local_end})"
        )
        return buf

    def _setup_layer(self, layer_idx: int) -> None:
        weight_layouts = self._layouts[layer_idx]
        weight_specs = self._layer_weight_specs[layer_idx]
        buf_slot = self.buffer_index_for_layer(layer_idx)

        self._tensors[layer_idx] = {}
        self._remote_slices[layer_idx] = {}
        self._reservations[layer_idx] = []

        page_pool_offset = 0

        for name, layout in weight_layouts.items():
            spec = weight_specs[name]
            handle = self._handles.get_handle(layer_idx, name)

            reservation = VmmReservation(
                layout.total_size,
                self._prop,
                self._device_id,
                alignment=self._granularity,
            )
            self._reservations[layer_idx].append(reservation)
            va_base = reservation.base

            if layout.pre_size > 0:
                self._page_pool.map_pages(
                    slot=buf_slot,
                    reservation=reservation,
                    offset=0,
                    size=layout.pre_size,
                    page_offset=page_pool_offset,
                )
                page_pool_offset += layout.pre_pages

            reservation.map_existing(layout.pre_size, layout.mnnvl_size, handle)

            if layout.post_size > 0:
                self._page_pool.map_pages(
                    slot=buf_slot,
                    reservation=reservation,
                    offset=layout.pre_size + layout.mnnvl_size,
                    size=layout.post_size,
                    page_offset=page_pool_offset,
                )
                page_pool_offset += layout.post_pages

            tensor_start = va_base + layout.pre_padding
            full_tensor = tensor_from_pointer(
                tensor_start,
                layout.num_experts * layout.expert_bytes,
                shape=spec.full_shape,
                dtype=spec.dtype,
                device_id=self._device_id,
            )

            self._tensors[layer_idx][name] = full_tensor

            slices = []
            if self._local_start > 0:
                slices.append((full_tensor[: self._local_start], 0, self._local_start))
            if self._local_end < spec.num_experts:
                slices.append(
                    (full_tensor[self._local_end :], self._local_end, spec.num_experts)
                )
            self._remote_slices[layer_idx][name] = slices

    def get_full_tensor(self, layer_idx: int, name: str) -> torch.Tensor:
        return self._tensors[layer_idx][name]

    def get_remote_slices(
        self, layer_idx: int, name: str
    ) -> List[Tuple[torch.Tensor, int, int]]:
        return self._remote_slices[layer_idx][name]

    def get_edge_info(self, layer_idx: int, name: str) -> EdgeInfo:
        return self._layouts[layer_idx][name].get_edge_info()

    def get_layout(self, layer_idx: int, name: str) -> PageAlignedLayout:
        return self._layouts[layer_idx][name]

    @property
    def layer_indices(self) -> List[int]:
        return list(self._moe_layer_indices)

    @property
    def local_start(self) -> int:
        return self._local_start

    @property
    def local_end(self) -> int:
        return self._local_end

    @property
    def device_id(self) -> int:
        return self._device_id

    def weight_names(self, layer_idx: int) -> List[str]:
        return list(self._layer_weight_specs[layer_idx].keys())

    def buffer_index_for_layer(self, layer_idx: int) -> int:
        if layer_idx in self._moe_layer_indices:
            return self._moe_layer_indices.index(layer_idx) % 2
        return layer_idx % 2

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        for reservations in self._reservations.values():
            for reservation in reservations:
                reservation.close()
        self._reservations.clear()
        self._tensors.clear()
        self._remote_slices.clear()
        if self._page_pool is not None:
            self._page_pool.release()
            self._page_pool = None
