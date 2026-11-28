# Adapted from NVIDIA TensorRT-LLM (https://github.com/NVIDIA/TensorRT-LLM)
"""Composite VA presenting a contiguous full-expert weight tensor per (layer, weight)."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple

import torch

from sglang.srt.layers.moe.dwdp.layout import (
    EdgeInfo,
    LayerWeightSpecs,
    MnnvlHandleSet,
    PageAlignedLayout,
    PeerRanges,
    lookup_owner,
)
from sglang.srt.layers.moe.dwdp.page_pool import (
    DEFAULT_PAGE_SIZE_MULTIPLIER,
    PagePool,
    PoolBinding,
    compute_slot_sizes,
)
from sglang.srt.utils.vmm_backend import get_vmm_backend
from sglang.srt.utils.vmm_common import Reservation

logger = logging.getLogger(__name__)

# Layer L's compute triggers the prefetch of layer L + PREFETCH_DEPTH.
PREFETCH_DEPTH = 2
# a remapping backend takes back a slot whose layer was launched a layer earlier, so the host-side remap overlaps compute
ALIASED_NUM_SLOTS = PREFETCH_DEPTH
REMAPPED_NUM_SLOTS = PREFETCH_DEPTH + 1


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
        self._backend = get_vmm_backend(device_id)
        self._granularity = self._backend.granularity
        self._pool_page_size = DEFAULT_PAGE_SIZE_MULTIPLIER * self._granularity
        self._page_pool: Optional[PagePool] = None
        self._moe_layer_indices = sorted(layer_weight_specs.keys())
        self._rebinds_pool_pages = not self._backend.supports_aliased_mappings()
        self._num_slots = (
            REMAPPED_NUM_SLOTS if self._rebinds_pool_pages else ALIASED_NUM_SLOTS
        )
        self._layer_slots = {
            layer_idx: position % self._num_slots
            for position, layer_idx in enumerate(self._moe_layer_indices)
        }
        self._layouts: Dict[int, Dict[str, PageAlignedLayout]] = {}
        self._tensors: Dict[int, Dict[str, torch.Tensor]] = {}
        self._remote_slices: Dict[
            int, Dict[str, List[Tuple[torch.Tensor, int, int]]]
        ] = {}
        self._reservations: Dict[int, List[Reservation]] = {}
        self._pool_bindings: Dict[int, List[PoolBinding]] = {}
        self._bound_layers: Set[int] = set()
        # remapping backends only: the layer whose pool pages a slot currently holds
        self._slot_owners: Dict[int, int] = {}
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
        buffer = cls(
            layer_weight_specs, handles, local_start, local_end, dwdp_size, device_id
        )
        for layer_idx, weight_specs in layer_weight_specs.items():
            buffer._layouts[layer_idx] = {}
            for name, spec in weight_specs.items():
                buffer._layouts[layer_idx][name] = PageAlignedLayout.compute(
                    expert_bytes=spec.expert_bytes,
                    num_experts=spec.num_experts,
                    local_start=local_start,
                    local_end=local_end,
                    granularity=buffer._granularity,
                    handle_phys_size=handles.get_size(layer_idx, name),
                    pool_granularity=buffer._pool_page_size,
                )

        assignments = {
            layer_idx: buffer.buffer_index_for_layer(layer_idx)
            for layer_idx in layer_weight_specs
        }
        slot_sizes = compute_slot_sizes(
            buffer._layouts, assignments, num_slots=buffer._num_slots
        )
        buffer._page_pool = PagePool(slot_sizes, device_id, buffer._pool_page_size)

        for layer_idx in buffer._moe_layer_indices:
            buffer._setup_layer(layer_idx)

        logger.debug(
            f"WeightBuffer created for {len(buffer._moe_layer_indices)} layers, "
            f"local [{local_start}, {local_end})"
        )
        return buffer

    def _setup_layer(self, layer_idx: int) -> None:
        weight_layouts = self._layouts[layer_idx]
        weight_specs = self._layer_weight_specs[layer_idx]
        buf_slot = self.buffer_index_for_layer(layer_idx)

        self._tensors[layer_idx] = {}
        self._remote_slices[layer_idx] = {}
        self._reservations[layer_idx] = []
        bindings: List[PoolBinding] = []

        page_pool_offset = 0

        for name, layout in weight_layouts.items():
            spec = weight_specs[name]
            handle = self._handles.get_handle(layer_idx, name)

            reservation = self._backend.make_reservation(
                layout.total_size,
                exportable=False,
                alignment=self._granularity,
            )
            self._reservations[layer_idx].append(reservation)
            va_base = reservation.base

            if layout.pre_size > 0:
                bindings.append(
                    PoolBinding(
                        slot=buf_slot,
                        reservation=reservation,
                        offset=0,
                        num_pages=layout.pre_pages,
                        page_offset=page_pool_offset,
                    )
                )
                page_pool_offset += layout.pre_pages

            reservation.map_existing(layout.pre_size, layout.mnnvl_size, handle)

            if layout.post_size > 0:
                bindings.append(
                    PoolBinding(
                        slot=buf_slot,
                        reservation=reservation,
                        offset=layout.pre_size + layout.mnnvl_size,
                        num_pages=layout.post_pages,
                        page_offset=page_pool_offset,
                    )
                )
                page_pool_offset += layout.post_pages

            tensor_start = va_base + layout.pre_padding
            full_tensor = self._backend.tensor_from_pointer(
                tensor_start,
                layout.num_experts * layout.expert_bytes,
                shape=spec.full_shape,
                dtype=spec.dtype,
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

        self._pool_bindings[layer_idx] = bindings
        self.bind_pool_pages(layer_idx)

    def bind_pool_pages(self, layer_idx: int) -> None:
        """Map this layer's pool pages; a remapping backend first unmaps them from the slot's previous layer, whose device work the caller must have drained."""
        if layer_idx in self._bound_layers:
            return
        if self._rebinds_pool_pages:
            slot = self.buffer_index_for_layer(layer_idx)
            owner = self._slot_owners.get(slot)
            if owner is not None:
                for binding in self._pool_bindings[owner]:
                    self._page_pool.unmap_binding(binding)
                self._bound_layers.discard(owner)
            self._slot_owners[slot] = layer_idx
        for binding in self._pool_bindings[layer_idx]:
            self._page_pool.map_binding(binding)
        self._bound_layers.add(layer_idx)

    def is_bound(self, layer_idx: int) -> bool:
        """Whether this layer's pool pages are mapped into its composite VA."""
        return layer_idx in self._bound_layers

    @property
    def num_slots(self) -> int:
        return self._num_slots

    @property
    def rebinds_pool_pages(self) -> bool:
        """Whether a layer's pool pages have to be remapped before each prefetch."""
        return self._rebinds_pool_pages

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

    @property
    def device(self) -> torch.device:
        return self._backend.torch_device

    def synchronize(self) -> None:
        self._backend.synchronize()

    def weight_names(self, layer_idx: int) -> List[str]:
        return list(self._layer_weight_specs[layer_idx].keys())

    def buffer_index_for_layer(self, layer_idx: int) -> int:
        if layer_idx not in self._layer_slots:
            raise KeyError(f"layer {layer_idx} has no DWDP weight buffer slot")
        return self._layer_slots[layer_idx]

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        for reservations in self._reservations.values():
            for reservation in reservations:
                reservation.close()
        self._reservations.clear()
        self._pool_bindings.clear()
        self._bound_layers.clear()
        self._slot_owners.clear()
        self._tensors.clear()
        self._remote_slices.clear()
        if self._page_pool is not None:
            self._page_pool.release()
            self._page_pool = None


def fill_edge_experts(
    weight_buffer: WeightBuffer,
    peer_views: Dict[Tuple[int, int, str], torch.Tensor],
    *,
    local_start: int,
    local_end: int,
    peer_ranges: PeerRanges,
) -> None:
    """Seed the two experts that straddle the local handle's edge pages from their owners, since no prefetch writes those pages."""
    for layer_idx in weight_buffer.layer_indices:
        for name in weight_buffer.weight_names(layer_idx):
            edge = weight_buffer.get_edge_info(layer_idx, name)
            if edge.leading_edge == 0 and edge.trailing_edge == 0:
                continue

            weight_buffer.bind_pool_pages(layer_idx)
            full_tensor = weight_buffer.get_full_tensor(layer_idx, name)
            edge_experts = []
            if edge.leading_edge > 0 and local_start > 0:
                edge_experts.append(local_start - 1)
            if edge.trailing_edge > 0 and local_end < full_tensor.shape[0]:
                edge_experts.append(local_end)
            for expert in edge_experts:
                peer_rank = lookup_owner(expert, peer_ranges)
                peer_start, _ = peer_ranges[peer_rank]
                full_tensor[expert].copy_(
                    peer_views[(peer_rank, layer_idx, name)][expert - peer_start]
                )

        if weight_buffer.rebinds_pool_pages:
            # the next layer takes these pool pages back, and unmapping does not wait for the copies
            weight_buffer.synchronize()

    if not weight_buffer.rebinds_pool_pages:
        weight_buffer.synchronize()
