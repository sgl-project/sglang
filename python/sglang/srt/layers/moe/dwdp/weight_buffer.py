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
    PagePool,
    PoolBinding,
    compute_slot_sizes,
)
from sglang.srt.utils.vmm_backend import get_vmm_backend

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
        self._backend = get_vmm_backend(device_id)
        self._granularity = self._backend.granularity
        self._pool_page_size = PagePool.DEFAULT_PAGE_SIZE_MULTIPLIER * self._granularity
        self._page_pool: Optional[PagePool] = None
        self._moe_layer_indices = sorted(layer_weight_specs.keys())
        # double buffered: consecutive MoE layers alternate slots
        self._layer_slots = {
            li: pos % 2 for pos, li in enumerate(self._moe_layer_indices)
        }
        self._layouts: Dict[int, Dict[str, PageAlignedLayout]] = {}
        self._tensors: Dict[int, Dict[str, torch.Tensor]] = {}
        self._remote_slices: Dict[
            int, Dict[str, List[Tuple[torch.Tensor, int, int]]]
        ] = {}
        self._reservations: Dict[int, List] = {}
        self._pool_bindings: Dict[int, List[PoolBinding]] = {}
        self._bound_layers: Set[int] = set()
        self._rebinds_pool_pages = not self._backend.supports_aliased_mappings()
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
        """Map this layer's share of the pool pages, evicting the layer that had them.

        Only backends that cannot alias a page into several composite VAs evict;
        there, the other layer on this slot loses everything outside its local
        shard. Callers must drain that layer's device work first -- unmapping is a
        host call and does not wait.
        """
        if layer_idx in self._bound_layers:
            return
        if self._rebinds_pool_pages:
            slot = self.buffer_index_for_layer(layer_idx)
            for other in [
                li
                for li in self._bound_layers
                if self.buffer_index_for_layer(li) == slot
            ]:
                self._unbind_pool_pages(other)
        for binding in self._pool_bindings[layer_idx]:
            self._page_pool.map_binding(binding)
        self._bound_layers.add(layer_idx)

    def _unbind_pool_pages(self, layer_idx: int) -> None:
        for binding in self._pool_bindings[layer_idx]:
            self._page_pool.unmap_binding(binding)
        self._bound_layers.discard(layer_idx)

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
    """Copy the two experts whose bytes straddle the local handle's edge pages.

    Page alignment puts bytes of the experts just outside [local_start, local_end)
    on those pages, where no prefetch ever writes them; seed them from the owner.
    """
    for li in weight_buffer.layer_indices:
        for name in weight_buffer.weight_names(li):
            edge = weight_buffer.get_edge_info(li, name)
            if edge.leading_edge == 0 and edge.trailing_edge == 0:
                continue

            weight_buffer.bind_pool_pages(li)
            full_tensor = weight_buffer.get_full_tensor(li, name)

            if edge.leading_edge > 0 and local_start > 0:
                prev = local_start - 1
                peer = lookup_owner(prev, peer_ranges)
                ps, _ = peer_ranges[peer]
                key = (peer, li, name)
                if key in peer_views:
                    full_tensor[prev].copy_(peer_views[key][prev - ps])

            if edge.trailing_edge > 0 and local_end < full_tensor.shape[0]:
                nxt = local_end
                peer = lookup_owner(nxt, peer_ranges)
                ps, _ = peer_ranges[peer]
                key = (peer, li, name)
                if key in peer_views:
                    full_tensor[nxt].copy_(peer_views[key][nxt - ps])

        if weight_buffer.rebinds_pool_pages:
            # the next layer takes these pool pages back, and unmapping does not
            # wait for the copies above
            weight_buffer.synchronize()

    weight_buffer.synchronize()
