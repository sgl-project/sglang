"""WeightBuffer: composite VA layout with zero-copy local + page-pool remote.

Creates a contiguous [num_experts, ...] tensor per (layer, weight) by
mapping:
  - pre_region (pool pages): experts before local_start
  - mnnvl_region (fabric handle): local experts — zero copy
  - post_region (pool pages): experts after local_end

Ported from TensorRT-LLM ``_torch/modules/dwdp/weight_buffer.py``.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch

from sglang.srt.layers.moe.dwdp.page_pool import PagePool, compute_slot_sizes
from sglang.srt.layers.moe.dwdp.specs import (
    EdgeInfo,
    LayerWeightSpecs,
    MnnvlHandleSet,
    PageAlignedLayout,
)
from sglang.srt.layers.moe.dwdp.vmm import (
    free_va,
    get_allocation_granularity,
    map_handle,
    reserve_va,
    set_access,
    tensor_from_ptr,
    unmap_va,
)

logger = logging.getLogger(__name__)


class WeightBuffer:
    """Composite VA layout with zero-copy local + page-pool-backed remote."""

    __slots__ = (
        "_layer_weight_specs",
        "_handles",
        "_local_start",
        "_local_end",
        "_dwdp_size",
        "_device_id",
        "_granularity",
        "_pool_page_size",
        "_page_pool",
        "_moe_layer_indices",
        # per-layer state
        "_layouts",  # {layer_idx: {name: PageAlignedLayout}}
        "_tensors",  # {layer_idx: {name: Tensor}}
        "_remote_slices",  # {layer_idx: {name: [(slice, start, end)]}}
        "_mappings",  # {layer_idx: [(va, size)]}
        "_va_regions",  # {layer_idx: [(va, size)]}
        "_released",
    )

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
        self._granularity = get_allocation_granularity(device_id)
        self._pool_page_size = PagePool.DEFAULT_PAGE_SIZE_MULTIPLIER * self._granularity
        self._page_pool: Optional[PagePool] = None
        self._moe_layer_indices = sorted(layer_weight_specs.keys())
        self._layouts: Dict[int, Dict[str, PageAlignedLayout]] = {}
        self._tensors: Dict[int, Dict[str, torch.Tensor]] = {}
        self._remote_slices: Dict[
            int, Dict[str, List[Tuple[torch.Tensor, int, int]]]
        ] = {}
        self._mappings: Dict[int, List[Tuple[int, int]]] = {}
        self._va_regions: Dict[int, List[Tuple[int, int]]] = {}
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
        try:
            # 1. Compute layouts
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

            # 2. Page pool
            assignments = {
                li: buf.buffer_index_for_layer(li) for li in layer_weight_specs
            }
            slot_sizes = compute_slot_sizes(buf._layouts, assignments)
            buf._page_pool = PagePool.create(
                slot_sizes, device_id, page_size=buf._pool_page_size
            )

            # 3. Setup each layer
            for li in buf._moe_layer_indices:
                buf._setup_layer(li)

            logger.info(
                f"[WeightBuffer] Created for {len(buf._moe_layer_indices)} layers, "
                f"local [{local_start}, {local_end})"
            )
            return buf
        except Exception:
            buf.release()
            raise

    def _setup_layer(self, layer_idx: int) -> None:
        weight_layouts = self._layouts[layer_idx]
        weight_specs = self._layer_weight_specs[layer_idx]
        buf_slot = self.buffer_index_for_layer(layer_idx)

        self._tensors[layer_idx] = {}
        self._remote_slices[layer_idx] = {}
        self._mappings[layer_idx] = []
        self._va_regions[layer_idx] = []

        page_pool_offset = 0

        for name, layout in weight_layouts.items():
            spec = weight_specs[name]
            handle = self._handles.get_handle(layer_idx, name)

            va_base = reserve_va(layout.total_size, self._granularity)
            all_maps: List[Tuple[int, int]] = []

            try:
                # pre region
                if layout.pre_size > 0:
                    pre_maps = self._page_pool.map_pages(
                        slot=buf_slot,
                        va_start=va_base,
                        size=layout.pre_size,
                        page_offset=page_pool_offset,
                    )
                    all_maps.extend(pre_maps)
                    page_pool_offset += layout.pre_pages

                # mnnvl region
                mnnvl_va = va_base + layout.pre_size
                map_handle(mnnvl_va, layout.mnnvl_size, handle, offset=0)
                all_maps.append((mnnvl_va, layout.mnnvl_size))

                # post region
                if layout.post_size > 0:
                    post_va = mnnvl_va + layout.mnnvl_size
                    post_maps = self._page_pool.map_pages(
                        slot=buf_slot,
                        va_start=post_va,
                        size=layout.post_size,
                        page_offset=page_pool_offset,
                    )
                    all_maps.extend(post_maps)
                    page_pool_offset += layout.post_pages

                # Single set_access for entire composite VA
                set_access(va_base, layout.total_size, self._device_id)

                # Create tensor view
                tensor_start = va_base + layout.pre_padding
                full_tensor = tensor_from_ptr(
                    ptr=tensor_start,
                    shape=spec.full_shape,
                    dtype=spec.dtype,
                    device_id=self._device_id,
                )

            except Exception:
                for va, sz in all_maps:
                    try:
                        unmap_va(va, sz)
                    except Exception:
                        pass
                try:
                    free_va(va_base, layout.total_size)
                except Exception:
                    pass
                raise

            self._mappings[layer_idx].extend(all_maps)
            self._va_regions[layer_idx].append((va_base, layout.total_size))
            self._tensors[layer_idx][name] = full_tensor

            # Compute remote slices
            slices = []
            if self._local_start > 0:
                slices.append((full_tensor[: self._local_start], 0, self._local_start))
            if self._local_end < spec.num_experts:
                slices.append(
                    (full_tensor[self._local_end :], self._local_end, spec.num_experts)
                )
            self._remote_slices[layer_idx][name] = slices

    # --- Public accessors ---

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

    # --- Lifecycle ---

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        for li, maps in self._mappings.items():
            for va, sz in maps:
                try:
                    unmap_va(va, sz)
                except Exception:
                    pass
        for li, regions in self._va_regions.items():
            for va, sz in regions:
                try:
                    free_va(va, sz)
                except Exception:
                    pass
        self._mappings.clear()
        self._va_regions.clear()
        self._tensors.clear()
        self._remote_slices.clear()
        if self._page_pool is not None:
            self._page_pool.release()
            self._page_pool = None

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass
