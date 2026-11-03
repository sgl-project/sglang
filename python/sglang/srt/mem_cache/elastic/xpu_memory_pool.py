"""
Copyright 2025 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Elastic MHA pool for Intel XPU, backed by kvcached's Level Zero VMM.

kvcached's XPU build has no ETensor; it exposes per-layer KV tensors
(`create_kv_tensors`) whose pages are mapped by byte offset
(`map_to_kv_tensors` / `unmap_from_kv_tensors`), one allocator group per pool.
This pool speaks the same cu_page interface as `ElasticMHATokenToKVPool`, so the
elastic allocators and orchestrator drive it unchanged.
"""

from __future__ import annotations

import logging
import math
from typing import List, Tuple

import torch

from sglang.srt.mem_cache.elastic.elastic_memory_pool import ElasticMHATokenToKVPool
from sglang.srt.mem_cache.elastic.elasticmem_orchestrator import (
    cu_page_size,
    install_shutdown_handlers,
)

logger = logging.getLogger(__name__)

# kvcached rejects page sizes that are not a multiple of 2 MiB, on every backend.
KVCACHED_PAGE_ALIGN = 2 << 20

_kvcached_initialized = False


def init_kvcached_xpu_once() -> None:
    """Initialize kvcached on the current XPU; idempotent."""
    global _kvcached_initialized
    if _kvcached_initialized:
        return

    import kvcached.vmm_ops as vmm_ops

    assert vmm_ops.backend_name() == "XPU", (
        f"kvcached is built for {vmm_ops.backend_name()}, not XPU; "
        "rebuild it with KVCACHED_BACKEND=xpu"
    )
    # The pool slices per-layer K/V views, so kvcached's compound-page
    # (contiguous) layout is not used; kvcached defaults it off on XPU too.
    vmm_ops.init_kvcached(
        f"xpu:{torch.xpu.current_device()}", cu_page_size(), contiguous_layout=False
    )
    install_shutdown_handlers(vmm_ops.shutdown_kvcached)
    _kvcached_initialized = True


class ElasticXPUMHATokenToKVPool(ElasticMHATokenToKVPool):
    """XPU variant of `ElasticMHATokenToKVPool`: a virtual range sized from
    device memory, of which only the pages under `size + page_size` rows are
    mapped. A map/unmap at page offset `o` covers that page of K and of V in
    every layer, so one offset counts as `2 * layer_num` cu_pages.
    """

    # Process-wide: one kvcached allocator group per pool instance.
    _next_group_id = 0

    def _create_buffers_normal(self):
        assert self.kv_cache_layout == "nhd", (
            f"elasticmem supports the NHD KV layout only, got {self.kv_cache_layout}"
        )
        self.group_id = ElasticXPUMHATokenToKVPool._next_group_id
        ElasticXPUMHATokenToKVPool._next_group_id += 1
        self.mapped_pages = 0
        from kvcached.vmm_ops import create_kv_tensors

        init_kvcached_xpu_once()
        itemsize = self.store_dtype.itemsize
        self.k_row_bytes = self.head_num * self.head_dim * itemsize
        self.v_row_bytes = self.head_num * self.v_head_dim * itemsize
        # One offset maps K and V together, so the wider row decides how many
        # offsets a row count needs; the allocator reads it as the row size.
        self.state_memsize = max(self.k_row_bytes, self.v_row_bytes)
        self.page_bytes = cu_page_size()
        assert self.page_bytes % KVCACHED_PAGE_ALIGN == 0, (
            f"SGLANG_ELASTIC_MEM_CU_PAGE_SIZE={self.page_bytes} must be a multiple "
            f"of {KVCACHED_PAGE_ALIGN} for kvcached"
        )
        # Virtual only, sized like kvcached's own alloc_kv_cache. Physical
        # memory never exceeds what SGLang budgeted: the pools only trade pages.
        total_memory = torch.xpu.get_device_properties(
            torch.xpu.current_device()
        ).total_memory
        self.kv_region_bytes = (
            total_memory // self.layer_num // 2 // self.page_bytes * self.page_bytes
        )
        self.max_size = self._esize_to_size(self.kv_region_bytes // self.state_memsize)
        assert self.size <= self.max_size, (
            f"elasticmem: {self.size=} exceeds the virtual capacity {self.max_size}"
        )

        self.raw_kv_buffer = create_kv_tensors(
            2 * self.kv_region_bytes,
            itemsize,
            f"xpu:{torch.xpu.current_device()}",
            self.layer_num,
            num_kv_buffers=2,
            group_id=self.group_id,
            page_size=self.page_bytes,
        )
        # Level Zero has no shared zero page: every row a kernel can touch
        # must be mapped before first use, or the access faults.
        self._map_pages(self._pages_for_size(self.size))
        self._slice_kv_views()
        logger.info(
            f"elasticmem xpu {self._pool_name()}: group={self.group_id}, "
            f"{self.size=}, {self.max_size=}, {self.page_bytes=}, "
            f"{self.state_memsize=}"
        )

    def get_kv_size_bytes(self):
        # Physically mapped bytes; the virtual range behind the views is larger.
        kv_size = self.mapped_pages * self.page_bytes * self.layer_num
        return kv_size, kv_size

    def reduce(self, new_size: int) -> Tuple[int, int]:
        keep_pages = self._pages_for_size(new_size)
        # Rows in the last kept page stay usable, but never beyond the old size.
        size_after = min(self.size, self._size_for_pages(keep_pages))
        unmap_num = self._unmap_pages_from(keep_pages)
        self._set_size(size_after)
        return unmap_num, self.size

    def expand(self, new_size: int) -> Tuple[int, int]:
        need_pages = self._pages_for_size(min(new_size, self.max_size))
        map_num = self._map_pages(need_pages)
        self._set_size(max(self.size, self._size_for_pages(need_pages)))
        return map_num, self.size

    def cu_page_to_token(self, cu_page_num: int) -> int:
        # Exact: the tokens that fit once that many more pages are mapped.
        budget_pages = cu_page_num // (2 * self.layer_num)
        new_size = self._size_for_pages(self.mapped_pages + budget_pages)
        return max(new_size - self.size, 0)

    # ---- helpers ----

    def _pool_name(self) -> str:
        return (self.allocation_label or "kv").lower()

    def _pages_for_size(self, size: int) -> int:
        return math.ceil(
            self._size_to_esize(size) * self.state_memsize / self.page_bytes
        )

    def _size_for_pages(self, num_pages: int) -> int:
        rows = num_pages * self.page_bytes // self.state_memsize
        return min(self._esize_to_size(rows), self.max_size)

    def _page_offsets(self, start_page: int, end_page: int) -> List[int]:
        return [page * self.page_bytes for page in range(start_page, end_page)]

    def _map_pages(self, num_pages: int) -> int:
        from kvcached.vmm_ops import map_to_kv_tensors

        offsets = self._page_offsets(self.mapped_pages, num_pages)
        if not offsets:
            return 0
        if not map_to_kv_tensors(offsets, group_id=self.group_id):
            raise RuntimeError(
                f"elasticmem xpu {self._pool_name()}: kvcached failed to map "
                f"{len(offsets)} pages"
            )
        self.mapped_pages = num_pages
        return len(offsets) * 2 * self.layer_num

    def _unmap_pages_from(self, num_pages: int) -> int:
        from kvcached.vmm_ops import unmap_from_kv_tensors

        offsets = self._page_offsets(num_pages, self.mapped_pages)
        if not offsets:
            return 0
        if not unmap_from_kv_tensors(offsets, group_id=self.group_id):
            raise RuntimeError(
                f"elasticmem xpu {self._pool_name()}: kvcached failed to unmap "
                f"{len(offsets)} pages"
            )
        self.mapped_pages = num_pages
        return len(offsets) * 2 * self.layer_num

    def _set_size(self, size: int) -> None:
        self.size = size
        self._slice_kv_views()
        # PD transfer and the descriptor check expect exactly size + page_size rows.
        self._kv_buffer_descs = self._build_kv_buffer_descs()

    def _slice_kv_views(self) -> None:
        """Point k/v_buffer at exactly `size + page_size` rows of the raw tensors;
        rows past them may be unmapped, and an unmapped read faults on XPU."""
        rows = self._size_to_esize(self.size)
        k_elems = self.k_row_bytes // self.store_dtype.itemsize
        v_elems = self.v_row_bytes // self.store_dtype.itemsize
        v_base = self.kv_region_bytes // self.store_dtype.itemsize
        k_buffer, v_buffer = [], []
        for raw in self.raw_kv_buffer:
            flat = raw.view(self.store_dtype)
            k_buffer.append(
                flat[: rows * k_elems].view(rows, self.head_num, self.head_dim)
            )
            v_buffer.append(
                flat[v_base : v_base + rows * v_elems].view(
                    rows, self.head_num, self.v_head_dim
                )
            )
        self.k_buffer, self.v_buffer = k_buffer, v_buffer
