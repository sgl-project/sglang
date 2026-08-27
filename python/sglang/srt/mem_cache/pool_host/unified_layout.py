# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Unified KV page layout and staging for partitioned L3 objects.

Pools expose one normalized page view.  This module turns that view into an
ordered slab plan and implements the shared gather/read/scatter mechanics.
The logical slab order is deliberately defined once here because it must match
the L3 chunk-key ordering exactly.  Read targets may point at a different
physical order in staging so the H2D kernel can consume them without a CPU
repack.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)

Range = tuple[int, int]
Selection = tuple[int | slice, ...]


@dataclass(frozen=True)
class UnifiedKVSlab:
    """One contiguous-or-staged object in a page's transfer plan."""

    component: str
    selection: Selection
    shape: tuple[int, ...]
    nbytes: int
    direct: bool


class UnifiedKVPageLayout:
    """Compiled transfer plan for a pool and one partition grid."""

    def __init__(
        self,
        *,
        page_size: int,
        dtype: torch.dtype,
        page_view: Callable[[int], torch.Tensor],
        sample: torch.Tensor,
        components: tuple[str, ...],
        selections: Sequence[tuple[str, Selection]],
        read_orders: Sequence[Sequence[int]],
    ):
        if sample.ndim != 5:
            raise ValueError(
                "a unified KV page view must have shape "
                "(component, layer, token, head, dim)"
            )
        if sample.shape[0] != len(components):
            raise ValueError(
                f"page view has {sample.shape[0]} components, expected "
                f"{len(components)}"
            )

        self.page_size = page_size
        self.dtype = dtype
        self.page_view = page_view
        self.components = components
        self.slabs = tuple(
            self._make_slab(sample, component, selection)
            for component, selection in selections
        )
        self.read_orders = tuple(tuple(order) for order in read_orders)
        if len(self.read_orders) != len(self.components):
            raise ValueError("read staging needs one slab order per component")
        read_slab_indices = [index for order in self.read_orders for index in order]
        if sorted(read_slab_indices) != list(range(len(self.slabs))):
            raise ValueError("read staging orders must contain every slab exactly once")
        for component, order in zip(self.components, self.read_orders):
            if any(self.slabs[index].component != component for index in order):
                raise ValueError(
                    f"read staging order for {component!r} contains another component"
                )
        self.bytes_per_page = sum(slab.nbytes for slab in self.slabs)
        self.zero_copy = all(slab.direct for slab in self.slabs)

    @staticmethod
    def _make_slab(
        sample: torch.Tensor, component: str, selection: Selection
    ) -> UnifiedKVSlab:
        view = sample[selection]
        return UnifiedKVSlab(
            component=component,
            selection=selection,
            shape=tuple(view.shape),
            nbytes=view.numel() * view.element_size(),
            direct=view.is_contiguous(),
        )

    def _page_starts(self, indices) -> list[int]:
        if len(indices) % self.page_size != 0:
            raise ValueError(
                f"expected complete {self.page_size}-token pages, got "
                f"{len(indices)} indices"
            )
        return indices.tolist()[:: self.page_size]

    def _staging_view(self, staging, cursor: int, slab: UnifiedKVSlab):
        if staging is None:
            raise RuntimeError("this unified KV layout requires a staging buffer")
        return staging[cursor : cursor + slab.nbytes].view(self.dtype).view(slab.shape)

    def _validate_staging(self, staging, required_bytes: int) -> None:
        if self.zero_copy:
            return
        if staging is None:
            raise RuntimeError("this unified KV layout requires a staging buffer")
        if staging.dtype != torch.uint8 or not staging.is_contiguous():
            raise ValueError("unified KV staging must be a contiguous uint8 tensor")
        if staging.numel() < required_bytes:
            raise ValueError(
                f"unified KV staging has {staging.numel()} bytes, "
                f"needs {required_bytes}"
            )

    def _collect_metas(self, indices, staging, *, pack: bool):
        ptrs, sizes = [], []
        cursor = 0
        for index in self._page_starts(indices):
            page = self.page_view(index)
            for slab in self.slabs:
                chunk = page[slab.selection]
                if slab.direct:
                    ptr = chunk.data_ptr()
                else:
                    target = self._staging_view(staging, cursor, slab)
                    if pack:
                        target.copy_(chunk)
                    ptr = staging.data_ptr() + cursor
                ptrs.append(ptr)
                sizes.append(slab.nbytes)
                cursor += slab.nbytes
        return ptrs, sizes

    def _read_staging_plan(self, page_count: int):
        """Map logical (page, slab) entries into an H2D-ready byte buffer.

        Mooncake descriptors stay in page-major L3-key order.  Their target
        pointers assemble staging component-major, with each component laid
        out as ``(page, head group, layer, token, head, dim)``.  In particular,
        MHA produces one contiguous K arena followed by one contiguous V arena.
        """
        offsets = [[0] * len(self.slabs) for _ in range(page_count)]
        regions = {}
        cursor = 0
        for component, order in zip(self.components, self.read_orders):
            region_start = cursor
            for page_pos in range(page_count):
                for slab_index in order:
                    offsets[page_pos][slab_index] = cursor
                    cursor += self.slabs[slab_index].nbytes
            regions[component] = (region_start, cursor - region_start)
        assert cursor == page_count * self.bytes_per_page
        return offsets, regions

    def gather(self, indices, staging):
        """Return write-side pointers, packing only non-contiguous slabs."""
        self._validate_staging(
            staging, len(self._page_starts(indices)) * self.bytes_per_page
        )
        return self._collect_metas(indices, staging, pack=True)

    def read_metas(self, indices, staging):
        """Return key-ordered targets that assemble H2D-ready read staging."""
        page_starts = self._page_starts(indices)
        self._validate_staging(staging, len(page_starts) * self.bytes_per_page)
        offsets, _ = self._read_staging_plan(len(page_starts))
        ptrs, sizes = [], []
        for page_pos, index in enumerate(page_starts):
            page = self.page_view(index)
            for slab_index, slab in enumerate(self.slabs):
                chunk = page[slab.selection]
                ptr = (
                    chunk.data_ptr()
                    if self.zero_copy
                    else staging.data_ptr() + offsets[page_pos][slab_index]
                )
                ptrs.append(ptr)
                sizes.append(slab.nbytes)
        return ptrs, sizes

    def read_component_regions(self, indices):
        """Return ``component -> (byte offset, size)`` for active read staging."""
        page_count = len(self._page_starts(indices))
        _, regions = self._read_staging_plan(page_count)
        return regions

    def scatter(self, indices, staging, page_ok=None) -> int:
        """Copy successfully fetched staged slabs back into the host pool."""
        page_starts = self._page_starts(indices)
        if page_ok is not None and len(page_ok) != len(page_starts):
            raise ValueError(
                f"expected {len(page_starts)} page results, got {len(page_ok)}"
            )
        if page_ok is None or any(page_ok):
            self._validate_staging(staging, len(page_starts) * self.bytes_per_page)
        offsets, _ = self._read_staging_plan(len(page_starts))
        for pos, index in enumerate(page_starts):
            if page_ok is not None and not page_ok[pos]:
                continue
            page = self.page_view(index)
            for slab_index, slab in enumerate(self.slabs):
                if not self.zero_copy:
                    page[slab.selection].copy_(
                        self._staging_view(staging, offsets[pos][slab_index], slab)
                    )
        return len(page_starts) * self.bytes_per_page


class UnifiedKVLayoutHostMixin:
    """Shared unified-layout API for MHA and MLA host pools.

    A pool implements only :meth:`_unified_page_view`, normalized to
    ``(component, layer, token, head, dim)``.  MHA has K/V components and a
    real head axis; MLA has one latent-KV component and a singleton head axis.
    """

    def _unified_page_view(self, index: int) -> torch.Tensor:
        raise NotImplementedError

    def _check_unified_layout_pool(self) -> None:
        if not torch.is_tensor(self.kv_buffer):
            raise NotImplementedError(
                "KV layout adapter is not supported for split K/V host pools."
            )

    def build_unified_layout(
        self, layer_ranges: Sequence[Range], head_ranges: Sequence[Range] | None = None
    ) -> UnifiedKVPageLayout:
        """Compile the partition grid into its canonical object order."""
        self._check_unified_layout_pool()
        sample = self._unified_page_view(0)
        component_count = int(sample.shape[0])
        if component_count == 2:
            components = ("k", "v")
        elif component_count == 1:
            components = ("k",)
        else:
            raise ValueError(
                f"unified KV layout supports one or two components, got "
                f"{component_count}"
            )

        layer_ranges = tuple(layer_ranges)
        if head_ranges is None:
            head_ranges = ((0, int(sample.shape[3])),)
        else:
            head_ranges = tuple(head_ranges)

        selections = []
        slab_indices = {}
        for layer_index, (layer_start, layer_end) in enumerate(layer_ranges):
            for head_index, (head_start, head_end) in enumerate(head_ranges):
                for component_index, component in enumerate(components):
                    slab_indices[(component_index, layer_index, head_index)] = len(
                        selections
                    )
                    selections.append(
                        (
                            component,
                            (
                                component_index,
                                slice(layer_start, layer_end),
                                slice(None),
                                slice(head_start, head_end),
                                slice(None),
                            ),
                        )
                    )

        # L3 keys above are layer-major/head-minor/KV-interleaved.  Incoming
        # objects target the order consumed by transfer_kv_per_layer_pfdhg_lf:
        # separate K/V arenas, each page/head-major with contiguous layers.
        read_orders = [
            [
                slab_indices[(component_index, layer_index, head_index)]
                for head_index in range(len(head_ranges))
                for layer_index in range(len(layer_ranges))
            ]
            for component_index in range(len(components))
        ]
        return UnifiedKVPageLayout(
            page_size=self.page_size,
            dtype=self.dtype,
            page_view=self._unified_page_view,
            sample=sample,
            components=components,
            selections=selections,
            read_orders=read_orders,
        )

    # Compatibility entry points used by storage backends and benchmarks.
    def unified_bytes_per_page(self, layer_ranges, head_ranges=None) -> int:
        return self.build_unified_layout(layer_ranges, head_ranges).bytes_per_page

    def _slab_schedule(self, layer_ranges, head_ranges=None):
        return self.build_unified_layout(layer_ranges, head_ranges).slabs

    def unified_zero_copy(self, layer_ranges, head_ranges=None) -> bool:
        return self.build_unified_layout(layer_ranges, head_ranges).zero_copy

    def gather_unified_chunks(self, indices, layer_ranges, head_ranges, staging):
        return self.build_unified_layout(layer_ranges, head_ranges).gather(
            indices, staging
        )

    def get_unified_chunk_meta(self, indices, layer_ranges, head_ranges, staging):
        return self.build_unified_layout(layer_ranges, head_ranges).read_metas(
            indices, staging
        )

    def scatter_unified_chunks(
        self, indices, layer_ranges, head_ranges, staging, page_ok=None
    ) -> int:
        return self.build_unified_layout(layer_ranges, head_ranges).scatter(
            indices, staging, page_ok
        )


class KVCacheLayoutAdapter:
    """Bind a compiled layout plan to L3 keys and staging buffers."""

    def __init__(self, mem_pool_host, storage_config, register_buffer=None):
        self.pool = mem_pool_host
        self.page_size = mem_pool_host.page_size
        self.layer_ranges = storage_config.unified_layer_ranges
        self.head_ranges = storage_config.unified_head_ranges
        suffixes = storage_config.unified_suffix
        if not isinstance(suffixes, list) or self.layer_ranges is None:
            raise ValueError(
                "the unified KV layout adapter requires chunk suffixes and "
                "local layer ranges"
            )
        self.suffixes = tuple(suffixes)
        self.layout = self.pool.build_unified_layout(
            self.layer_ranges, self.head_ranges
        )
        self.keys_per_page = len(self.suffixes) * len(self.layout.components)
        if self.keys_per_page != len(self.layout.slabs):
            raise ValueError(
                "unified KV suffix count does not match the compiled slab plan: "
                f"{self.keys_per_page} keys versus {len(self.layout.slabs)} slabs"
            )

        self.staging_set = None
        self.staging_get = None
        self.staging_pages = 0
        if self.layout.zero_copy:
            logger.info(
                "HiCache KV layout adapter: everything pool-contiguous, "
                "zero-copy (no staging buffers)."
            )
            return

        staging_mb = (storage_config.extra_config or {}).get("staging_buffer_mb", 256)
        page_bytes = self.layout.bytes_per_page
        staging_bytes = max(int(staging_mb) << 20, page_bytes)
        self.staging_pages = staging_bytes // page_bytes
        staging_numel = self.staging_pages * page_bytes
        self.staging_set = self._alloc_staging(staging_numel)
        self.staging_get = self._alloc_staging(staging_numel)
        if register_buffer is not None:
            register_buffer(self.staging_set)
            register_buffer(self.staging_get)
        logger.info(
            "HiCache KV layout adapter: 2 x %d-page staging buffers "
            "(%.1f MB each) for the backup and prefetch threads.",
            self.staging_pages,
            staging_numel / (1 << 20),
        )

    def _alloc_staging(self, numel):
        # Pinned so RDMA transports can register and DMA it directly.
        return torch.empty(numel, dtype=torch.uint8, pin_memory=True)

    def chunk_keys(self, page_keys: list) -> list:
        """Expand page keys in the exact order of ``layout.slabs``."""
        return [
            f"{page_key}_{suffix}_{component}"
            for page_key in page_keys
            for suffix in self.suffixes
            for component in self.layout.components
        ]

    def sub_batches(self, keys: list, host_indices):
        """Yield staging-sized page batches, reusing buffers from offset zero."""
        if not keys:
            return
        pages_per_batch = self.staging_pages or len(keys)
        for start in range(0, len(keys), pages_per_batch):
            page_keys = keys[start : start + pages_per_batch]
            indices = host_indices[
                start * self.page_size : (start + len(page_keys)) * self.page_size
            ]
            yield page_keys, indices

    def gather(self, indices):
        return self.layout.gather(indices, self.staging_set)

    def read_metas(self, indices):
        return self.layout.read_metas(indices, self.staging_get)

    def read_component_views(self, indices):
        """Return active K/V byte arenas assembled by ``read_metas``.

        The transfer kernel must pair these compact arenas with
        :meth:`read_source_indices`, never with the original host-pool slots.
        The views remain owned by this adapter and are invalidated when the
        next staging sub-batch starts.
        """
        if self.staging_get is None:
            return {}
        return {
            component: self.staging_get[offset : offset + size]
            for component, (offset, size) in self.layout.read_component_regions(
                indices
            ).items()
        }

    def read_source_indices(self, indices, *, device=None):
        """Return compact token positions for an H2D read-staging arena."""
        self.layout._page_starts(indices)
        return torch.arange(len(indices), dtype=torch.int64, device=device)

    def scatter(self, indices, page_ok):
        if self.staging_get is None or not any(page_ok):
            return
        self.layout.scatter(indices, self.staging_get, page_ok)
