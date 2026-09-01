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

"""Unified KV page layout for partitioned L3 objects.

Pools expose one normalized page view.  This module turns that view into an
ordered slab plan whose order must match the L3 chunk-key ordering exactly.

Every plan is a **direct copy**: each L3 chunk is one contiguous byte range of
host pool memory, so a get lands straight in the pool and a put reads straight
out of it.  There is no staging buffer and no CPU repack in either direction.

Cutting the kv-head axis is what makes that non-trivial.  A chunk covers one
head group across a layer range, which is strided in ``page_first_direct``'s
own ``(layer, token, head, dim)`` page block.  So a pool whose L3 grid cuts
heads stores its page blocks head-group-major instead::

    (head_group, layer, token, head_in_group, dim)

the same bytes, permuted, which makes every chunk contiguous again (see
``_permuted_slab``).  Both KV transfer directions absorb that permutation into
the copy they were already doing --- ``transfer_kv_per_layer_pfdhg_lf`` on H2D
and ``transfer_kv_all_layer_lf_pfdhg`` on D2H --- so the pool has exactly one
byte order regardless of whether a page arrived from L3 or from the device.
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
    """One L3 object's footprint inside a page, as a contiguous byte range."""

    component: str
    component_index: int
    # Byte offset of this chunk inside its component's page block.
    byte_offset: int
    nbytes: int


class UnifiedKVPageLayout:
    """Compiled transfer plan for a pool and one partition grid.

    The plan is a list of ``(byte offset, size)`` chunks inside a page,
    ordered to match the L3 chunk keys exactly.  Chunk ``(head group g,
    layers [l0, l1))`` of a component lives at::

        offset = (g * L * P * hg * D + l0 * P * hg * D) * itemsize
        size   = ((l1 - l0) * P * hg * D) * itemsize

    ``L`` is the pool's full layer count, ``P`` the page size, ``hg`` the heads
    per group and ``D`` the head dim.  With a single head group this reduces to
    ``l0 * P * H * D`` --- exactly ``page_first_direct``'s natural order --- so
    one formula covers both the cut and uncut cases, and ``head_group_num == 1``
    means the pool needs no permuting kernel at all.
    """

    def __init__(
        self,
        *,
        page_size: int,
        dtype: torch.dtype,
        page_view: Callable[[int], torch.Tensor],
        sample: torch.Tensor,
        components: tuple[str, ...],
        layer_ranges: Sequence[Range],
        head_ranges: Sequence[Range],
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
        _, layer_num, page_tokens, head_num, head_dim = (int(d) for d in sample.shape)
        if page_tokens != page_size:
            raise ValueError(
                f"page view spans {page_tokens} tokens, expected {page_size}"
            )
        for component_index in range(len(components)):
            if not sample[component_index].is_contiguous():
                raise ValueError(
                    "the unified KV layout needs each component's page block to "
                    "be contiguous so an L3 chunk is one byte range; this pool "
                    "layout stores pages strided."
                )

        self.page_size = page_size
        self.dtype = dtype
        self.page_view = page_view
        self.components = components
        self.layer_ranges = tuple(tuple(r) for r in layer_ranges)
        self.head_ranges = tuple(tuple(r) for r in head_ranges)
        self.head_group_num = len(self.head_ranges)
        self.itemsize = torch.tensor([], dtype=dtype).element_size()

        heads_per_group = {end - start for start, end in self.head_ranges}
        if len(heads_per_group) != 1:
            raise ValueError(
                f"unified head ranges must be uniform, got {self.head_ranges}"
            )
        self.heads_per_group = heads_per_group.pop()
        if self.heads_per_group * self.head_group_num != head_num:
            raise ValueError(
                f"unified head ranges {self.head_ranges} do not tile this "
                f"pool's {head_num} kv heads"
            )

        group_stride = layer_num * page_size * self.heads_per_group * head_dim
        layer_stride = page_size * self.heads_per_group * head_dim
        # L3 keys are layer-major / head-minor / KV-interleaved; the slab order
        # below must stay in lockstep with build_unified_suffixes and with
        # KVCacheLayoutAdapter.chunk_keys.
        slabs = []
        for layer_start, layer_end in self.layer_ranges:
            for head_index, (head_start, _) in enumerate(self.head_ranges):
                if head_start != head_index * self.heads_per_group:
                    raise ValueError(
                        f"unified head ranges must be ordered and gapless, got "
                        f"{self.head_ranges}"
                    )
                for component_index, component in enumerate(components):
                    slabs.append(
                        UnifiedKVSlab(
                            component=component,
                            component_index=component_index,
                            byte_offset=(
                                head_index * group_stride + layer_start * layer_stride
                            )
                            * self.itemsize,
                            nbytes=(layer_end - layer_start)
                            * layer_stride
                            * self.itemsize,
                        )
                    )
        self.slabs = tuple(slabs)
        self.bytes_per_page = sum(slab.nbytes for slab in self.slabs)

        # The chunks must tile every component's page block exactly. This is
        # what rejects a pool carrying layers the L3 grid does not name --- an
        # MTP draft pool appends its draft layers to layer_num, and under a head
        # cut they would sit *inside* each head group's region, so the offsets
        # above would be wrong and the untiled bytes would be silently
        # transferred in the other byte order.
        expected = (
            len(components)
            * self.head_group_num
            * layer_num
            * layer_stride
            * self.itemsize
        )
        if self.bytes_per_page != expected:
            raise ValueError(
                f"the unified L3 grid covers {self.bytes_per_page} bytes per "
                f"page but this pool's pages are {expected} bytes: the grid "
                f"names {sum(e - s for s, e in self.layer_ranges)} of "
                f"{layer_num} layers. Pools with layers outside the grid "
                f"(e.g. MTP draft layers) are not supported; use "
                f"--hicache-storage-key-scheme rank-suffix."
            )
        self._check_slab_coverage(layer_num, layer_stride)

    def _check_slab_coverage(self, layer_num: int, layer_stride: int) -> None:
        """Every byte of every component page block is named exactly once."""
        block_bytes = self.head_group_num * layer_num * layer_stride * self.itemsize
        for component_index in range(len(self.components)):
            spans = sorted(
                (slab.byte_offset, slab.byte_offset + slab.nbytes)
                for slab in self.slabs
                if slab.component_index == component_index
            )
            cursor = 0
            for start, end in spans:
                if start != cursor:
                    raise ValueError(
                        f"unified chunks do not tile component "
                        f"{self.components[component_index]!r}: gap or overlap "
                        f"at byte {cursor} (next chunk starts at {start})"
                    )
                cursor = end
            if cursor != block_bytes:
                raise ValueError(
                    f"unified chunks cover {cursor} of {block_bytes} bytes of "
                    f"component {self.components[component_index]!r}"
                )

    @property
    def permuted(self) -> bool:
        """True when page blocks are head-group-major rather than natural."""
        return self.head_group_num > 1

    def _page_starts(self, indices) -> list[int]:
        if len(indices) % self.page_size != 0:
            raise ValueError(
                f"expected complete {self.page_size}-token pages, got "
                f"{len(indices)} indices"
            )
        starts = indices.tolist()[:: self.page_size]
        # Chunk pointers are derived from the page block (index // page_size),
        # so a misaligned start silently addresses a DIFFERENT page rather than
        # a wrong offset inside the right one. Cheap to check, unbounded to
        # debug.
        misaligned = [i for i in starts if i % self.page_size]
        if misaligned:
            raise ValueError(
                f"host indices must start on a {self.page_size}-token page "
                f"boundary; got {misaligned[:4]}"
            )
        return starts

    def chunk_metas(self, indices):
        """Return ``(pointers, sizes)`` addressing host pool memory directly.

        Used for both directions: an L3 get writes these ranges and an L3 put
        reads them.  Order is page-major then :attr:`slabs` order, matching
        :meth:`KVCacheLayoutAdapter.chunk_keys`.
        """
        ptrs, sizes = [], []
        for index in self._page_starts(indices):
            page = self.page_view(index)
            bases = [
                page[component_index].data_ptr()
                for component_index in range(len(self.components))
            ]
            for slab in self.slabs:
                ptrs.append(bases[slab.component_index] + slab.byte_offset)
                sizes.append(slab.nbytes)
        return ptrs, sizes


class UnifiedKVLayoutHostMixin:
    """Shared unified-layout API for MHA and MLA host pools.

    A pool implements only :meth:`_unified_page_view`, normalized to
    ``(component, layer, token, head, dim)``.  MHA has K/V components and a
    real head axis; MLA has one latent-KV component and a singleton head axis.
    """

    # Number of head groups this pool's page blocks are ordered by. 1 means
    # page_first_direct's natural order; >1 means head-group-major, and the KV
    # transfer paths must use the pfdhg kernels. Set at L3 attach by
    # set_unified_head_groups() and reset on detach.
    unified_head_groups: int = 1

    def _unified_page_view(self, index: int) -> torch.Tensor:
        raise NotImplementedError

    def _check_unified_layout_pool(self) -> None:
        if not torch.is_tensor(self.kv_buffer):
            raise NotImplementedError(
                "KV layout adapter is not supported for split K/V host pools."
            )

    def set_unified_head_groups(self, head_groups: int) -> None:
        """Declare this pool's page-block byte order.

        Refuses to change order while pages are live: the pool has no per-page
        provenance, so a mid-flight switch would leave pages of both orders in
        one pool and the transfer kernels would read half of them wrong.
        """
        if head_groups == self.unified_head_groups:
            return
        in_use = int(self.slot_used.sum())
        if in_use:
            raise RuntimeError(
                f"cannot switch the host pool to {head_groups} head groups "
                f"while {in_use} slots are resident: pages already written in "
                f"the current order would be read back permuted. Attach the L3 "
                f"backend before serving, or flush the cache first."
            )
        self.unified_head_groups = head_groups

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
        if head_ranges is None:
            head_ranges = ((0, int(sample.shape[3])),)
        return UnifiedKVPageLayout(
            page_size=self.page_size,
            dtype=self.dtype,
            page_view=self._unified_page_view,
            sample=sample,
            components=components,
            layer_ranges=layer_ranges,
            head_ranges=head_ranges,
        )

    # Compatibility entry points used by storage backends and benchmarks.
    def unified_bytes_per_page(self, layer_ranges, head_ranges=None) -> int:
        return self.build_unified_layout(layer_ranges, head_ranges).bytes_per_page

    def _slab_schedule(self, layer_ranges, head_ranges=None):
        return self.build_unified_layout(layer_ranges, head_ranges).slabs

    def get_unified_chunk_meta(self, indices, layer_ranges, head_ranges):
        return self.build_unified_layout(layer_ranges, head_ranges).chunk_metas(indices)


class KVCacheLayoutAdapter:
    """Bind a compiled layout plan to L3 keys.

    Both directions are direct copies into and out of host pool memory, so the
    adapter owns no buffers and does no packing.
    """

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
        # Declaring the order is what makes the transfer paths pick the
        # permuting kernels; it must happen before any page is written.
        self.pool.set_unified_head_groups(self.layout.head_group_num)
        logger.info(
            "HiCache KV layout adapter: %d chunks/page, %d head group(s), "
            "direct copy in both directions (no staging).",
            self.keys_per_page,
            self.layout.head_group_num,
        )

    def chunk_keys(self, page_keys: list) -> list:
        """Expand page keys in the exact order of ``layout.slabs``."""
        return [
            f"{page_key}_{suffix}_{component}"
            for page_key in page_keys
            for suffix in self.suffixes
            for component in self.layout.components
        ]

    def chunk_metas(self, indices):
        """Host pool ``(pointers, sizes)`` in ``chunk_keys`` order."""
        return self.layout.chunk_metas(indices)
