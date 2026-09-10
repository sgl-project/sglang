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

A pool exposes one normalized page view; this module compiles it into an
ordered slab plan matching the L3 chunk-key order exactly.

Every plan is a direct copy: a chunk names byte runs of host pool memory, so a
get lands in the pool and a put reads out of it, with no staging or CPU repack.
The host layout changes only the NUMBER of runs -- the object order is always
``(layer, token, head_in_group, dim)`` per component, which is what lets a
``page_first`` writer and a ``page_first_direct`` reader share one keyspace.

Why layer-major: A PARTITION AXIS MUST BE OUTER. Chunks are layer ranges, which
in a layer-major block are an offset and a length; in a token-major one, layers
are interior and naming a sub-range costs a run per token. So this order makes
``page_first_direct`` one descriptor for any layer partition, MLA included. The
head-group-major permutation below applies the same principle to the head axis,
which is interior in every layout.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)

Range = tuple[int, int]


@dataclass(frozen=True)
class UnifiedKVSlab:
    """One L3 object's footprint inside a page.

    ``runs`` is ``(byte offset from the component's page base, length)`` pairs
    in unified order -- one pair when the page block already IS that order,
    several otherwise. No byte is copied on the CPU either way.
    """

    component: str
    component_index: int
    runs: tuple[tuple[int, int], ...]
    nbytes: int

    @property
    def contiguous(self) -> bool:
        return len(self.runs) == 1


class UnifiedKVPageLayout:
    """Compiled transfer plan for a pool and one partition grid.

    Two strategies produce a chunk's runs:

    * **Permuted** (``page_first_direct``, head cut, kernel io backend): the
      page block is stored head-group-major, so a chunk is ONE run and the
      pfdhg kernels absorb the permutation.
    * **Strided** (everything else): runs come from the page view's strides.
      ``page_first_direct`` uncut collapses to one run; ``page_first`` yields
      ``(l1 - l0) * P``.

    :attr:`contiguous_chunks` is a property of the whole PLAN, not a slab: the
    transport picks its scatter/gather arm once per batch from the first
    pointer (``MooncakeStore._uses_multi_buffer``), so a mixed plan would hand
    it bare integers. A ragged layer partition is what mixes them -- L=61 with
    layer_partition=30 leaves a width-1 tail.
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
        permuted: bool = False,
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

        self.page_size = page_size
        self.dtype = dtype
        self.page_view = page_view
        self.components = components
        self.layer_ranges = tuple(tuple(r) for r in layer_ranges)
        self.head_ranges = tuple(tuple(r) for r in head_ranges)
        self.head_group_num = len(self.head_ranges)
        self.permuted = permuted
        self.itemsize = torch.tensor([], dtype=dtype).element_size()

        # Chunk offsets come from a single heads-per-group stride, so the head
        # axis must be a uniform, ordered, gapless tiling or they silently
        # mis-address. One equality rejects all three failures.
        self.heads_per_group = (
            self.head_ranges[0][1] - self.head_ranges[0][0] if self.head_ranges else 0
        )
        if (
            not self.head_ranges
            or self.head_ranges
            != tuple(
                (i * self.heads_per_group, (i + 1) * self.heads_per_group)
                for i in range(self.head_group_num)
            )
            or (self.heads_per_group * self.head_group_num != head_num)
        ):
            raise ValueError(
                f"unified head ranges {self.head_ranges} must tile this pool's "
                f"{head_num} kv heads uniformly, in order and without gaps"
            )
        if permuted and not all(
            sample[c].is_contiguous() for c in range(len(components))
        ):
            raise ValueError(
                "the head-group-major page order needs a contiguous page block"
            )

        # Layer-major / head-minor / KV-interleaved: this order must stay in
        # lockstep with build_unified_suffixes and chunk_keys.
        slabs = []
        for layer_start, layer_end in self.layer_ranges:
            for head_index, (head_start, head_end) in enumerate(self.head_ranges):
                for component_index, component in enumerate(components):
                    runs = (
                        self._permuted_runs(
                            layer_num, head_dim, head_index, layer_start, layer_end
                        )
                        if permuted
                        else self._strided_runs(
                            sample,
                            component_index,
                            layer_start,
                            layer_end,
                            head_start,
                            head_end,
                        )
                    )
                    slabs.append(
                        UnifiedKVSlab(
                            component=component,
                            component_index=component_index,
                            runs=runs,
                            nbytes=sum(n for _, n in runs),
                        )
                    )
        self.slabs = tuple(slabs)
        self.bytes_per_page = sum(slab.nbytes for slab in self.slabs)
        self.descriptors_per_page = sum(len(slab.runs) for slab in self.slabs)
        # See the class docstring: uniform per plan, never per slab.
        self.contiguous_chunks = all(slab.contiguous for slab in self.slabs)
        # chunk_metas is O(pages x slabs x runs) per batch, so precompute
        # everything page-independent -- only the base pointer varies.
        if self.contiguous_chunks:
            self._slab_offsets = tuple(slab.runs[0][0] for slab in self.slabs)
            self._slab_sizes = tuple(slab.runs[0][1] for slab in self.slabs)
        else:
            self._slab_offsets = tuple(
                [off for off, _ in slab.runs] for slab in self.slabs
            )
            # chunk_metas still hands out a per-call copy: the transport
            # marshals the list through pybind, and callers must not be able to
            # reach back into the plan.
            self._slab_sizes = tuple([n for _, n in slab.runs] for slab in self.slabs)
        self._slab_components = tuple(slab.component_index for slab in self.slabs)

        # A layer the grid does not name (MTP draft pools append them)
        # transfers in the wrong byte order. Compare the RANGES, not the total
        # width: an overlap that double-covers one and drops another sums the
        # same.
        named = sum(end - start for start, end in self.layer_ranges)
        contiguous_from_zero = self.layer_ranges and all(
            start == (0 if i == 0 else self.layer_ranges[i - 1][1]) and end > start
            for i, (start, end) in enumerate(self.layer_ranges)
        )
        if not contiguous_from_zero or named != layer_num:
            raise ValueError(
                f"the unified L3 grid {self.layer_ranges} must tile this "
                f"pool's {layer_num} layers in order and without gaps; it "
                f"names {named}. Pools with layers outside the grid (e.g. MTP "
                f"draft layers) are not supported; use "
                f"--hicache-storage-key-scheme rank-suffix."
            )

    def _permuted_runs(self, layer_num, head_dim, head_index, layer_start, layer_end):
        group_stride = layer_num * self.page_size * self.heads_per_group * head_dim
        layer_stride = self.page_size * self.heads_per_group * head_dim
        offset = (
            head_index * group_stride + layer_start * layer_stride
        ) * self.itemsize
        return (((offset, (layer_end - layer_start) * layer_stride * self.itemsize)),)

    def _strided_runs(
        self, sample, component_index, layer_start, layer_end, head_start, head_end
    ):
        """Coalesce a chunk into maximal contiguous runs of the page view.

        Merges axes inwards-out while each stride equals the run so far, then
        emits one descriptor per unmerged-axis combination, OUTERMOST-first
        because the object order is layer-major. Sorting by address would
        transpose the object -- ``page_first`` stores tokens outside layers, so
        ascending addresses are token-major.
        """
        strides = [int(x) for x in sample[component_index].stride()]
        stride_l, stride_t, stride_h, _ = strides
        head_dim = int(sample.shape[4])
        # innermost -> outermost
        axes = [
            (head_dim, int(strides[3])),
            (head_end - head_start, stride_h),
            (self.page_size, stride_t),
            (layer_end - layer_start, stride_l),
        ]
        run, merged = 1, 0
        for count, stride in axes:
            if stride != run:
                break
            run *= count
            merged += 1

        offsets = [0]
        for count, stride in reversed(axes[merged:]):  # outermost first
            offsets = [o + i * stride for o in offsets for i in range(count)]

        base = layer_start * stride_l + head_start * stride_h
        return tuple(((base + o) * self.itemsize, run * self.itemsize) for o in offsets)

    def _page_starts(self, indices) -> list[int]:
        if len(indices) % self.page_size != 0:
            raise ValueError(
                f"expected complete {self.page_size}-token pages, got "
                f"{len(indices)} indices"
            )
        # Slice before .tolist(), which would otherwise build one Python int
        # per TOKEN and discard all but every page_size-th.
        starts = indices[:: self.page_size].tolist()
        # A misaligned start addresses a DIFFERENT page, not a wrong offset in
        # the right one: cheap to check, unbounded to debug.
        misaligned = [i for i in starts if i % self.page_size]
        if misaligned:
            raise ValueError(
                f"host indices must start on a {self.page_size}-token page "
                f"boundary; got {misaligned[:4]}"
            )
        return starts

    def chunk_metas(self, indices):
        """Return ``(pointers, sizes)`` addressing host pool memory directly.

        A get writes these ranges, a put reads them. Under
        :attr:`contiguous_chunks` every chunk is a plain ``int``, otherwise
        every chunk is a ``list`` -- including single-run ones. Order is
        page-major then :attr:`slabs`, matching ``chunk_keys``.
        """
        ptrs, sizes = [], []
        contiguous = self.contiguous_chunks
        components = range(len(self.components))
        for index in self._page_starts(indices):
            page = self.page_view(index)
            bases = [page[c].data_ptr() for c in components]
            for component, offsets, run_sizes in zip(
                self._slab_components, self._slab_offsets, self._slab_sizes
            ):
                base = bases[component]
                if contiguous:
                    ptrs.append(base + offsets)
                    sizes.append(run_sizes)
                else:
                    ptrs.append([base + off for off in offsets])
                    sizes.append(list(run_sizes))
        return ptrs, sizes


class UnifiedKVLayoutHostMixin:
    """Shared unified-layout API for MHA and MLA host pools.

    A pool implements only :meth:`_unified_page_view`, normalized to
    ``(component, layer, token, head, dim)``. MHA has K/V components and a real
    head axis; MLA has one latent-KV component and a singleton one.
    """

    # How this pool's page blocks are ordered: 1 is the host layout's own
    # order, >1 is head-group-major (pfdhg kernels only). Set at L3 attach.
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

        Refuses while pages are live: the pool has no per-page provenance, so a
        mid-flight switch would leave two orders in one pool.
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
        self,
        layer_ranges: Sequence[Range],
        head_ranges: Sequence[Range] | None = None,
        permute_head_groups: bool = True,
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
        # Head-group-major turns a cut-head chunk from many runs into one, but
        # only page_first_direct has a contiguous block to permute and only the
        # pfdhg kernels can read it back -- hence the caller's veto.
        permuted = (
            permute_head_groups
            and self.layout == "page_first_direct"
            and len(head_ranges) > 1
        )
        return UnifiedKVPageLayout(
            page_size=self.page_size,
            dtype=self.dtype,
            page_view=self._unified_page_view,
            sample=sample,
            components=components,
            layer_ranges=layer_ranges,
            head_ranges=head_ranges,
            permuted=permuted,
        )

    # Test/benchmark only: these recompile the plan per call. Production goes
    # through KVCacheLayoutAdapter, which compiles it once at attach.
    def unified_bytes_per_page(self, layer_ranges, head_ranges=None) -> int:
        return self.build_unified_layout(layer_ranges, head_ranges).bytes_per_page

    def get_unified_chunk_meta(self, indices, layer_ranges, head_ranges=None):
        """Host pool ``(pointers, sizes)`` for one page batch, in slab order."""
        return self.build_unified_layout(layer_ranges, head_ranges).chunk_metas(indices)


class KVCacheLayoutAdapter:
    """Bind a compiled layout plan to this rank's L3 keys.

    :meth:`chunk_keys` and :meth:`chunk_metas` return parallel lists -- key
    ``i`` names the bytes at pointer ``i``. Both address host pool memory
    directly, so the adapter holds no buffers and does no packing.
    """

    def __init__(self, mem_pool_host, storage_config):
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
            self.layer_ranges,
            self.head_ranges,
            permute_head_groups=storage_config.unified_permute_head_groups,
        )
        self.keys_per_page = len(self.suffixes) * len(self.layout.components)
        if self.keys_per_page != len(self.layout.slabs):
            raise ValueError(
                "unified KV suffix count does not match the compiled slab plan: "
                f"{self.keys_per_page} keys versus {len(self.layout.slabs)} slabs"
            )
        # Makes the transfer paths pick the permuting kernels, so it must
        # happen before any page is written.
        self.pool.set_unified_head_groups(
            self.layout.head_group_num if self.layout.permuted else 1
        )
        # descriptors/page is the layout's cost to the transport; a fleet grid
        # mismatch is a silent total miss. Log both.
        logger.info(
            "HiCache KV layout adapter: %d chunks/page, %d descriptors/page, "
            "%d head group(s), layer ranges %s, %s. Direct copy in both "
            "directions (no staging).",
            self.keys_per_page,
            self.layout.descriptors_per_page,
            self.layout.head_group_num,
            self.layer_ranges,
            (
                "head-group-major page blocks"
                if self.layout.permuted
                else f"{self.pool.layout} page blocks"
            ),
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
