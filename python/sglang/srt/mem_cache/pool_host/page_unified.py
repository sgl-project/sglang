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

"""The ``page_unified`` host KV layout: one page block's byte order.

One page block, in this byte order::

    MHA: (head_group, layer, 2, page_size, head_in_group, head_dim)   K=0, V=1
    MLA: (layer, page_size, head_dim)

A PARTITION AXIS MUST BE OUTER. The unified L3 grid cuts a page along the kv-head
axis and then along the layer axis, so both sit outside ``page_size``: a chunk is
then an offset and a length, not a stride pattern. Put either one inside and
naming a sub-range costs a descriptor per token.

The K/V axis sits INSIDE the layer axis rather than outside the pool, which is
what makes a chunk a single run and lets one object carry both components.
``layer_first`` and ``page_first_direct`` both hoist it outside, so under those
a chunk is two objects whose runs interleave at token granularity.

The same order is what the write-back kernel produces and the load-back kernel
reads, so a page is byte-identical whether it arrived from the device or from
L3, and neither side has to ask which.
"""

from __future__ import annotations

import msgspec

# Byte order identity, recorded in the L3 namespace digest. Bump when the axis
# order changes, so old objects miss instead of deserializing permuted.
PAGE_UNIFIED_OBJECT_LAYOUT = "page-unified-v1"


class PageUnifiedLayout(msgspec.Struct, frozen=True, kw_only=True):
    """Byte layout of one ``page_unified`` page block.

    Built directly from a host pool's own dimensions. ``head_group_num`` is the
    number of kv-head GROUPS the page is cut into, which is the unified grid's
    head partition; MLA is rank-replicated and has no head axis, so both it and
    ``head_num`` are 1 there.
    """

    page_size: int
    layer_num: int
    head_num: int
    head_group_num: int
    head_dim: int
    itemsize: int
    is_mla: bool = False

    def __post_init__(self) -> None:
        for name in (
            "page_size",
            "layer_num",
            "head_num",
            "head_group_num",
            "head_dim",
            "itemsize",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive: {self}")
        if self.head_num % self.head_group_num != 0:
            raise ValueError(
                f"head_group_num={self.head_group_num} must divide the pool's "
                f"{self.head_num} kv heads."
            )
        if self.is_mla and (self.head_group_num != 1 or self.head_num != 1):
            raise ValueError(f"MLA page blocks have no head axis: {self}")
        # The transfer kernels move a token's head-group run as 16-byte vectors.
        if self.group_bytes % 16 != 0:
            raise ValueError(
                f"one head group's token row is {self.group_bytes} bytes, which "
                f"the page_unified transfer kernels cannot move: they require a "
                f"multiple of 16. Raise head_dim, lower head_group_num, or use "
                f"another --hicache-mem-layout."
            )

    @property
    def components(self) -> int:
        """K and V, or the single latent cache for MLA."""
        return 1 if self.is_mla else 2

    @property
    def heads_per_group(self) -> int:
        return self.head_num // self.head_group_num

    @property
    def group_bytes(self) -> int:
        """One head group's row for one token: the copy unit."""
        return self.heads_per_group * self.head_dim * self.itemsize

    @property
    def layer_bytes(self) -> int:
        """One (head group, layer) cell: both components, all tokens of a page."""
        return self.components * self.page_size * self.group_bytes

    @property
    def head_group_bytes(self) -> int:
        return self.layer_num * self.layer_bytes

    @property
    def bytes_per_page(self) -> int:
        return self.head_group_num * self.head_group_bytes

    def page_dims(self, page_num: int) -> tuple[int, ...]:
        """Buffer shape, whose C-contiguous order IS the byte order."""
        if self.is_mla:
            return (page_num, self.layer_num, self.page_size, self.head_dim)
        return (
            page_num,
            self.head_group_num,
            self.layer_num,
            self.components,
            self.page_size,
            self.heads_per_group,
            self.head_dim,
        )

    def chunk_span(
        self, layer_range: tuple[int, int], head_group: int = 0
    ) -> tuple[int, int]:
        """Byte ``(offset, length)`` of one L3 chunk within a page block.

        A chunk is a half-open LOCAL layer range within one head group. It is
        always exactly one run, including a ragged final layer window -- that
        is the property the axis order buys, and what lets the transport treat
        every chunk uniformly instead of switching on a descriptor count.
        """
        start, end = layer_range
        if not 0 <= start < end <= self.layer_num:
            raise ValueError(
                f"layer range {layer_range} is not a non-empty sub-range of "
                f"[0, {self.layer_num})."
            )
        if not 0 <= head_group < self.head_group_num:
            raise ValueError(
                f"head group {head_group} is outside [0, {self.head_group_num})."
            )
        offset = head_group * self.head_group_bytes + start * self.layer_bytes
        return offset, (end - start) * self.layer_bytes
