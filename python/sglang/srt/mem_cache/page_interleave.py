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

"""Logical-page KV sharding: placement arithmetic and shard-group resolution.

A *logical page* is ``shard_size`` consecutive, ``shard_size``-aligned physical
pages — one per rank of the shard group. Everything above the memory pool
(radix tree, allocator free list, ``req_to_token``, scheduler budgets) sees
only logical token slots, identical on every rank (SPMD); everything below
sees only its own physical pages; the boundary is the pure bijection below.

The shard group is the group across which KV storage is replicated today and
therefore can be striped without extra compute-time communication:

- GQA/MHA models: the **attention CP group** — prefill CP already allgathers
  the full chunk's K/V to every CP rank (``cp_allgather_and_save_kv_cache``).
- MLA models: the **attention TP group** — the latent KV projection is
  ``ReplicatedLinear``, so every attn-TP rank computes identical latent KV.
"""

from __future__ import annotations

import logging

import msgspec
import torch

logger = logging.getLogger(__name__)


class PageShardSpec(msgspec.Struct, frozen=True):
    """Everything both sides of any transfer need to reproduce the layout."""

    shard_rank: int
    shard_size: int
    page_size: int  # physical page size (kernel-visible)
    max_prefix_tokens: int  # scratch prefix-region capacity, granule-aligned
    chunk_tokens: int  # scratch chunk-region capacity, granule-aligned

    @property
    def logical_page_size(self) -> int:
        """The N*page_size span of one logical page. It bounds the assembly
        scratch and rounds chunked_prefill_size; the allocator and the tree
        both keep drawing and matching at the physical ``page_size``."""
        return self.shard_size * self.page_size


class PageInterleavePlacement:
    """``loc = Q*(N*ps) + r*ps + o`` -> owner ``r``, local physical row ``Q*ps + o``.

    Pure, stateless and invertible, so nothing has to be stored or kept
    coherent to translate. Because the allocator is
    mirrored, logical group ``Q`` resolves to local rows ``[Q*ps, (Q+1)*ps)``
    on every rank, so a reader computes a peer's source offset from arithmetic
    alone. Owned tokens form ``page_size``-long contiguous runs in both the
    logical and the local space.
    """

    def __init__(self, spec: PageShardSpec):
        self.spec = spec

    def owner_of(self, loc: torch.Tensor) -> torch.Tensor:
        ps, n = self.spec.page_size, self.spec.shard_size
        return (loc % (n * ps)) // ps

    def local_index(self, loc: torch.Tensor) -> torch.Tensor:
        ps, n = self.spec.page_size, self.spec.shard_size
        return (loc // (n * ps)) * ps + loc % ps

    def local_mask(self, loc: torch.Tensor, rank: int) -> torch.Tensor:
        return self.owner_of(loc) == rank

    def filter_local(self, loc: torch.Tensor, rank: int) -> torch.Tensor:
        """Logical slots -> this rank's physical pool rows, order-preserving."""
        return self.local_index(loc[self.local_mask(loc, rank)])
