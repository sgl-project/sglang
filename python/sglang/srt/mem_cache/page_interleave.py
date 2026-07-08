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
See ``DESIGN_kv_sharding_logical_page.md``.

The shard group is the group across which KV storage is replicated today and
therefore can be striped without extra compute-time communication:

- GQA/MHA models: the **attention CP group** — prefill CP already allgathers
  the full chunk's K/V to every CP rank (``cp_allgather_and_save_kv_cache``).
- MLA models: the **attention TP group** — the latent KV projection is
  ``ReplicatedLinear``, so every attn-TP rank computes identical latent KV.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Tuple

import msgspec
import torch

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state import GroupCoordinator
    from sglang.srt.model_executor.model_runner import ModelRunner

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
        """The allocation granule and radix-tree match quantum."""
        return self.shard_size * self.page_size


class PageInterleavePlacement:
    """``loc = Q*(N*ps) + r*ps + o`` -> owner ``r``, local physical row ``Q*ps + o``.

    Pure, stateless, and invertible — the reason no virtual-to-physical table
    or per-page ownership metadata exists anywhere. Because the allocator is
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


def is_kv_cache_sharding_enabled(model_runner: ModelRunner) -> bool:
    return (
        not model_runner.is_draft_worker
        and model_runner.server_args.enable_kv_cache_sharding
    )


def get_kv_shard_group(use_mla_backend: bool) -> GroupCoordinator:
    """The group KV pages are striped across — the axis that replicates KV
    at rest, chosen by topology:

    - An active attention-CP group takes precedence: prefill CP replicates
      KV storage across CP ranks for every attention type (GQA via the
      full-chunk allgather, MLA via rebuild_cp_kv_cache).
    - Without CP, MLA latent KV is still replicated across attention-TP
      (ReplicatedLinear projection), so the attn-TP group is the shard axis.
    - GQA without CP has no replicated axis (KV is head-sharded across TP);
      the returned trivial CP group has world_size 1, which disables
      sharding in get_kv_shard_group_info.
    """
    from sglang.srt.layers.dp_attention import (
        get_attention_cp_group,
        get_attention_tp_group,
    )

    cp_group = get_attention_cp_group()
    if cp_group.world_size > 1:
        return cp_group
    if use_mla_backend:
        return get_attention_tp_group()
    return cp_group


def get_kv_shard_group_info(
    model_runner: ModelRunner,
) -> Tuple[Optional[int], int]:
    """``(shard_rank, shard_size)`` for the KV pool; ``(None, 1)`` disables."""
    if not is_kv_cache_sharding_enabled(model_runner):
        return None, 1
    group = get_kv_shard_group(model_runner.use_mla_backend)
    if group.world_size <= 1:
        return None, 1
    return group.rank_in_group, group.world_size


def compute_page_shard_scratch_bytes(model_runner: ModelRunner) -> int:
    """Fixed HBM cost of the double-buffered assembly scratch, charged against
    the KV budget before pool sizing (the layer-split ``remote_kv_buffer``
    precedent). Two slots, each ``[max prefix | chunk | trash page]`` rows of
    ONE layer's KV."""
    shard_rank, shard_size = get_kv_shard_group_info(model_runner)
    if shard_rank is None:
        return 0

    from sglang.srt.layers.dp_attention import get_attention_tp_size
    from sglang.srt.utils.common import ceil_align

    model_config = model_runner.model_config
    granule = shard_size * model_runner.page_size
    rows = (
        ceil_align(model_config.context_len, granule)
        + ceil_align(model_runner.server_args.chunked_prefill_size, granule)
        + model_runner.page_size
    )
    kv_size = torch._utils._element_size(model_runner.kv_cache_dtype)
    if model_runner.use_mla_backend:
        row_bytes = (
            model_config.kv_lora_rank + model_config.qk_rope_head_dim
        ) * kv_size
    else:
        row_bytes = (
            model_config.get_num_kv_heads(get_attention_tp_size())
            * (model_config.head_dim + model_config.v_head_dim)
            * kv_size
        )
    return 2 * rows * row_bytes
