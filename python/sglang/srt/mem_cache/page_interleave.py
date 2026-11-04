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

- GQA/MHA models: the **attention CP group** — the prefill CP strategy
  materializes the full chunk's K/V on every CP rank before the pool write.
- MLA models: the **attention TP group** — the latent KV projection is
  ``ReplicatedLinear``, so every attn-TP rank computes identical latent KV.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Optional, Tuple

import msgspec
import torch

from sglang.srt.mem_cache import page_interleave_pool
from sglang.srt.mem_cache.allocator.page_interleave import (
    PageInterleavePoolAllocator,
)
from sglang.srt.runtime_context import get_parallel, get_schedule, get_spec
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils.common import ceil_align, get_cuda_graph_batch_size_alignment

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state import GroupCoordinator
    from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
    from sglang.srt.mem_cache.page_interleave_pool import PageInterleaveKVPoolMixin

logger = logging.getLogger(__name__)


class PageShardSpec(msgspec.Struct, frozen=True):
    """Everything both sides of any transfer need to reproduce the layout."""

    shard_rank: int
    shard_size: int
    page_size: int  # physical page size (kernel-visible)
    max_prefix_tokens: int  # scratch prefix-region capacity, granule-aligned
    chunk_tokens: int  # scratch chunk-region capacity, physical-page-aligned

    @property
    def scratch_rows(self) -> int:
        """Rows in each slot: assembled prefix, current batch chunk, trash page."""
        return self.max_prefix_tokens + self.chunk_tokens + self.page_size

    @property
    def logical_page_size(self) -> int:
        """The N*page_size span used to align prefix-gather scratch.

        Allocation, prefix matching, and chunk boundaries use the physical
        ``page_size``.
        """
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


def get_kv_shard_group(use_mla_backend: bool) -> GroupCoordinator:
    """The group KV pages are striped across — the axis that replicates KV
    at rest, chosen by topology:

    - An active attention-CP group takes precedence: prefill CP replicates
      KV storage across CP ranks for every attention type (GQA via the
      full-chunk allgather, MLA via the CP latent-KV rebuild).
    - Without CP, MLA latent KV is still replicated across attention-TP
      (ReplicatedLinear projection), so the attn-TP group is the shard axis.
    - GQA without CP has no replicated axis (KV is head-sharded across TP);
      the returned trivial CP group has world_size 1, which disables
      sharding in get_kv_shard_group_info.
    """
    cp_group = get_parallel().attn_cp_group
    if cp_group.world_size > 1:
        return cp_group
    if use_mla_backend:
        return get_parallel().attn_tp_group
    return cp_group


def get_shared_kv_shard_pool(
    kvc: KVCacheConfigurator,
) -> Optional[PageInterleaveKVPoolMixin]:
    """Find a draft's sharded target through their shared logical allocator."""
    if not kvc.is_draft_worker:
        return None
    algorithm = getattr(kvc, "spec_algorithm", None)
    if algorithm is not None and algorithm.is_dflash_family():
        # DFlash owns ordinary attention over the target logical address space.
        return None

    allocator = kvc.token_to_kv_pool_allocator
    if not isinstance(allocator, PageInterleavePoolAllocator):
        return None

    pool = allocator.get_kvcache()
    if not isinstance(pool, page_interleave_pool.PageInterleaveKVPoolMixin):
        raise ValueError("A sharded MTP draft requires the target's sharded KV pool.")
    if (
        allocator.shard_spec != pool.shard_spec
        or allocator.shard_size != pool.shard_spec.shard_size
        or allocator.page_size != pool.page_size
        or allocator.size != pool.size * allocator.shard_size
    ):
        raise ValueError(
            "The shared allocator and target KV pool have different shard layouts."
        )
    return pool


def get_kv_shard_group_info(
    kvc: KVCacheConfigurator,
) -> Tuple[Optional[int], int]:
    """``(shard_rank, shard_size)`` for the KV pool; ``(None, 1)`` disables."""
    if kvc.is_draft_worker:
        pool = get_shared_kv_shard_pool(kvc)
        if pool is None:
            return None, 1
        return pool.shard_spec.shard_rank, pool.shard_spec.shard_size
    if not get_parallel().enable_kv_cache_sharding:
        return None, 1
    group = get_kv_shard_group(kvc.use_mla_backend)
    if group.world_size <= 1:
        return None, 1
    return group.rank_in_group, group.world_size


def _page_shard_tensor_page_bytes(kvc: KVCacheConfigurator) -> Tuple[int, ...]:
    """Byte widths of each separately aligned scratch tensor per physical page."""
    model_config = kvc.model_config
    kv_size = torch._utils._element_size(kvc.kv_cache_dtype)
    if kvc.use_mla_backend:
        from sglang.srt.configs.model_config import (
            get_dsa_index_head_dim,
            get_dsa_index_kpool,
            is_deepseek_dsa,
        )
        from sglang.srt.mem_cache.kv_cache_configurator import (
            calculate_mla_kv_cache_dim,
        )
        from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool

        row_bytes = (
            calculate_mla_kv_cache_dim(
                model_config=model_config, kv_cache_dtype=kvc.kv_cache_dtype
            )
            * kv_size
        )
        tensor_page_bytes = (row_bytes * kvc.page_size,)
        if is_deepseek_dsa(model_config.hf_config):
            dim = get_dsa_index_head_dim(model_config.hf_config)
            kpool = get_dsa_index_kpool(model_config.hf_config)
            if kpool <= 0 or kvc.page_size % kpool:
                raise ValueError("KV-shard page size must be divisible by index_kpool")
            # Match IndexKeyCache._buffer_shape: scales are stored with each
            # pooled key; rounding per token would overcharge pooled layouts.
            index_bytes = dim + dim // DSATokenToKVPool.quant_block_size * 4
            tensor_page_bytes += (kvc.page_size // kpool * index_bytes,)
        return tensor_page_bytes
    head_bytes = (
        kvc.page_size
        * model_config.get_num_kv_heads(get_parallel().attn_tp_size)
        * kv_size
    )
    return (head_bytes * model_config.head_dim, head_bytes * model_config.v_head_dim)


def _page_shard_scratch_bytes(page_bytes: Tuple[int, ...], num_pages: int) -> int:
    # Use the same per-tensor alignment as the symmetric scratch window, even
    # when the pool falls back to ordinary torch allocations.
    return page_interleave_pool._NUM_SCRATCH_SLOTS * sum(
        ceil_align(num_pages * width, page_interleave_pool._SCRATCH_ALIGN_BYTES)
        for width in page_bytes
    )


def make_page_shard_spec(kvc: KVCacheConfigurator) -> Optional[PageShardSpec]:
    """Fit both assembly slots in the per-GPU byte budget.

    Charge current-token, trash, and aligned tensor storage first; the
    remainder determines a gather-granule-aligned prefix capacity. MTP drafts
    inherit the target layout because they serve the same scheduled batch.
    """
    if kvc.is_draft_worker:
        pool = get_shared_kv_shard_pool(kvc)
        return pool.shard_spec if pool is not None else None
    shard_rank, shard_size = get_kv_shard_group_info(kvc)
    if shard_rank is None:
        return None

    scratch_size = get_parallel().kv_cache_sharding_scratch_size
    budget = scratch_size * 1024**3
    if not math.isfinite(budget) or budget <= 0:
        raise ValueError("--kv-cache-sharding-scratch-size must be positive and finite")
    budget = int(budget)

    page_size = kvc.page_size
    granule = shard_size * page_size
    chunk_tokens = ceil_align(get_schedule().chunked_prefill_size, page_size)
    algorithm = getattr(kvc, "spec_algorithm", None)
    minimum_prefix_tokens = kvc.model_config.context_len
    if algorithm is not None and algorithm.is_dflash_family():
        # Each verify request may start on a fresh page. The total token budget
        # alone misses per-request page rounding, and graph buckets can include
        # extra request slots. Charge this bound before sizing persistent KV.
        requests = ceil_align(
            get_schedule().max_running_requests, get_cuda_graph_batch_size_alignment()
        )
        verify_tokens = get_spec().speculative_num_draft_tokens
        minimum_prefix_tokens += 2 * verify_tokens
        chunk_tokens = max(
            chunk_tokens, requests * ceil_align(verify_tokens, page_size)
        )
    page_bytes = _page_shard_tensor_page_bytes(kvc)
    fixed_pages = chunk_tokens // page_size + 1
    # The raw byte quotient is an upper bound; binary search accounts for the
    # separate tensor/window alignments without rounding the budget upwards.
    low, high = 0, budget // (2 * sum(page_bytes) * shard_size)
    while low < high:
        mid = (low + high + 1) // 2
        if (
            _page_shard_scratch_bytes(page_bytes, fixed_pages + mid * shard_size)
            <= budget
        ):
            low = mid
        else:
            high = mid - 1
    max_prefix_tokens = low * granule
    if max_prefix_tokens < ceil_align(minimum_prefix_tokens, granule):
        required = _page_shard_scratch_bytes(
            page_bytes,
            fixed_pages + ceil_align(minimum_prefix_tokens, granule) // page_size,
        )
        raise ValueError(
            "--kv-cache-sharding-scratch-size cannot fit one full context and decode headroom plus "
            f"chunk/trash/alignment in both slots: need at least {required / 1024**3:.6f} GiB"
        )
    return PageShardSpec(
        shard_rank=shard_rank,
        shard_size=shard_size,
        page_size=page_size,
        max_prefix_tokens=max_prefix_tokens,
        # chunked_prefill_size is the TOTAL new-token budget for the batch.
        chunk_tokens=chunk_tokens,
    )


def compute_page_shard_scratch_bytes(
    kvc: KVCacheConfigurator, *, include_mtp: bool = False
) -> int:
    """Fixed HBM cost of the double-buffered assembly scratch, charged against
    the KV budget before pool sizing. Use the same spec as the pool builders:
    two slots, each holding ONE layer's ``[prefix | chunk | trash page]``.

    ``include_mtp`` additionally budgets a target's single EAGLE MTP runner
    at the same KV geometry, as required by the draft pool builder. Its two
    scratch slots are reused across draft layers. Target sizing integration
    can opt into this estimate without changing the per-pool default.
    """
    spec = make_page_shard_spec(kvc)
    if spec is None:
        return 0
    scratch_bytes = _page_shard_scratch_bytes(
        _page_shard_tensor_page_bytes(kvc), spec.scratch_rows // spec.page_size
    )
    if include_mtp:
        if (
            kvc.is_draft_worker
            or kvc.spec_algorithm != SpeculativeAlgorithm.EAGLE
            or (kvc.spec_aux_config.eagle_draft_num_layers or 0) <= 0
            or get_spec().enable_multi_layer_eagle
        ):
            raise ValueError(
                "MTP scratch budgeting requires a target with one EAGLE MTP runner."
            )
        scratch_bytes *= 2
    return scratch_bytes
