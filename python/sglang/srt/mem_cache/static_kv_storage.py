"""The pools built without unified memory, as a request's lifecycle sees them.

A request's rows and token ids are taken and given back through one storage
interface, the unified pool's `kv/storage.py::KVStorage`. Pools built without
unified memory answer the same calls through `StaticKVStorage`, which also
holds what only they need: the NPU DeepSeek-V4 c128 pages and the rotation
bases of KV page interleave. Both go when those pools do.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, List, Optional, Tuple

import msgspec
import torch

from sglang.srt.mem_cache.allocator.page_interleave import page_interleave_shard_size
from sglang.srt.mem_cache.kv.state_storage import StateSlotsExhausted
from sglang.srt.mem_cache.kv.storage import SubPoolUsage

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
    from sglang.srt.mem_cache.kv.state_storage import StateStorage
    from sglang.srt.mem_cache.kv.storage import KVStorage
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool


class StaticPoolShape(msgspec.Struct, frozen=True, kw_only=True):
    """What the static pools' usage needs and only the scheduler knows: which
    pools the model has, and the token capacities they were sized to."""

    has_window: bool
    has_state: bool
    # The token pool's capacity without a window, in the allocator's units.
    token_capacity: int
    full_tokens_per_layer: Optional[int] = None
    swa_tokens_per_layer: Optional[int] = None


def kv_storage_of(
    req_to_token_pool: ReqToTokenPool,
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
    *,
    tree_cache: Optional[BasePrefixCache] = None,
    batch: Optional[ScheduleBatch] = None,
    shape: Optional[StaticPoolShape] = None,
) -> KVStorage | StaticKVStorage:
    """What a request's lifecycle and the pool observers go through: a unified
    pool's own storage, else the static pools behind `StaticKVStorage`
    (``tree_cache``, ``batch`` and ``shape`` feed only what the static pools
    need)."""
    storage = token_to_kv_pool_allocator.storage
    if storage is not None:
        return storage
    return StaticKVStorage(
        req_to_token_pool,
        token_to_kv_pool_allocator,
        tree_cache=tree_cache,
        batch=batch,
        shape=shape,
    )


def acquire_rows(
    table: ReqToTokenPool, state: Optional[StateStorage], reqs: List[Req]
) -> Optional[List[int]]:
    """Bind the requests' rows in a static table and, with recurrent state,
    their state and tracking slots; None when the table has too few free rows.
    A request already holding a row or a slot keeps it. If the slots run out
    after the rows were bound, the rows and slots taken go back before
    `StateSlotsExhausted` surfaces."""
    fresh_rows = [req.kv.req_pool_idx is None for req in reqs]
    rows = table.alloc(reqs)
    if rows is None or state is None:
        return rows
    try:
        state.acquire(table, reqs, rows, fresh_rows)
    except StateSlotsExhausted:
        for req, fresh in zip(reqs, fresh_rows):
            if fresh:
                table.free(req)
        raise
    return rows


class StaticKVStorage:
    """The static request table and token allocator behind `KVStorage`'s
    request lifecycle calls; see that class for what each call does."""

    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        *,
        tree_cache: Optional[BasePrefixCache] = None,
        batch: Optional[ScheduleBatch] = None,
        shape: Optional[StaticPoolShape] = None,
    ):
        self.requests = req_to_token_pool
        self._allocator = token_to_kv_pool_allocator
        self._tree_cache = tree_cache
        self._batch = batch
        self._shape = shape

    def acquire(self, reqs: List[Req]) -> Optional[List[int]]:
        return acquire_rows(self.requests, self._allocator.state, reqs)

    def take_tokens(
        self, num_tokens: int, *, make_room: Callable[[int], Any]
    ) -> Optional[torch.Tensor]:
        make_room(num_tokens)
        return self._allocator.alloc(num_tokens)

    def take_extend(
        self,
        *,
        rows: Optional[torch.Tensor],
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        num_tokens: int,
        make_room: Callable[[int], Any],
        window_tail: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        # Over estimate the number of tokens: assume each request needs a new
        # page (one page per CLASS per request under sharding — the min-class
        # availability floor must cover every request's ceil(K_i/N) rounding,
        # matching the N*ps per-request admission reserve).
        allocator = self._allocator
        shard_size = page_interleave_shard_size(allocator)
        conservative_num_tokens = num_tokens + len(seq_lens_cpu) * (
            allocator.page_size * shard_size
        )
        make_room(
            allocator.get_extend_allocation_demand(
                prefix_lens_cpu,
                seq_lens_cpu,
                conservative_num_tokens=conservative_num_tokens,
                shard_size=shard_size,
            )
        )

        is_dsv4 = rows is not None and hasattr(allocator, "c128_attn_allocator")
        extra_alloc_kwargs = {}
        kv_shard_rotation_bases = None
        if shard_size > 1:
            kv_shard_rotation_bases = _kv_shard_rotation_bases(
                tree_cache=self._tree_cache,
                batch=self._batch,
                prefix_lens_cpu=prefix_lens_cpu,
            )
            extra_alloc_kwargs["rotation_bases"] = kv_shard_rotation_bases
        if is_dsv4:
            c128_num_pages = allocator.c128_num_pages_needed(
                prefix_lens_cpu, seq_lens_cpu
            )
            allocator.ensure_c128_capacity(self._tree_cache, c128_num_pages)
            extra_alloc_kwargs["req_pool_indices"] = rows
            # Per-call per-req table for the C128 KV last_loc lookup.
            if self._batch is not None:
                extra_alloc_kwargs["req_to_token_pool"] = self._batch.req_to_token_pool

        if window_tail is not None:
            out = allocator.alloc_extend_swa_tail(
                prefix_lens=prefix_lens,
                prefix_lens_cpu=prefix_lens_cpu,
                seq_lens=seq_lens,
                seq_lens_cpu=seq_lens_cpu,
                last_loc=last_loc,
                extend_num_tokens=num_tokens,
                swa_tail_len=window_tail,
                **extra_alloc_kwargs,
            )
        else:
            out = allocator.alloc_extend(
                prefix_lens,
                prefix_lens_cpu,
                seq_lens,
                seq_lens_cpu,
                last_loc,
                num_tokens,
                **extra_alloc_kwargs,
            )
        if is_dsv4:
            bundle = out
            out = None if bundle is None else bundle.out_full_loc
            if self._batch is not None:
                self._batch.out_cache_loc_dsv4 = bundle

        if out is not None and kv_shard_rotation_bases is not None:
            # The allocator resolved None entries (new chains) in place from the
            # least-full class at each request's turn; record the bases for the
            # radix insert to stamp onto new tree nodes (UnifiedTreeNode.rotation_base).
            for req, base in zip(self._batch.reqs, kv_shard_rotation_bases):
                req.kv_rotation_base = base
        return out

    def take_decode(
        self,
        *,
        rows: Optional[torch.Tensor],
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        make_room: Callable[[int], Any],
    ) -> Optional[torch.Tensor]:
        allocator = self._allocator
        # Over estimate the number of tokens: assume each request needs a new page.
        make_room(len(seq_lens) * allocator.page_size)

        # DSV4-NPU allocator also needs req_pool_indices for C128 KV allocation and
        # returns a DSV4OutCacheLoc bundle; hasattr-gated so others stay unchanged.
        is_dsv4 = rows is not None and hasattr(allocator, "c128_attn_allocator")
        extra_alloc_kwargs = {}
        if is_dsv4:
            c128_num_pages = allocator.c128_num_pages_needed(
                (seq_lens_cpu - 1).clamp(min=0), seq_lens_cpu
            )
            allocator.ensure_c128_capacity(self._tree_cache, c128_num_pages)
            extra_alloc_kwargs["req_pool_indices"] = rows
            # Per-call per-req C128 table for the last_loc lookup.
            if self._batch is not None:
                extra_alloc_kwargs["req_to_token_pool"] = self._batch.req_to_token_pool

        out = allocator.alloc_decode(
            seq_lens, seq_lens_cpu, last_loc, **extra_alloc_kwargs
        )
        if is_dsv4:
            bundle = out
            out = None if bundle is None else bundle.out_full_loc
            if self._batch is not None:
                self._batch.out_cache_loc_dsv4 = bundle
        return out

    def free_tokens(self, ids: torch.Tensor) -> None:
        self._allocator.free(ids)

    def release_row(self, req: Req) -> None:
        # The DSV4-NPU table's free() also releases the c4/c128 state pages.
        self.requests.free(req)

    def take_rows(self, num_rows: int) -> Optional[List[int]]:
        return self.requests.alloc_rows(num_rows)

    def release_rows(self, rows: List[int]) -> None:
        self.requests.free_rows(rows)

    def accounting_violations(self) -> List[str]:
        return self._allocator.verify_byte_accounting()

    def usage(self) -> Tuple[SubPoolUsage, ...]:
        shape, allocator = self._shape, self._allocator
        if shape.has_window:
            (full_capacity, full_available), (swa_capacity, swa_available) = (
                allocator.swa_capacity_and_available(
                    full_capacity=shape.full_tokens_per_layer,
                    swa_capacity=shape.swa_tokens_per_layer,
                )
            )
            usage = [
                SubPoolUsage(
                    name="full",
                    metrics="full",
                    capacity=full_capacity,
                    available=full_available,
                ),
                SubPoolUsage(
                    name="swa",
                    metrics="swa",
                    capacity=swa_capacity,
                    available=swa_available,
                ),
            ]
        else:
            # A state model's token capacity is what its requests can schedule.
            capacity = (
                self.requests.schedulable_token_capacity(allocator.size)
                if shape.has_state
                else shape.token_capacity
            )
            usage = [
                SubPoolUsage(
                    name="full",
                    metrics="full",
                    capacity=capacity,
                    available=allocator.available_size(),
                )
            ]
        if shape.has_state:
            usage.append(
                SubPoolUsage(
                    name="mamba",
                    metrics="mamba",
                    capacity=allocator.state.pool.size,
                    available=allocator.state_slots.available_size(),
                )
            )
        return tuple(usage)


def _kv_shard_rotation_bases(
    tree_cache: BasePrefixCache, batch: ScheduleBatch, prefix_lens_cpu: torch.Tensor
) -> list:
    """Per-request rotation bases ``b_i`` of the batch's chains, host-only.

    The owner class of position-page P is ``(b_i + P) % shard_size``. Rules:

    - Read through ``req.last_node`` at alloc time, never a value cached on
      the request: a ``checkpoint`` can rebind a chunked request onto
      another chain's canonical locs between chunks, changing the base. The
      read goes through ``tree_cache.rotation_base_of`` because the node
      handle is tree-specific (a NodeId on the unified tree).
    - A request without a cached prefix starts a new chain: None here — the
      allocator draws from the least-full class at that request's turn (so
      the draw sees earlier requests' pops in the same batch) and resolves
      the entry in place.
    - A radix-disabled cache keeps no nodes past the root, which carries no
      base; its chunked continuations fall back to the base recorded on the
      request at the previous chunk's alloc (no cross-request reuse, no
      rebind there).
    """
    assert batch is not None and len(batch.reqs) == len(prefix_lens_cpu)
    bases = []
    for i, req in enumerate(batch.reqs):
        if int(prefix_lens_cpu[i]) == 0:
            bases.append(None)
            continue
        node_base = tree_cache.rotation_base_of(req.last_node)
        if node_base is not None:
            bases.append(node_base)
        else:
            assert req.kv_rotation_base is not None, (
                "sharded extend with a cached prefix but no rotation base: "
                "req.last_node carries none and the request recorded none "
                "at a previous alloc"
            )
            bases.append(req.kv_rotation_base)
    return bases
