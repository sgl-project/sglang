"""The HiCache HiSparse backing: the radix tree and its host tier own the KV.

Attention KV stays in the regular GPU pool, so prefixes are shared and hot KV
stays in HBM; HiCache writes a prefix back to host only when the tree evicts it.
The indexer keeps scoring the whole history from GPU -- prefill indexer KV is
copied once into a private expanded region at admission, and decode-written
indexer KV lives in pool pages, which are never evicted (only the page-aligned
prefill prefix is tree-owned, and only tree nodes are evictable).

    prefill      -> KV and indexer written to the same page indices
    admit        -> copy the prefix's indexer pages into the expanded region,
                    take a temp device buffer, release the tree lock
    decode       -> indexer scores through a hybrid page table (expanded page for
                    an evicted prefix page, the original otherwise); the swap-in
                    kernel fetches the selected positions HiCache moved to host
    finish       -> release expanded pages, temp slots, node claims, host locks

Where a position lives is therefore per-position and changes over time -- the
whole difference from the private-host backing, whose staging makes the host copy
complete by construction. Two consequences shape everything below:

- **Eviction has to be coordinated.** `req_to_token` carries a `-1` sentinel for
  an evicted position and `req_to_host_pool` its host row; the tree reports
  evictions on the scheduler thread, and `_sync_evictions` applies the writes at
  the one controlled point between two forwards, so no in-flight forward sees a
  sentinel without its host index.
- **Admission has to be rationed** before the prefill forward runs; that half
  lives in `managers/hisparse_hicache_admission.py`.

CUDA-graph safety: every tensor a captured kernel reads is persistent, including
the host pool's base address -- HiCache attaches its host tier after capture, so
the address travels through a device tensor rather than a kernel argument.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Optional

import msgspec
import torch

from sglang.kernels.ops.kvcache.hisparse import load_cache_to_device_buffer_mla
from sglang.srt.environ import envs
from sglang.srt.managers.hisparse_hicache_admission import (
    AdmissionLedger,
    HiCacheAdmitBudget,
)
from sglang.srt.managers.hisparse_protocol import (
    HiSparseEvictionHooks,
    HiSparseTokenStats,
)
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.schedule_policy import remaining_max_new
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.memory_pool_host import HostPoolGroup
from sglang.srt.mem_cache.sparsity.factory import (
    HiSparseBacking,
    hisparse_indexer_regions,
)
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.runtime_context import get_schedule
from sglang.srt.utils import get_device_module

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool

device_module = get_device_module()

logger = logging.getLogger(__name__)

# How many times a deferred admission is retried before the request is left to
# run standard for good. Each attempt evicts for real, so an unbounded retry
# would keep paying device-to-host traffic for a request the pool has no room
# for; a couple of tries is enough to cover a write-back that acks late.
_MAX_ADMISSION_ATTEMPTS = 3


class _EvictionEvent(msgspec.Struct):
    """One node eviction, snapshotted for `_sync_evictions` to apply.

    `sorted_dev` / `sorted_host` are the node's device and host index arrays
    sorted together, so a request row matches against them with a searchsorted.
    `sorted_host` is None when the node was dropped without a host copy -- an
    invariant violation for any position backing an active request, reported as
    one.

    The node is held under one temporary host lock until the event is applied.
    `lock_params` replays the acquire-time skip set at release: components without
    a host_value (MAMBA) skipped the increment, so the decrement must skip them
    too or their host_lock_ref underflows.
    """

    sorted_dev: torch.Tensor
    sorted_host: Optional[torch.Tensor]
    node: object
    lock_params: object


class _PendingAdmission(msgspec.Struct):
    """A request whose admission was deferred out of the prefill-result pass."""

    req: Req
    # Attempts spent so far; each one evicts for real, hence the cap.
    attempts: int = 0


class _AdmittedNodes(msgspec.Struct):
    """The tree nodes an admitted request holds, released together at finish.

    `matched` is its whole prefix path, claimed at admission so the ledger bills
    the residency once per node. `host_locks` are the nodes evicted on its behalf
    afterwards, each with the params to replay at release. Two lists, one
    lifetime, one dict entry -- the ledger holds the token counts, this holds the
    objects.
    """

    matched: tuple
    host_locks: list = []


class _ExpandedIndexerPages:
    """Free list over the indexer buffer's expanded region.

    Page ids are region-local; the coordinator adds the region offset before
    writing them into a page table. A plain list is enough: allocation happens
    once per admitted request, of `tree_len // page_size` pages.
    """

    def __init__(self, *, num_pages: int, device: str):
        self._free = list(range(num_pages))
        self._device = device

    def alloc(self, num_pages: int) -> Optional[torch.Tensor]:
        if len(self._free) < num_pages:
            return None
        pages = self._free[-num_pages:]
        del self._free[-num_pages:]
        return torch.tensor(pages, dtype=torch.int32, device=self._device)

    def free(self, pages: torch.Tensor) -> None:
        self._free.extend(pages.tolist())

    def available(self) -> int:
        return len(self._free)


class HiCacheHiSparseCoordinator:
    """`HiSparseCoordinator` over the radix tree plus the HiCache host tier.

    Implements `managers/hisparse_protocol.py`. The entry points below the
    protocol (`on_node_evicted`, `node_backs_active_request`) are the tree cache's
    hooks; it reaches them through the concrete class after checking `backing`.
    """

    backing = HiSparseBacking.HICACHE

    def __init__(
        self,
        *,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator,
        top_k: int,
        device_buffer_size: int,
        device: str,
        tp_group,
        swap_in_block_size: int = 960,
    ):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.token_to_kv_pool: DSATokenToKVPool = (
            token_to_kv_pool_allocator.get_kvcache()
        )
        self.top_k = top_k
        self.device = device
        self.tp_group = tp_group
        self.tp_world_size = torch.distributed.get_world_size(group=self.tp_group)
        self.swap_in_block_size = swap_in_block_size
        # Timing probe: skip the host->device KV bytes to measure the "IO is free"
        # floor. Produces garbage output; benchmarking only.
        self.skip_io = envs.SGLANG_DEBUG_HISPARSE_SKIP_IO.get()

        self.page_size = self.token_to_kv_pool.page_size
        self.layer_num = self.token_to_kv_pool.layer_num
        self.start_layer = self.token_to_kv_pool.start_layer
        # The per-request device buffer, in tokens. Whole pages, because it is
        # allocated from the paged pool; at least top_k, because one decode step
        # can need every selected position swapped in at once.
        self.device_buffer_size = device_buffer_size
        assert (
            device_buffer_size >= top_k and device_buffer_size % self.page_size == 0
        ), (
            f"HiSparse on the HiCache backing needs device_buffer_size "
            f"({device_buffer_size}) >= top_k ({top_k}) and a multiple of "
            f"page_size ({self.page_size}): the buffer is allocated as whole "
            "pages and must hold one step's whole selection."
        )
        # The swap-in kernel tracks slots in an int16 LRU order.
        assert device_buffer_size <= 32767, (
            f"device_buffer_size ({device_buffer_size}) exceeds the int16 slot "
            "ordering the swap-in kernel keeps"
        )

        self._init_kv_views()
        self._init_request_state()
        self._init_expanded_indexer_region()

        self.tree_cache = None
        self._host_kv_cache = None
        self.decode_producer_stream = None
        self._eviction_queue: List[_EvictionEvent] = []
        self._announced_first_eviction = False
        # req_pool_idx -> admission deferred out of the prefill-result pass.
        self._pending_admission: Dict[int, _PendingAdmission] = {}
        # req_pool_idx -> tree nodes an admitted request holds. See _AdmittedNodes.
        self._req_nodes: Dict[int, _AdmittedNodes] = {}
        self.ledger = AdmissionLedger(
            # The ALLOCATOR's size, not the pool's: it is what hands out tokens,
            # and the two differ under attention DCP (the allocator covers
            # dcp_size shards of the pool).
            device_pool_tokens=int(self.token_to_kv_pool_allocator.size),
            temp_slot_tokens=device_buffer_size,
            page_size=self.page_size,
            # -1 when chunked prefill is disabled; the ledger reads that as "no
            # chunk reserve", which is right -- an unchunked prefill's whole
            # prompt is charged as pending instead.
            chunk_tokens=get_schedule().chunked_prefill_size,
        )

    def _init_kv_views(self) -> None:
        """Byte views and layout constants the swap-in kernel addresses through.

        KV rows hold mixed content (fp8 payload, scales, rope bytes), so the
        swap-in copy has to be bit-exact: a uint8 view gives byte-sized
        addressing without reinterpreting anything.
        """
        kv_u8 = [buf.view(torch.uint8) for buf in self.token_to_kv_pool.kv_buffer]
        self.item_size_bytes = kv_u8[0].stride(0)
        strides = {buf.stride(0) for buf in kv_u8}
        assert len(strides) == 1, f"KV layers must share a row stride, got {strides}"
        # [layer, 2] = the host pool's [base address, row stride in bytes], filled
        # when the host tier attaches (set_tree_cache). Persistent and read on the
        # device at launch time: cuda graphs are captured BEFORE attachment, so a
        # kernel argument would bake in a stale address. One row per layer because
        # each layer is a separate host allocation; the row's address is fixed at
        # capture, its contents are not.
        self._host_binding = torch.zeros(
            (self.layer_num, 2), dtype=torch.int64, device=self.device
        )

    def _init_request_state(self) -> None:
        """Per-request swap-in state, all persistent for cuda-graph replay."""
        max_num_req_slots = self.req_to_token_pool.req_to_token.shape[0]
        max_context_len = self.req_to_token_pool.max_context_len

        # The request's temp device buffer: pool rows it owns for swapped-in KV.
        # int32 because it is the kernel's `device_buffer_locs`; one row for all
        # layers, since a slot names the same pool row in every layer (the kernel
        # takes one row stride for this and the residency map below, so the two
        # must stay the same width).
        self.req_device_buffer_locs = torch.full(
            (max_num_req_slots, self.device_buffer_size),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        # Which position each buffer slot currently holds (-1 = empty), per layer:
        # every layer's indexer selects its own top-k, so residency diverges.
        self.req_device_buffer_tokens = torch.full(
            (self.layer_num, max_num_req_slots, self.device_buffer_size),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        # LRU order over the buffer slots, per layer. The kernel rewrites it in
        # place: evictables at the front, this step's hits at the back.
        self._lru_init = torch.arange(
            self.device_buffer_size, dtype=torch.int16, device=self.device
        )
        self.lru_slots = (
            self._lru_init.view(1, 1, -1)
            .repeat(self.layer_num, max_num_req_slots, 1)
            .contiguous()
        )
        # Host row per (request, position), -1 = no host copy. Same shape, dtype
        # and sentinel as the private-host backing's table: it is the swap-in
        # kernel's `host_cache_locs` ABI, not a private choice.
        self.req_to_host_pool = torch.full(
            (max_num_req_slots, max_context_len),
            -1,
            dtype=torch.int64,
            device=self.device,
        )
        # Pre-allocated swap-in output (cuda-graph safe).
        self.top_k_device_locs_buffer = torch.full(
            (max_num_req_slots, self.top_k), -1, dtype=torch.int32, device=self.device
        )
        # Number of real (non-padded) requests in the batch, as a device scalar so
        # padded blocks early-return on replay.
        self.num_real_reqs = torch.zeros(1, dtype=torch.int32, device=self.device)

    def _init_expanded_indexer_region(self) -> None:
        """Carve the indexer buffer and set up the per-request page table.

        The buffer covers more tokens than the attention pool holds (see
        `hisparse_indexer_expansion_ratio`). Its base region is co-addressed with
        the pool -- a KV page id IS its indexer page id -- and the expanded region
        is handed out per request, because a page the tree evicts goes back to the
        allocator and the next request writes over it, indexer rows included.
        """
        indexer_bufs = self.token_to_kv_pool.index_k_with_scale_buffer
        base_pages, expanded_pages = hisparse_indexer_regions(
            page_size=self.page_size,
            num_indexer_pages=len(indexer_bufs[0]),
            device_pool_size=self.token_to_kv_pool.size,
        )
        self._indexer_bufs = indexer_bufs
        # Expanded page ids start where the base region ends.
        self._indexer_page_offset = base_pages
        self._indexer_pages = _ExpandedIndexerPages(
            num_pages=expanded_pages, device=self.device
        )
        max_pages_per_req = (
            self.req_to_token_pool.max_context_len + self.page_size - 1
        ) // self.page_size
        # Per-request expanded page per prefix page (-1 = the original page is
        # still valid).
        self.req_to_indexer_page = torch.full(
            (self.req_to_token_pool.req_to_token.shape[0], max_pages_per_req),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        logger.info(
            "HiSparse: indexer buffer holds %d pages, %d taken by the attention "
            "pool, %d expanded (%.1f MB/layer)",
            base_pages + expanded_pages,
            base_pages,
            expanded_pages,
            expanded_pages * indexer_bufs[0][0].nbytes / 1e6,
        )

    # ------------------------------------------------------------------
    # Setup / teardown
    # ------------------------------------------------------------------

    def set_tree_cache(self, tree_cache) -> None:
        """Attach the tree cache, register the eviction hooks, publish the host
        pool's address so captured swap-ins can reach it."""
        self.tree_cache = tree_cache
        self._host_kv_cache = None
        if tree_cache is None:
            return

        assert isinstance(tree_cache, UnifiedRadixCache), (
            "HiSparse on the HiCache backing needs the unified radix cache (it is "
            "what carries the host tier and the device-eviction hooks), got "
            f"{type(tree_cache).__name__}"
        )
        tree_cache.set_hisparse_eviction_hooks(
            HiSparseEvictionHooks(
                on_device_released=self.on_node_evicted,
                backs_live_request=self.node_backs_active_request,
            )
        )
        host_pool = tree_cache.cache_controller.mem_pool_host
        if isinstance(host_pool, HostPoolGroup):
            host_pool = host_pool.get_pool(PoolName.KV)
        self._host_kv_cache = host_pool
        self.ledger.set_host_capacity(int(host_pool.size))
        self._publish_host_binding(host_pool)

    def _publish_host_binding(self, host_pool) -> None:
        """Fill the per-layer [base address, row stride] the kernel reads.

        `data_refs` is the host pool's per-layer view of its buffer, which for the
        page_first layout is a transpose of [token, layer, cell]: its rows are
        layer_num cells apart, not one. Both the base and that stride have to
        reach the kernel through device memory, since the graph that reads them
        was captured before this pool existed.
        """
        refs = [ref.view(torch.uint8) for ref in host_pool.data_refs]
        assert len(refs) >= self.layer_num, (
            f"host pool exposes {len(refs)} layers, fewer than the device pool's "
            f"{self.layer_num}"
        )
        strides = {ref.stride(0) for ref in refs[: self.layer_num]}
        assert len(strides) == 1, f"inconsistent host row strides: {strides}"
        row_bytes = strides.pop()
        # The kernel copies exactly item_size_bytes at each host row offset, so a
        # host cell that is not the device row is a silent overread or a partial
        # write, and a row stride off a cell boundary corrupts every fetch.
        assert host_pool.token_stride_size == self.item_size_bytes, (
            f"host KV cell is {host_pool.token_stride_size} bytes but the device "
            f"row is {self.item_size_bytes}"
        )
        assert row_bytes % self.item_size_bytes == 0, (
            f"host row stride {row_bytes} is not a multiple of the KV cell "
            f"{self.item_size_bytes}"
        )
        self._host_binding.copy_(
            torch.tensor(
                [[ref.data_ptr(), row_bytes] for ref in refs[: self.layer_num]],
                dtype=torch.int64,
            )
        )
        logger.info(
            "HiSparse: host pool attached, %d layers, row stride %d bytes over a "
            "%d-byte KV cell",
            self.layer_num,
            row_bytes,
            self.item_size_bytes,
        )

    def set_decode_producer_stream(self, stream) -> None:
        self.decode_producer_stream = stream

    def destroy(self) -> None:
        """Drop the host addresses. HiCache owns and tears down the pool itself;
        what must not survive it is this coordinator's non-owning pointers, which
        a late swap-in would otherwise dereference."""
        self._host_binding.zero_()

    # ------------------------------------------------------------------
    # Request lifecycle
    # ------------------------------------------------------------------

    def on_prefill_complete(self, req: Req) -> bool:
        """Take over the request's prefix, or decline and let it run standard.

        Only the tree-owned page-aligned prefix `[0, cache_protected_len)` can
        ever be evicted, so only those pages need expanded indexer copies; the
        unaligned prefill tail and every decode token are request-owned and stay
        in regular pool pages.

        Declining is a normal outcome (nothing evictable, a quota spent), and it
        costs nothing: the request keeps its KV in the pool and its tree lock.
        """
        req_pool_idx = req.req_pool_idx
        # At most ONCE per active slot, an invariant the CALLER holds up (last
        # chunk only, and retraction tears down first). Asserted rather than
        # trusted because a second call is silent: it takes fresh pages, buffer
        # and claims, overwrites the tables holding the previous ones, and leaks
        # them for the server's lifetime.
        assert req_pool_idx not in self.ledger.active_reqs, (
            "HiSparse: on_prefill_complete re-entered for an already-admitted "
            f"request slot {req_pool_idx} (recorded "
            f"{self.ledger.active_reqs.get(req_pool_idx)}, now tree_len "
            f"{req.cache_protected_len}). The caller must finalize a slot "
            "(request_finished / retract_req) before re-admitting it."
        )
        tree_len = req.cache_protected_len

        num_pages = tree_len // self.page_size
        if num_pages == 0:
            # Nothing page-aligned reached the tree, so nothing is evictable.
            self.ledger.drop_pending(req.rid)
            return False
        if tree_len < self.device_buffer_size:
            # Net capacity LOSS: the temp buffer is pinned for the request's
            # lifetime while only tree_len becomes evictable, which would exceed
            # the scheduler's standard-path reservation (retract/OOM on long
            # outputs). Such a short prefix gains nothing from eviction anyway.
            self.ledger.drop_pending(req.rid)
            return False

        # Eligible, but NOT admitted here: everything below allocates, and this
        # runs inside the prefill-result pass, whose batched frees hide the pages
        # an eviction would hand back (measured: `avail 384 -> 384` with a device
        # leaf and 20480 evictable tokens right there). `admit_pending` does the
        # allocation once that pass has flushed. The request needs nothing in the
        # meantime -- it is an ordinary request until then, and the deferral is
        # resolved before its first decode forward.
        self._pending_admission[req_pool_idx] = _PendingAdmission(req=req)
        return True

    def admit_pending(self) -> None:
        """Admit the deferred requests. See the protocol for why it is separate."""
        if not self._pending_admission:
            return
        # The whole point of deferring. If a future change wraps this call in a
        # free group the way #33475 wrapped on_prefill_complete, admission would
        # silently stop working again (concurrency quietly drops to one and
        # throughput falls back to the dense baseline), so refuse to run instead.
        assert self.token_to_kv_pool_allocator.is_not_in_free_group, (
            "HiSparse: admit_pending must run outside the allocator's free "
            "group; inside one, the pages an eviction frees are parked where "
            "alloc() cannot see them and every admission fails."
        )
        for req_pool_idx, pending in list(self._pending_admission.items()):
            if self._try_admit(pending.req):
                self._pending_admission.pop(req_pool_idx, None)
                continue
            pending.attempts += 1
            if pending.attempts >= _MAX_ADMISSION_ATTEMPTS:
                # Give up: each attempt evicts for real, so retrying forever
                # would keep paying device-to-host traffic for a request the
                # pool has no room for.
                self._cancel_pending(req_pool_idx, rid=pending.req.rid)

    def _try_admit(self, req: Req) -> bool:
        """One admission attempt; False leaves the request exactly as it was."""
        req_pool_idx = req.req_pool_idx
        tree_len = req.cache_protected_len
        num_pages = tree_len // self.page_size

        # No host gate here: the request is already in the tree, so a match-based
        # check would compare against its own freshly inserted prefix. Host space
        # is enforced where it can be acted on -- at prefill admission
        # (HiCacheAdmitBudget) and at eviction, where the copy-less-drop veto
        # keeps a full host tier from dropping data an active request needs.

        # Apply pending evictions BEFORE activating this request: queued events
        # predate it, and its freshly allocated prefill indices may collide with
        # device indices those events freed.
        self._sync_evictions()

        pages = self._indexer_pages.alloc(num_pages)
        if pages is None:
            logger.warning(
                "HiSparse: expanded indexer alloc failed for req_pool_idx=%d "
                "(need %d pages, %d available); running standard",
                req_pool_idx,
                num_pages,
                self._indexer_pages.available(),
            )
            return False

        temp_slots, before, after = self._alloc_temp_buffer()
        if temp_slots is None:
            # Not an edge case: under load the pool sits at its eviction
            # low-water mark, so this is the normal retry path. Each number below
            # separates one cause from another -- avail unchanged with no
            # evictable leaf means a write-back has not landed yet (the retry
            # covers it); avail up but short means the eviction under-delivered;
            # free_group_open means a broken contract.
            allocator = self.token_to_kv_pool_allocator
            leaves = self.tree_cache.tree_core.evictable_device_leaves
            logger.warning(
                "HiSparse: temp buffer alloc failed for req_pool_idx=%d "
                "(%d tokens; avail %d -> %d, evictable=%d protected=%d, "
                "%d device leaves, free_group_open=%s, %d admitted); "
                "running standard",
                req_pool_idx,
                self.device_buffer_size,
                before,
                after,
                self.tree_cache.full_evictable_size(),
                self.tree_cache.full_protected_size(),
                len(leaves),
                not allocator.is_not_in_free_group,
                len(self.ledger.active_reqs),
            )
            self._indexer_pages.free(pages)
            return False

        self._copy_indexer_to_expanded_pages(
            req_pool_idx=req_pool_idx, tree_len=tree_len, pages=pages
        )
        self._reset_request_state(req_pool_idx)
        self.req_device_buffer_locs[req_pool_idx] = temp_slots.to(torch.int32)

        # Claim the whole tree-resident prefix: it must keep ONE home (device or
        # host) for the request's lifetime. Deduped per node, so requests sharing
        # a prefix bill it once, and a re-hit prefix is billed even though it
        # added nothing new to the tree.
        prefix_nodes = self._prefix_path_nodes(req)
        for node in prefix_nodes:
            self.ledger.claim_node(node)
        self._req_nodes[req_pool_idx] = _AdmittedNodes(matched=tuple(prefix_nodes))
        # After the claims, so the ledger goes straight from billing this prefix
        # as in-flight to billing it as claimed, never both (activate drops the
        # pending charge) and never neither.
        self.ledger.activate(
            req_pool_idx,
            tree_len,
            rid=req.rid,
            # Device pages this request will still take for its own output, on
            # the same basis the adder budgets with.
            decode_reserve=remaining_max_new(req),
        )

        # Last, once every allocation has succeeded: this is what makes the
        # prefix evictable, and it cannot be taken back.
        self._release_lock_for_eviction(req)
        # The eviction this admission just triggered queued its own events; the
        # forward must not launch with rows that still point at pages the temp
        # buffer above may already own.
        self._sync_evictions()
        return True

    def _alloc_temp_buffer(self):
        """Reserve the request's private device buffer out of the regular pool.

        Returns (slots, available before the eviction, available after it): the
        caller reports the pair on failure, because "the eviction freed nothing"
        and "the eviction freed some but not enough" are different bugs.
        """
        from sglang.srt.mem_cache.common import evict_from_tree_cache

        allocator = self.token_to_kv_pool_allocator
        before = allocator.available_size()
        evict_from_tree_cache(self.tree_cache, self.device_buffer_size)
        after = allocator.available_size()
        return allocator.alloc(self.device_buffer_size), before, after

    def _copy_indexer_to_expanded_pages(
        self, *, req_pool_idx: int, tree_len: int, pages: torch.Tensor
    ) -> None:
        """Copy the prefix's indexer rows into the request's private pages.

        A KV page id doubles as its base-region indexer page id, so the source is
        just the prefix's page ids. After this the indexer can score the prefix
        from GPU whatever the tree does with its attention KV.
        """
        token_indices = self.req_to_token_pool.req_to_token[req_pool_idx, :tree_len]
        orig_pages = (token_indices[:: self.page_size] // self.page_size).long()
        expanded = (pages + self._indexer_page_offset).long()
        for indexer_buf in self._indexer_bufs:
            indexer_buf[expanded] = indexer_buf[orig_pages]
        self.req_to_indexer_page[req_pool_idx, : pages.numel()] = (
            pages + self._indexer_page_offset
        )

    def _reset_request_state(self, req_pool_idx: int) -> None:
        """Clear whatever the slot's previous tenant left in the swap-in tables.

        The attention backend hands the swap-in the WHOLE batch's
        req_pool_indices, so a stale residency row would let this request "hit" a
        slot it does not own and have that top-k position masked.
        """
        self.req_device_buffer_tokens[:, req_pool_idx, :] = -1
        self.lru_slots[:, req_pool_idx, :].copy_(self._lru_init)
        self.req_to_host_pool[req_pool_idx] = -1

    def request_finished(self, req: Req) -> None:
        # Apply pending evictions while this request is still active: the
        # prefix-intact check in cache_finished_req (which runs right after) must
        # see up-to-date sentinels.
        self._sync_evictions()
        # Both stages, in order, because a request can finish in either: a
        # deferred admission that never ran, and the charge it was budgeted at
        # prefill entry (released here or it starves host headroom for the
        # server's lifetime).
        self._cancel_pending(req.req_pool_idx, rid=req.rid)
        if req.req_pool_idx in self.ledger.active_reqs:
            self._release_active(req.req_pool_idx)

    def _cancel_pending(self, req_pool_idx: int, *, rid: str) -> None:
        """Drop a deferred admission and the prefill-time charge behind it.

        The two are one step everywhere: a request whose admission never
        completes must hold neither, and the charge is what the adder's ceiling
        counts.
        """
        self._pending_admission.pop(req_pool_idx, None)
        self.ledger.drop_pending(rid)

    def _release_active(self, req_pool_idx: int) -> None:
        """Give back everything an admitted request holds, in one place.

        Ordered: the fence first, then accounting, then the device resources the
        forward may still be reading.
        """
        # After the potentially overlapped forward: under overlap scheduling the
        # batch launched one step ahead still contains this request, and its
        # kernels read req_to_indexer_page, req_to_host_pool and the temp buffer
        # freed below. (_sync_evictions in the caller only waits when the eviction
        # queue is non-empty.)
        if self.decode_producer_stream is not None:
            device_module.current_stream().wait_stream(self.decode_producer_stream)

        self.ledger.deactivate(req_pool_idx)
        nodes = self._req_nodes.pop(req_pool_idx, None)
        if nodes is not None:
            # The host locks taken at eviction time on this request's behalf...
            for node, dec_params in nodes.host_locks:
                self.tree_cache.dec_host_lock_ref(node.id, dec_params)
                self.ledger.release_node(node)
            # ... and the matched-path claims taken at admission.
            for node in nodes.matched:
                self.ledger.release_node(node)

        # The whole temp buffer, always fully allocated at admission. The
        # int32 -> int64 conversion materializes a copy, so overwriting the row
        # with -1 below cannot reach what free() holds (free() also copies for
        # itself when deferring into a free group).
        temp_slots = self.req_device_buffer_locs[req_pool_idx].to(torch.int64)
        self.token_to_kv_pool_allocator.free(temp_slots)
        self.req_device_buffer_locs[req_pool_idx] = -1
        self._reset_request_state(req_pool_idx)

        pages = self.req_to_indexer_page[req_pool_idx]
        allocated = pages[pages >= 0]
        if allocated.numel() > 0:
            self._indexer_pages.free(allocated - self._indexer_page_offset)
        self.req_to_indexer_page[req_pool_idx] = -1

    def retract_req(self, req: Req) -> None:
        """Retraction and completion tear down the same state, at every stage:
        `request_finished` drops a deferred admission and a pending charge before
        it looks at `active_reqs`, so a request retracted between the deferral and
        the admission leaves nothing behind either."""
        self.request_finished(req)

    # ------------------------------------------------------------------
    # Protocol members with nothing to do on this backing
    # ------------------------------------------------------------------

    def on_prefill_finished_early(self, req: Req) -> None:
        """Settle the prefill-time claim of a request that never reaches
        on_prefill_complete (max_new == 0, or EOS on the first token)."""
        self._cancel_pending(req.req_pool_idx, rid=req.rid)

    def collect_ready_reqs(self) -> List[Req]:
        """Nothing is ever waiting here: a deferred request is runnable while it
        waits, because it runs as an ordinary request until `admit_pending` takes
        its KV over. Only a staging backing has requests that cannot run yet."""
        return []

    def has_ongoing_staging(self) -> bool:
        return False

    def wait_for_pending_backup(self) -> None:
        """Device-to-host copies belong to HiCache's cache controller, which
        orders them against the tree operations that depend on them."""

    # ------------------------------------------------------------------
    # Data plane
    # ------------------------------------------------------------------

    def indexer_page_table(
        self, *, req_pool_indices: torch.Tensor, num_pages: int
    ) -> Optional[torch.Tensor]:
        """Build the hybrid indexer page table (vectorized, no host syncs).

        Per page slot, keyed on the page's first `req_to_token` entry: `>= 0` ->
        the original page id (same page in the KV buffer and the indexer's base
        region); `-1`, an evicted prefix page -> the request's private expanded
        page, copied at admission and GPU-resident for its lifetime.

        Expanded ids fall outside the KV buffer's page range, so the result is
        only valid for indexer scoring -- attention resolves its own positions
        through `swap_in_selected_pages`. Rows of requests this coordinator never
        admitted come out bit-identical to the standard table, so a mixed batch
        can substitute the whole thing. None while nothing is admitted.
        """
        if not self.ledger.active_reqs:
            return None

        rows = req_pool_indices.long()
        firsts = self.req_to_token_pool.req_to_token[
            rows, : num_pages * self.page_size : self.page_size
        ]
        # A strided slice past the row's end silently returns fewer columns, and
        # the caller writes this table into a fixed-width buffer.
        assert firsts.shape[1] == num_pages, (
            f"{num_pages} indexer pages asked for, but req_to_token only covers "
            f"{firsts.shape[1]} pages of {self.page_size} tokens"
        )
        orig_pages = firsts // self.page_size
        expanded = self.req_to_indexer_page[rows, :num_pages]
        table = torch.where((firsts < 0) & (expanded >= 0), expanded, orig_pages)
        # A negative id would read out of bounds: a sentinelled position with no
        # expanded page is a broken invariant, but not one to chase into the
        # indexer's address arithmetic.
        return table.clamp_(min=0).to(torch.int32)

    def translate_page_table(self, page_table: torch.Tensor) -> torch.Tensor:
        """Identity: attention KV stays in the regular pool, so a logical page id
        already addresses it, and `req_to_token` is the int32 the sparse kernels
        want."""
        return page_table

    def swap_in_selected_pages(
        self,
        *,
        req_pool_indices: torch.Tensor,
        compressed_seq_lens: torch.Tensor,
        top_k_result: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        """Make one layer's selected KV device-resident; return its slot table.

        Three sources per selected position, resolved by the kernel: still in the
        pool (`req_to_token >= 0`, which covers every position of every request
        this coordinator did not admit) passes through with no copy; an
        already-resident temp slot is reused; anything else is DMA'd from host into
        a slot the kernel evicts by LRU. Only persistent tensors are read, so this
        is overlap- and graph-safe.

        `compressed_seq_lens` goes unused: residency here is per position, not per
        length, and the kernel's length-driven shortcuts are compiled out by the
        pool source.
        """
        num_reqs = req_pool_indices.size(0)
        top_k_device_locs = self.top_k_device_locs_buffer[:num_reqs]
        local_layer = layer_id - self.start_layer
        assert 0 <= local_layer < self.layer_num, (
            f"layer {layer_id} maps to local layer {local_layer}, outside "
            f"[0, {self.layer_num})"
        )
        load_cache_to_device_buffer_mla(
            top_k_tokens=top_k_result,
            device_buffer_tokens=self.req_device_buffer_tokens[local_layer],
            host_cache_locs=self.req_to_host_pool,
            device_buffer_locs=self.req_device_buffer_locs,
            # The host source travels through host_binding instead: this pool is
            # attached after cuda-graph capture, and its rows are not one KV cell
            # apart under the page_first layout.
            host_cache=None,
            device_buffer=self.token_to_kv_pool.kv_buffer[local_layer],
            top_k_device_locs=top_k_device_locs,
            req_pool_indices=req_pool_indices,
            seq_lens=compressed_seq_lens,
            lru_slots=self.lru_slots[local_layer],
            item_size_bytes=self.item_size_bytes,
            num_top_k=self.top_k,
            hot_buffer_size=self.device_buffer_size,
            page_size=1,
            block_size=self.swap_in_block_size,
            num_real_reqs=self.num_real_reqs,
            skip_io=self.skip_io,
            device_locs=self.req_to_token_pool.req_to_token,
            host_binding=self._host_binding[local_layer],
        )
        return top_k_device_locs

    def prepare_decode_batch(
        self,
        *,
        seq_lens: torch.Tensor,
        out_cache_loc: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
    ) -> None:
        """Apply queued eviction events before the next forward launches.

        Nothing to route into a buffer: decode tokens live in pool pages, which
        are never evicted. What MUST happen is the eviction sync -- an evicted page
        goes straight back to the allocator, so the next forward must not launch
        with a `req_to_token` entry still pointing at it. This is the last
        controlled point between two forwards, and also the retry point for an
        admission whose device buffer only frees up once a write-back is acked.
        """
        self._sync_evictions()
        self.admit_pending()

    # ------------------------------------------------------------------
    # Eviction handling
    # ------------------------------------------------------------------

    def on_node_evicted(self, node) -> None:
        """Tree-cache hook: this node's device KV is about to be released.

        Contract the caller owes: device indices still in
        `component_data[FULL].value`, the host copy (if any) in `.host_value`, and
        the device indices not yet handed back to the allocator.

        Runs on the scheduler thread and only SNAPSHOTS those arrays -- no GPU
        writes, so it is safe against an in-flight forward. `_sync_evictions`
        matches them against request rows later, in FIFO order: a device index
        freed, reused and evicted again cannot mis-attribute host indices, because
        the first event already sentinelled the old row positions.
        """
        if not self.ledger.active_reqs:
            return

        cd = node.component_data[ComponentType.FULL]
        device_values = cd.value
        if device_values is None or len(device_values) == 0:
            return
        device_values = device_values.to(device=self.device, dtype=torch.int64)
        host_values = cd.host_value
        if host_values is not None:
            host_values = host_values.to(device=self.device, dtype=torch.int64)

        sorted_dev, order = torch.sort(device_values)
        sorted_host = host_values[order] if host_values is not None else None

        # One TEMPORARY host lock keeps host_value alive until the event is
        # applied. It is taken here and not at admission because under write_back
        # host_value only exists after the first device eviction, and the lock
        # silently no-ops without it. _sync_evictions attributes ownership --
        # requests whose rows actually hold these indices get their own lock, then
        # this one is dropped -- so an unrelated churn node is never pinned for the
        # lifetime of a long decode.
        locked_node = None
        lock_params = None
        if sorted_host is not None:
            lock_params = self.tree_cache.inc_host_lock_ref(node.id).to_dec_params()
            locked_node = node
        self._eviction_queue.append(
            _EvictionEvent(
                sorted_dev=sorted_dev,
                sorted_host=sorted_host,
                node=locked_node,
                lock_params=lock_params,
            )
        )

    def node_backs_active_request(self, node) -> bool:
        """Tree-cache hook: whether dropping this node would mask live positions.

        The write-back drop fallback (host full, backup failed) asks before
        dropping a node's KV without a host copy. For a node backing an admitted
        request that drop is unrecoverable -- the position has no home left --
        so the tree keeps it on device instead. Rare path, so the active-row
        concat is rebuilt per call.
        """
        active = self.ledger.active_reqs
        if not active:
            return False
        device_values = node.component_data[ComponentType.FULL].value
        if device_values is None or len(device_values) == 0:
            return False
        rows = torch.cat(
            [
                self.req_to_token_pool.req_to_token[
                    req_pool_idx, : state.tree_len
                ].long()
                for req_pool_idx, state in active.items()
            ]
        )
        rows = rows[rows >= 0]
        if rows.numel() == 0:
            return False
        rows, _ = torch.sort(rows)
        values = device_values.to(device=rows.device, dtype=torch.int64)
        # Sort once, then binary-search the node's values into it. `torch.isin`
        # reads better and needs no sorted input, but measured 2.3x slower here
        # (433 us vs 185 us at 4 admitted prefixes against a 512-token node): it
        # redoes this sort internally, over the larger operand.
        pos = torch.searchsorted(rows, values).clamp_(max=rows.numel() - 1)
        return bool((rows[pos] == values).any().item())

    def _sync_evictions(self) -> None:
        """Apply queued evictions: sentinel `req_to_token`, record the host rows.

        Waits on the forward stream first, so an in-flight forward never observes
        a sentinel without its host index (or loses a device index it is still
        reading). Runs before the next forward launches, whose own
        wait_stream makes these writes visible to it.
        """
        if not self._eviction_queue:
            return

        if self.decode_producer_stream is not None:
            device_module.current_stream().wait_stream(self.decode_producer_stream)

        # Per (event, request) hit counts, read back in one batch below: a
        # per-pair .item() would sync once per pair.
        attributions: List[tuple] = []
        hit_counts: List[torch.Tensor] = []
        dropped: List[tuple] = []
        for event_idx, event in enumerate(self._eviction_queue):
            num_indices = len(event.sorted_dev)
            for req_pool_idx, state in self.ledger.active_reqs.items():
                tree_len = state.tree_len
                row = self.req_to_token_pool.req_to_token[req_pool_idx, :tree_len]
                row64 = row.long()
                pos = torch.searchsorted(event.sorted_dev, row64).clamp_(
                    max=num_indices - 1
                )
                match = (event.sorted_dev[pos] == row64) & (row64 >= 0)
                if event.sorted_host is not None:
                    host_row = self.req_to_host_pool[req_pool_idx, :tree_len]
                    host_row.copy_(torch.where(match, event.sorted_host[pos], host_row))
                else:
                    dropped.append((req_pool_idx, match.sum()))
                row.masked_fill_(match, -1)
                if event.node is not None:
                    attributions.append((event_idx, req_pool_idx))
                    hit_counts.append(match.sum())

        self._report_dropped_without_host(dropped)
        self._attribute_host_locks(attributions, hit_counts)
        for event in self._eviction_queue:
            if event.node is not None:
                self.tree_cache.dec_host_lock_ref(event.node.id, event.lock_params)
        self._eviction_queue.clear()

    @staticmethod
    def _report_dropped_without_host(dropped: List[tuple]) -> None:
        """Fail loud on a position that ended up with no home at all.

        The admission ceiling plus the copy-less-drop veto are supposed to make
        this unreachable. Continuing would mask those positions in attention and
        return silently degraded output, which is worse than a crash: nothing
        downstream can tell that answer from a good one.
        """
        if not dropped:
            return
        counts = torch.stack([count for _, count in dropped]).cpu().tolist()
        for (req_pool_idx, _), count in zip(dropped, counts):
            if count > 0:
                raise RuntimeError(
                    f"HiSparse ACCURACY LOSS: {count} active positions of "
                    f"req_pool_idx={req_pool_idx} were evicted without a host "
                    "copy, and would be masked in attention. The admission gate / "
                    "drop veto invariant is broken."
                )

    def _attribute_host_locks(
        self, attributions: List[tuple], hit_counts: List[torch.Tensor]
    ) -> None:
        """Give each request its own host lock on the nodes it actually holds.

        One batched readback (on eviction steps only), then a lock per real hit;
        the events' temporary locks are dropped by the caller right after.
        """
        if not hit_counts:
            return
        counts = torch.stack(hit_counts).cpu().tolist()
        for (event_idx, req_pool_idx), hits in zip(attributions, counts):
            if not hits or req_pool_idx not in self.ledger.active_reqs:
                continue
            if not self._announced_first_eviction:
                # One-shot, because the host source is otherwise unobservable: a
                # run where the tree never evicted a live prefix exercises only
                # the pass-through path, and looks exactly like a run where the
                # swap-in works.
                self._announced_first_eviction = True
                logger.info(
                    "HiSparse: first eviction of an admitted prefix -- %d "
                    "positions of req_pool_idx=%d are now host-resident, and "
                    "decode will fetch them back per step.",
                    hits,
                    req_pool_idx,
                )
            self.ledger.note_evicted_positions(req_pool_idx, hits)
            node = self._eviction_queue[event_idx].node
            dec_params = self.tree_cache.inc_host_lock_ref(node.id).to_dec_params()
            self._req_nodes[req_pool_idx].host_locks.append((node, dec_params))
            self.ledger.claim_node(node, host_locked=True)

    # ------------------------------------------------------------------
    # Tree locking
    # ------------------------------------------------------------------

    def _prefix_path_nodes(self, req: Req) -> list:
        """The request's tree-resident prefix nodes, leaf first.

        Walks from `req.last_node` -- the node covering its cached prefix after
        the insert -- and NOT from `req.best_match_node`, which is a match-time id
        the tree may already have split or freed by the time admission runs.
        """
        if self.tree_cache is None or req.last_node is None:
            return []
        try:
            node = self.tree_cache.tree_core.node_by_id(req.last_node)
        except KeyError:
            return []
        nodes = []
        while node is not None and node.parent is not None:
            nodes.append(node)
            node = node.parent
        return nodes

    def _release_lock_for_eviction(self, req: Req) -> None:
        """Release the tree lock the insert took, so the prefix can be evicted.

        Host locks are NOT taken in its place: under write_back a node has no
        host_value until its first device eviction, and the host lock silently
        no-ops without one. They are taken in `on_node_evicted`, where the host
        copy is guaranteed live.
        """
        if self.tree_cache is None or req.last_node is None:
            return
        # The tree's own entry point, not a hand-rolled dec_lock_ref: it also
        # replays the components the request skipped locking, and releasing one it
        # never took underflows that component's ref count.
        self.tree_cache.dec_req_lock(req, skip_swa=req.swa_prefix_lock_released)
        # The scheduler's own release must not run a second time.
        req.hisparse_prefix_lock_released = True

    # ------------------------------------------------------------------
    # Scheduler hooks
    # ------------------------------------------------------------------

    def admit_budget(self) -> Optional[HiCacheAdmitBudget]:
        return self.ledger.make_budget(
            expanded_pages_left=self._indexer_pages.available(),
            tree_evictable_tokens=(
                0 if self.tree_cache is None else self.tree_cache.full_evictable_size()
            ),
        )

    def get_token_stats(self) -> HiSparseTokenStats:
        allocator = self.token_to_kv_pool_allocator
        device_capacity = allocator.size
        device_tokens = device_capacity - allocator.available_size()
        host_capacity = 0 if self._host_kv_cache is None else self._host_kv_cache.size
        host_tokens = (
            0
            if self._host_kv_cache is None
            else host_capacity - self._host_kv_cache.available_size()
        )
        return HiSparseTokenStats(
            device_tokens=device_tokens,
            device_token_usage=(
                device_tokens / device_capacity if device_capacity > 0 else 0.0
            ),
            host_tokens=host_tokens,
            host_token_usage=(
                host_tokens / host_capacity if host_capacity > 0 else 0.0
            ),
        )
