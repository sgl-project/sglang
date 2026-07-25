"""HiSparse V2 coordinator — separated indexer/attention KV eviction.

Indexer KV (index_k_with_scale_buffer) stays on GPU during decode: prefill
indexer KV is copied once into expanded pages at admission; decode-written
indexer KV lives in the regular pool pages, which are never evicted (the
radix tree only holds the page-aligned prefill prefix, and only tree nodes
are evictable). Attention KV is managed by HiCache and may be evicted to
host. On each decode step the indexer scores ALL tokens from GPU, selects
top-k, then the dual-source kernel swap-ins evicted attention KV for the
selected positions.

Flow:
  Prefill → both kv_buffer and indexer written to same page indices
  → admit: copy indexer KV to expanded pages, dec_lock_ref (allow eviction)
  → Decode: indexer reads hybrid page table (expanded for evicted prefill
            pages, original otherwise),
            swap_in loads evicted attention KV for top-k from host
  → Finish: free expanded indexer pages + temp slots

Overlap/CUDA-graph safety:
  The eviction callback runs on the scheduler thread. It computes match
  masks from req_to_token rows (reads only, async GPU ops) and enqueues
  them. _sync_evictions() applies the sentinel/host-loc writes after
  waiting on the forward stream, so in-flight forwards never observe a
  half-written eviction. All tensors read by captured kernels
  (req_to_token, host_locs, temp_slots) are persistent → CUDA-graph safe
  (fixed data_ptr, content updated before replay).

Eviction tracking is device-index based: the evicted node's device indices
are matched against each active request's req_to_token row, so it is
robust to radix node split/merge and to requests sharing prefix nodes.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.mem_cache.base_prefix_cache import DecLockRefParams
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool

logger = logging.getLogger(__name__)


class _EvictionEvent:
    """Record of one node eviction: sorted device indices + aligned host
    indices + the node itself (held under one temporary host lock until
    the event is applied and ownership is attributed)."""

    __slots__ = ("sorted_dev", "sorted_host", "node")

    def __init__(
        self,
        sorted_dev: torch.Tensor,
        sorted_host: Optional[torch.Tensor],
        node,
    ):
        self.sorted_dev = sorted_dev
        self.sorted_host = sorted_host
        self.node = node


class _SimplePageAllocator:
    """Lightweight page allocator for the expanded indexer buffer."""

    def __init__(self, num_pages: int, device: str):
        self._free = list(range(num_pages))
        self.device = device

    def alloc(self, n: int) -> Optional[torch.Tensor]:
        if len(self._free) < n:
            return None
        pages = self._free[-n:]
        del self._free[-n:]
        return torch.tensor(pages, dtype=torch.int32, device=self.device)

    def free(self, pages: torch.Tensor) -> None:
        self._free.extend(pages.tolist())

    def available(self) -> int:
        return len(self._free)


class HiSparseV2AdmitBudget:
    """Scheduler-side admission budget for one PrefillAdder round.

    Snapshots the coordinator quotas (see make_admit_budget) and PREDICTS
    admission outcomes so the PrefillAdder can pick each candidate's
    device reservation; the REAL accounting lives in the coordinator and
    is re-snapshotted into every new adder. Within one round the
    quotas act as shadow copies, depleted on commit.
    """

    def __init__(
        self,
        page_size: int,
        temp_slot_tokens: int,
        expanded_pages_left: int,
        host_tokens_left: int,
        device_pool_tokens: int,
    ):
        self._page_size = page_size
        self.temp_slot_tokens = temp_slot_tokens
        self._expanded_left = expanded_pages_left
        self._host_left = host_tokens_left
        self._device_pool_tokens = device_pool_tokens

    def _infeasible(self, total_seq_len: int, max_new: int) -> bool:
        """The candidate's V2 reservation cannot fit the device pool even
        at full idle → budget it as standard. The entry-side max_new clamp
        only guarantees the STANDARD budget fits the pool; reserving the
        temp device buffer on top could push add_one_req's total past
        rem_total_tokens forever (NO_TOKEN livelock on an idle system).
        Note the coordinator may still V2-admit such a request when the
        pool has room at admission time — that is strictly better than
        standard (the prefix becomes evictable), and the reservation here
        stays the conservative standard one."""
        return (
            total_seq_len + self.temp_slot_tokens + max_new + self._page_size
            >= self._device_pool_tokens
        )

    def future_tokens(self, total_seq_len: int, max_new: int, commit: bool) -> int:
        """Device-pool future-token reservation for a candidate.

        Predicted-admission success → ``temp_slot_tokens + max_new`` (the
        temp device buffer is allocated from the regular pool at admission
        and held for the request lifetime); predicted failure → the
        standard ``max_new``."""
        num_pages = total_seq_len // self._page_size
        tree_len = num_pages * self._page_size
        if (
            num_pages <= 0
            or num_pages > self._expanded_left
            or tree_len > self._host_left
            or self._infeasible(total_seq_len, max_new)
        ):
            return max_new
        if commit:
            self._expanded_left -= num_pages
            self._host_left -= tree_len
        return self.temp_slot_tokens + max_new

    def admission_exhausted(self, total_seq_len: int, max_new: int) -> bool:
        """True when a candidate is only blocked by quota exhaustion
        (expanded pages / host tokens) — the add_one_req gate keeps such
        candidates queued. Short prompts (< 1 page) and pool-infeasible
        reservations legitimately run as standard: nothing exhausted."""
        num_pages = total_seq_len // self._page_size
        if num_pages <= 0 or self._infeasible(total_seq_len, max_new):
            return False
        return (
            num_pages > self._expanded_left
            or num_pages * self._page_size > self._host_left
        )


class HiSparseV2Coordinator:
    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool: DSATokenToKVPool,
        token_to_kv_pool_allocator,
        top_k: int,
        device_buffer_tokens: int,
        device: str,
        tp_group,
    ):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool = token_to_kv_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.top_k = top_k
        self.device = device
        self.tree_cache = None

        # KV rows hold mixed content (fp8 payload + scales + rope bytes), so
        # the swap-in copy must be bit-exact: use uint8 views (same itemsize
        # for fp8; view(dtype) keeps strides) and byte-sized addressing.
        self._device_kv_u8 = [
            buf.view(torch.uint8) for buf in token_to_kv_pool.kv_buffer
        ]
        self.token_bytes = self._device_kv_u8[0].stride(0)
        self.block_kv = 1 << (self.token_bytes - 1).bit_length()
        # Per-layer host base pointers + row stride, filled when the host
        # pool is attached (set_tree_cache). Persistent tensors: CUDA graphs
        # are captured before attachment, so the kernel reads the pointer at
        # replay time instead of baking in a stale address; zero = masked.
        self._host_ptrs = torch.zeros(
            token_to_kv_pool.layer_num, dtype=torch.int64, device=device
        )
        self._host_stride_t = torch.zeros(1, dtype=torch.int64, device=device)
        self.page_size = token_to_kv_pool.page_size
        self.num_layers = token_to_kv_pool.layer_num
        assert self.top_k % self.page_size == 0, (
            f"HiSparse V2 requires top_k ({self.top_k}) to be a multiple of "
            f"page_size ({self.page_size}) so temp slots map to whole pages."
        )
        assert (
            device_buffer_tokens >= self.top_k
            and device_buffer_tokens % self.page_size == 0
            and (device_buffer_tokens & (device_buffer_tokens - 1)) == 0
        ), (
            f"HiSparse V2 requires device_buffer_size ({device_buffer_tokens}) "
            f"to be >= top_k ({self.top_k}), a multiple of page_size "
            f"({self.page_size}), and a power of 2 (tl.arange slot vectors)."
        )

        max_num_req_slots = req_to_token_pool.req_to_token.shape[0]
        max_context_len = req_to_token_pool.max_context_len
        max_pages_per_req = (max_context_len + self.page_size - 1) // self.page_size

        # Swap-in hit cache (always on): V1-style stable-slot residency
        # on the SINGLE temp device buffer. A per-(request, layer) map
        # records which position each buffer slot holds; the plan kernel
        # retains hit slots (zero copy, zero DMA) and assigns misses
        # empty-first to non-retained slots, keeping stale residents
        # around for multi-step reuse (V1's LRU-ish behavior). Buffer
        # size is V1's device_buffer_size knob (default 2*top_k); it is
        # the per-request lifetime device floor.
        self.temp_slot_tokens = device_buffer_tokens

        # Temp GPU slots for attention KV DMA (per-request device buffer)
        self.temp_slots = torch.full(
            (max_num_req_slots, self.temp_slot_tokens),
            -1,
            dtype=torch.int64,
            device=device,
        )
        # Resident position per buffer slot (-1 = empty), per layer
        # (each layer's indexer selects its own top-k).
        self._slot_pos = torch.full(
            (max_num_req_slots, self.num_layers, self.temp_slot_tokens),
            -1,
            dtype=torch.int32,
            device=device,
        )
        # Per-lane serve plan produced by swap_in_plan_kernel.
        self._plan = torch.full(
            (max_num_req_slots, self.top_k),
            -1,
            dtype=torch.int32,
            device=device,
        )
        # Transient position→slot mapping (shared across layers; the
        # plan kernel self-cleans it back to -1) + small per-call
        # compaction scratch.
        self._pos2slot = torch.full(
            (max_num_req_slots, max_context_len),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self._plan_scratch = torch.zeros(
            (max_num_req_slots, 2 * self.temp_slot_tokens + self.top_k),
            dtype=torch.int32,
            device=device,
        )
        self.top_k_device_locs_buffer = torch.full(
            (max_num_req_slots, self.top_k),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self.num_real_reqs = torch.zeros(1, dtype=torch.int32, device=device)

        # Host location per (req, position) for evicted attention KV.
        # The swap-in kernel reads req_to_token (sentinel -1 = evicted) and
        # falls back to this tensor for the host index.
        self._host_locs = torch.full(
            (max_num_req_slots, max_context_len),
            -1,
            dtype=torch.int64,
            device=device,
        )

        # Eviction queue — callback enqueues precomputed masks,
        # _sync_evictions applies the writes.
        self._eviction_queue: list[_EvictionEvent] = []

        # Expanded indexer page management
        gpu_pool_pages = (token_to_kv_pool.size + self.page_size) // self.page_size
        total_indexer_pages = len(token_to_kv_pool.index_k_with_scale_buffer[0])
        expanded_start = gpu_pool_pages
        expanded_count = total_indexer_pages - expanded_start
        logger.info(
            "HiSparse V2: indexer buffer %d total pages, %d expanded (%.1f MB/layer)",
            total_indexer_pages,
            expanded_count,
            expanded_count
            * token_to_kv_pool.index_k_with_scale_buffer[0].shape[1]
            / 1e6,
        )
        self._indexer_page_allocator = _SimplePageAllocator(expanded_count, device)
        self._indexer_page_offset = expanded_start

        # Per-request expanded indexer page table (-1 = use original page)
        self.req_to_indexer_page = torch.full(
            (max_num_req_slots, max_pages_per_req),
            -1,
            dtype=torch.int32,
            device=device,
        )

        self.tp_group = tp_group
        self.tp_world_size = torch.distributed.get_world_size(group=self.tp_group)
        self.decode_producer_stream = None
        self._host_kv_cache = None
        # req_pool_idx -> prefill_len for admitted (V2-active) requests
        self._active_reqs: dict[int, int] = {}
        # req_pool_idx -> nodes host-locked on this request's behalf at
        # eviction time (see _on_node_evicted); released at finish.
        self._req_host_nodes: dict[int, list] = {}
        # Host-capacity reservation: Σ cache_protected_len over active
        # requests. An admitted prefix must be evictable to host WITH a
        # copy (dropping it masks live attention positions), so V2
        # admission is bounded by host capacity. Conservative: shared /
        # already-backed prefixes are double-counted.
        self._reserved_host_tokens = 0
        self._host_capacity_tokens = 0

    def set_tree_cache(self, tree_cache) -> None:
        self.tree_cache = tree_cache
        self._host_kv_cache = None
        if tree_cache is not None:
            cc = tree_cache.cache_controller
            host_pool = cc.mem_pool_host
            if hasattr(host_pool, "get_pool"):
                self._host_kv_cache = host_pool.get_pool(PoolName.KV)
            else:
                self._host_kv_cache = host_pool
            tree_cache._hisparse_v2_on_evict = self._on_node_evicted
            tree_cache._hisparse_v2_node_active = self._node_backs_active_request
            self._host_capacity_tokens = int(getattr(cc.mem_pool_host, "size", 0))

            # Publish host base pointers + row stride for the swap-in kernel.
            refs = self._host_kv_cache.data_refs
            u8 = [r.view(torch.uint8) for r in refs]
            strides = {v.stride(0) for v in u8}
            assert len(strides) == 1, f"inconsistent host strides: {strides}"
            self._host_ptrs.copy_(
                torch.tensor([v.data_ptr() for v in u8], dtype=torch.int64)
            )
            self._host_stride_t.fill_(u8[0].stride(0))
            logger.info(
                "HiSparse V2: host pool attached, %d layers, row stride %d bytes",
                len(u8),
                u8[0].stride(0),
            )

    def set_decode_producer_stream(self, stream) -> None:
        self.decode_producer_stream = stream

    # ------------------------------------------------------------------
    # Request lifecycle
    # ------------------------------------------------------------------

    def admit_request(self, req) -> bool:
        """Called after prefill (and after cache_unfinished_req).

        Only the tree-owned page-aligned prefix [0, cache_protected_len)
        can ever be evicted, so only those pages need expanded indexer
        copies. The unaligned prefill tail and all decode tokens are
        request-owned and stay in the regular pool pages.
        """
        req_pool_idx = req.req_pool_idx
        tree_len = req.cache_protected_len
        num_pages = tree_len // self.page_size
        if num_pages == 0:
            # Nothing evictable — run as a standard request.
            return False

        # Host-capacity gate: every admitted prefix must be evictable to
        # host with a copy (a copy-less drop masks live attention
        # positions — accuracy loss). Beyond host capacity, fall back to
        # the standard path (device lock kept, prefix pinned — correct).
        if (
            self._host_capacity_tokens > 0
            and self._reserved_host_tokens + tree_len > self._host_capacity_tokens
        ):
            logger.warning(
                "admit_request: host capacity gate rejects req_pool_idx=%d "
                "(reserved %d + need %d > capacity %d); standard fallback",
                req_pool_idx,
                self._reserved_host_tokens,
                tree_len,
                self._host_capacity_tokens,
            )
            return False

        # Apply pending evictions BEFORE activating this request: queued
        # events predate it, and its freshly allocated prefill indices may
        # collide with device indices freed by those evictions.
        self._sync_evictions()

        new_pages = self._indexer_page_allocator.alloc(num_pages)
        if new_pages is None:
            logger.warning(
                "admit_request: indexer page alloc failed for req_pool_idx=%d "
                "(need %d pages, %d available)",
                req_pool_idx,
                num_pages,
                self._indexer_page_allocator.available(),
            )
            return False

        # Temp device buffer for attention KV swap-in (whole pages)
        from sglang.srt.mem_cache.common import evict_from_tree_cache

        evict_from_tree_cache(self.tree_cache, self.temp_slot_tokens)
        temp = self.token_to_kv_pool_allocator.alloc(self.temp_slot_tokens)
        if temp is None:
            logger.warning(
                "admit_request: temp-slot alloc failed for req_pool_idx=%d",
                req_pool_idx,
            )
            self._indexer_page_allocator.free(new_pages)
            return False

        actual_pages = new_pages + self._indexer_page_offset

        token_indices = self.req_to_token_pool.req_to_token[req_pool_idx, :tree_len]
        orig_pages = (token_indices[:: self.page_size] // self.page_size).long()
        for layer_buf in self.token_to_kv_pool.index_k_with_scale_buffer:
            layer_buf[actual_pages.long()] = layer_buf[orig_pages]

        self.req_to_indexer_page[req_pool_idx, :num_pages] = actual_pages
        self.temp_slots[req_pool_idx, : self.temp_slot_tokens] = temp.to(torch.int64)
        self._slot_pos[req_pool_idx] = -1
        self._host_locs[req_pool_idx] = -1
        self._active_reqs[req_pool_idx] = tree_len
        self._reserved_host_tokens += tree_len

        # Release device lock (after all allocations succeed). Host locks
        # are NOT taken here: under write_back, host_value only exists
        # after the first device eviction, and acquire_component_lock
        # (lock_host=True) silently no-ops without it. They are taken in
        # _on_node_evicted instead, when the host copy is guaranteed live.
        self._release_lock_for_eviction(req)

        return True

    def get_indexer_page_table(
        self,
        req_pool_indices: torch.Tensor,
        num_pages: int,
    ) -> Optional[torch.Tensor]:
        """Build a hybrid indexer real_page_table (fully vectorized, no syncs).

        Per page slot, keyed on the page's first req_to_token entry:
        - >= 0 → original page id (base region: shared between kv_buffer
          and index_k_with_scale_buffer, valid in both);
        - -1 (evicted V2 prefix page) → the request's private
          expanded-region indexer page from req_to_indexer_page (copied
          at admission, GPU-resident for the request lifetime).

        Expanded ids exceed kv_buffer's page range, so the result is only
        valid against index_k_with_scale_buffer (indexer scoring); the
        attention side resolves top-k positions separately via
        swap_in_selected_pages. Rows of non-V2 requests come out
        bit-identical to the standard table (neither where-condition
        fires), so a mixed batch can substitute the whole table.

        Returns None when no request is V2-active (standard table correct).
        """
        if not self._active_reqs:
            return None

        idx = req_pool_indices.long()
        num_tokens = num_pages * self.page_size
        firsts = self.req_to_token_pool.req_to_token[idx, : num_tokens : self.page_size]
        orig_pages = firsts // self.page_size
        exp_pages = self.req_to_indexer_page[idx, :num_pages]
        table = torch.where((firsts < 0) & (exp_pages >= 0), exp_pages, orig_pages)
        # Defensive: never emit a negative page index (would read OOB).
        return table.clamp_(min=0).to(torch.int32)

    def swap_in_selected_pages(
        self,
        req_pool_indices: torch.Tensor,
        compressed_seq_lens: torch.Tensor,
        top_k_result: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        """Dual-source swap-in for attention KV (plan + serve).

        For every top-k position: device-resident (incl. all non-V2
        requests) passes through; a device-buffer hit reuses its slot with zero
        copy; a miss DMAs the KV row from pinned host memory into the
        assigned temp slot. Reads only persistent tensors →
        overlap/graph safe.
        """
        from sglang.kernels.ops.kvcache.hisparse_v2 import (
            swap_in_plan_kernel,
            swap_in_serve_kernel,
        )

        num_reqs = req_pool_indices.size(0)
        result = self.top_k_device_locs_buffer[:num_reqs]

        local_layer = layer_id - self.token_to_kv_pool.start_layer
        swap_in_plan_kernel[(num_reqs,)](
            req_pool_indices,
            top_k_result,
            self.req_to_token_pool.req_to_token,
            self._host_locs,
            self._slot_pos,
            self._pos2slot,
            self._plan_scratch,
            self._plan,
            self.num_real_reqs,
            local_layer,
            self.num_layers,
            device_locs_stride=self.req_to_token_pool.req_to_token.stride(0),
            host_locs_stride=self._host_locs.stride(0),
            pos2slot_stride=self._pos2slot.stride(0),
            TOPK=self.top_k,
            NSLOT=self.temp_slot_tokens,
            num_warps=16,
        )
        swap_in_serve_kernel[(num_reqs, self.top_k)](
            self.req_to_token_pool.req_to_token,
            req_pool_indices,
            top_k_result,
            self._host_locs,
            self._host_ptrs,
            self._host_stride_t,
            self._device_kv_u8[local_layer],
            result,
            self.temp_slots,
            self._plan,
            self.num_real_reqs,
            local_layer,
            device_locs_stride=self.req_to_token_pool.req_to_token.stride(0),
            host_locs_stride=self._host_locs.stride(0),
            device_kv_stride=self.token_bytes,
            TOPK=self.top_k,
            NSLOT=self.temp_slot_tokens,
            BLOCK_KV=self.block_kv,
        )
        return result

    def map_last_loc_to_buffer(
        self,
        seq_lens: torch.Tensor,
        out_cache_loc: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
    ) -> None:
        """V1-interface-compat hook (called from prepare_for_decode),
        repurposed as V2's pre-forward eviction sync point.

        The V1 meaning (map the new decode token into the device residency
        buffer) does not apply: V2 decode tokens live in regular pool pages
        which are never evicted (only page-aligned prefill prefixes are
        tree-owned), so no indexer copy or expanded page allocation is
        needed here.

        What MUST happen here is applying queued eviction events: evicted
        device pages get reused by the allocator, so the upcoming forward
        may not launch with stale req_to_token indices (it would read
        another request's KV). This is the last controlled point between
        two forwards — _sync_evictions waits on the in-flight forward
        before writing, and the forward stream's wait_stream makes the
        writes visible to the next one.
        """
        self._sync_evictions()

    def request_finished(self, req) -> None:
        req_pool_idx = req.req_pool_idx
        # Apply pending evictions while this request is still active: the
        # prefix-intact check in cache_finished_req (which runs right after)
        # must see up-to-date sentinels.
        self._sync_evictions()
        if req_pool_idx not in self._active_reqs:
            return
        # Release resources only after the potentially overlapped forward
        # has finished: under overlap scheduling the batch launched one
        # step ahead still contains this request, and its swap-in/indexer
        # kernels read temp_slots / req_to_indexer_page / _host_locs and
        # write into the temp-slot pool pages freed below. (Same fence as
        # V1's request_finished; _sync_evictions above only waits when the
        # eviction queue is non-empty.)
        if self.decode_producer_stream is not None:
            torch.cuda.current_stream().wait_stream(self.decode_producer_stream)
        self._reserved_host_tokens -= self._active_reqs[req_pool_idx]
        del self._active_reqs[req_pool_idx]
        # Release the host locks taken at eviction time on this request's
        # behalf (see _on_node_evicted).
        for node in self._req_host_nodes.pop(req_pool_idx, ()):
            self.tree_cache.dec_host_lock_ref(node)
        # Free temp slots (whole allocation: buffer size is page-aligned).
        # clone() is required: allocator.free() may defer the free into a
        # free-group that holds a reference, and we overwrite the buffer
        # with -1 right below — freeing a view would free page -1.
        temp = self.temp_slots[req_pool_idx, : self.temp_slot_tokens].clone()
        self.token_to_kv_pool_allocator.free(temp)
        self.temp_slots[req_pool_idx] = -1
        # (_slot_pos is left stale: only V2-active rows are consumed by the
        # plan kernel, and admit_request resets the row before activation.)
        # Free expanded indexer pages
        pages = self.req_to_indexer_page[req_pool_idx]
        valid_pages = pages[pages >= 0]
        if len(valid_pages) > 0:
            self._indexer_page_allocator.free(valid_pages - self._indexer_page_offset)
        self.req_to_indexer_page[req_pool_idx] = -1
        self._host_locs[req_pool_idx] = -1

    # ------------------------------------------------------------------
    # Eviction handling
    # ------------------------------------------------------------------

    def host_reservable_left(self) -> int:
        """Host tokens still reservable for new V2 admissions."""
        if self._host_capacity_tokens <= 0:
            return 1 << 60
        return self._host_capacity_tokens - self._reserved_host_tokens

    def make_admit_budget(self, device_pool_tokens: int) -> HiSparseV2AdmitBudget:
        """Snapshot the admission quotas for one PrefillAdder round."""
        return HiSparseV2AdmitBudget(
            page_size=self.page_size,
            temp_slot_tokens=self.temp_slot_tokens,
            expanded_pages_left=self._indexer_page_allocator.available(),
            host_tokens_left=self.host_reservable_left(),
            device_pool_tokens=device_pool_tokens,
        )

    def _node_backs_active_request(self, node) -> bool:
        """Whether *node*'s device KV backs any active V2 request's prefix.

        Used by the tree cache to veto the copy-less drop fallback: a drop
        of such a node would mask live attention positions. Rare path
        (only on write_backup failure under host saturation), so the
        active-row concat is rebuilt per call.
        """
        if not self._active_reqs:
            return False
        vals = node.component_data[ComponentType.FULL].value
        if vals is None or len(vals) == 0:
            return False
        rows = [
            self.req_to_token_pool.req_to_token[idx, :tree_len].long()
            for idx, tree_len in self._active_reqs.items()
        ]
        active = torch.cat(rows)
        active = active[active >= 0]
        if active.numel() == 0:
            return False
        active, _ = torch.sort(active)
        dev = vals.to(device=active.device, dtype=torch.int64)
        pos = torch.searchsorted(active, dev).clamp_(max=active.numel() - 1)
        return bool((active[pos] == dev).any().item())

    def _on_node_evicted(self, node) -> None:
        """Called by tree_cache during _evict_device_leaf, after write_backup.

        Runs on the scheduler thread. Only snapshots the evicted node's
        (device, host) index arrays — no GPU writes, so safe against
        in-flight forwards. The matching against request rows happens in
        _sync_evictions(), applied in FIFO order: if a device index is
        freed, reused, and evicted again, the first event already wiped
        the old row positions to sentinel, so the later event cannot
        mis-attribute host indices.
        """
        if not self._active_reqs:
            return

        cd = node.component_data[ComponentType.FULL]
        dev_vals = cd.value
        if dev_vals is None or len(dev_vals) == 0:
            return
        dev_vals = dev_vals.to(device=self.device, dtype=torch.int64)
        host_vals = cd.host_value
        if host_vals is not None:
            host_vals = host_vals.to(device=self.device, dtype=torch.int64)

        sorted_dev, order = torch.sort(dev_vals)
        sorted_host = host_vals[order] if host_vals is not None else None

        # One TEMPORARY host lock keeps host_value alive until the event is
        # applied (host_value exists now: write_backup acked before this
        # callback; at admission time the lock would silently no-op because
        # host_value is None under write_back). _sync_evictions attributes
        # ownership: requests whose rows actually hold these indices get
        # their own lock, then the temporary one is dropped — so unrelated
        # churn nodes are never pinned for the lifetime of long decodes.
        locked = None
        if sorted_host is not None:
            self.tree_cache.inc_host_lock_ref(node)
            locked = node
        self._eviction_queue.append(_EvictionEvent(sorted_dev, sorted_host, locked))

    def _sync_evictions(self) -> None:
        """Apply queued evictions: sentinel req_to_token + record host locs.

        Waits on the forward stream first so an in-flight forward never
        observes a sentinel without its host index (or loses a device
        index it is still reading). Runs before the next forward launches;
        forward_stream.wait_stream(schedule_stream) makes the writes
        visible to it.
        """
        if not self._eviction_queue:
            return

        if self.decode_producer_stream is not None:
            torch.cuda.current_stream().wait_stream(self.decode_producer_stream)

        match_flags = []  # (event_idx, req_pool_idx) aligned with flag tensor
        flag_tensors = []
        dropped_active = []  # (req_pool_idx, count_tensor) — no-host evictions
        for ei, ev in enumerate(self._eviction_queue):
            n = len(ev.sorted_dev)
            for req_pool_idx, tree_len in self._active_reqs.items():
                row = self.req_to_token_pool.req_to_token[req_pool_idx, :tree_len]
                row64 = row.long()
                pos = torch.searchsorted(ev.sorted_dev, row64).clamp_(max=n - 1)
                match = (ev.sorted_dev[pos] == row64) & (row64 >= 0)
                if ev.sorted_host is not None:
                    hrow = self._host_locs[req_pool_idx, :tree_len]
                    hrow.copy_(torch.where(match, ev.sorted_host[pos], hrow))
                else:
                    # Dropped without a host copy: any active match is
                    # masked in the dual-source kernel → accuracy loss.
                    dropped_active.append((req_pool_idx, match.sum()))
                row.masked_fill_(match, -1)
                if ev.node is not None:
                    match_flags.append((ei, req_pool_idx))
                    flag_tensors.append(match.any())

        if dropped_active:
            counts = torch.stack([c for _, c in dropped_active]).cpu().tolist()
            for (req_pool_idx, _), cnt in zip(dropped_active, counts):
                if cnt > 0:
                    # Invariant violation: the host-capacity gate + drop
                    # veto must make this unreachable. Continuing would
                    # return silently degraded output — fail loud.
                    raise RuntimeError(
                        f"HiSparse V2 ACCURACY LOSS: {cnt} active positions of "
                        f"req_pool_idx={req_pool_idx} evicted without a host "
                        f"copy (would be masked in attention). Admission "
                        f"gate/drop veto invariant broken."
                    )

        # Attribute host-lock ownership: one batched readback (single sync,
        # eviction steps only), then per-request locks for actual hits and
        # release of each event's temporary lock.
        if flag_tensors:
            hits = torch.stack(flag_tensors).cpu().tolist()
            for (ei, req_pool_idx), hit in zip(match_flags, hits):
                if hit and req_pool_idx in self._active_reqs:
                    node = self._eviction_queue[ei].node
                    self.tree_cache.inc_host_lock_ref(node)
                    self._req_host_nodes.setdefault(req_pool_idx, []).append(node)
        for ev in self._eviction_queue:
            if ev.node is not None:
                self.tree_cache.dec_host_lock_ref(ev.node)
        self._eviction_queue.clear()

    # ------------------------------------------------------------------
    # Lock management
    # ------------------------------------------------------------------

    def _release_lock_for_eviction(self, req) -> None:
        """Release radix tree lock to allow HiCache eviction of attention KV."""
        if self.tree_cache is None:
            return
        node = req.last_node
        if node is None:
            return

        self.tree_cache.dec_lock_ref(
            node,
            DecLockRefParams(
                swa_uuid_for_lock=getattr(req, "swa_uuid_for_lock", None),
            ),
            skip_swa=getattr(req, "swa_prefix_lock_released", False),
        )
        req._hisparse_v2_unlocked = True

    # ------------------------------------------------------------------
    # Compat
    # ------------------------------------------------------------------

    def has_ongoing_staging(self) -> bool:
        return False

    def wait_for_pending_backup(self) -> None:
        pass

    def retract_req(self, req) -> None:
        self.request_finished(req)

    def destroy(self) -> None:
        pass
