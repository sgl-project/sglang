"""HISA pool K cache (v3 — paged layout, TMA-friendly).

Pool rows are stored in pages that mirror the main KV cache layout:
``pool_k_pages[num_pool_pages_global, pool_page_size=64, D+4] uint8``.
Each "pool token" is the mean-pool of ``k_block_size=128`` real tokens; 64
pool tokens fit in one page = 8192 real tokens of context per page.

A per-request ``pool_page_tables[R, max_pool_pages_per_req]`` maps
logical pool-pages to physical pool-pages, parallel to sglang's main
``block_tables`` for the real KV cache.

Why paged instead of a flat ``pool_k_buffer``:
- block_mqa can read a pool page via TMA (one T.copy per page, block_N=64
  rows contiguous). Baseline paged block_mqa pattern.
- No gather-to-scratch pass. No per-batch blocked_k scratch. No tail
  scratch — tail is written in-place into ``pool_k_pages[phys, slot]``.
- Per-layer HBM drops ~15× vs the v2b flat+scratch layout.

Scheduler integrations live in ``mem_cache/common.py`` (alloc_for_extend/
alloc_for_decode/release_kv_cache) and ``scheduler_pp_mixin.py``.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch

from sglang.srt.layers.attention.nsa.hisa.tilelang_kernels import (
    fp8_native_paged_mean_pooling_completed_blocks_grouped_interface,
    fp8_native_paged_mean_pooling_completed_blocks_interface,
)
from sglang.srt.mem_cache.memory_pool import NSATokenToKVPool

logger = logging.getLogger(__name__)


POOL_PAGE_SIZE = 64  # pool rows per pool page (matches main KV page_size)


# ---------------------------------------------------------------------------
# Page allocator
# ---------------------------------------------------------------------------


class HisaPoolPageAllocator:
    """Free-list allocator for global pool-page IDs (range [0, num_pages))."""

    def __init__(self, num_pages: int, device: torch.device):
        self.num_pages = num_pages
        self.device = device
        self.free_pages: List[int] = list(range(num_pages))

    def available_size(self) -> int:
        return len(self.free_pages)

    def alloc(self, num_new_pages: int) -> Optional[torch.Tensor]:
        if num_new_pages > len(self.free_pages):
            return None
        out = self.free_pages[:num_new_pages]
        self.free_pages = self.free_pages[num_new_pages:]
        return torch.tensor(out, dtype=torch.int32, device=self.device)

    def free(self, page_ids: torch.Tensor) -> None:
        if page_ids.numel() == 0:
            return
        self.free_pages.extend(page_ids.tolist())

    def clear(self) -> None:
        self.free_pages = list(range(self.num_pages))


# ---------------------------------------------------------------------------
# Per-request pool-page mapping
# ---------------------------------------------------------------------------


class HisaReqToPoolPagePool:
    """Parallel to :class:`ReqToTokenPool` but for pool pages.

    Layout: ``req_to_pool_page[req_pool_idx, pool_page_idx_within_req]``
    stores the global ``pool_page_id`` assigned to that logical page.
    """

    def __init__(self, size: int, max_pool_pages_per_req: int, device: torch.device):
        self.size = size
        self.max_pool_pages_per_req = max_pool_pages_per_req
        self.device = device
        # +1 padding row at index 0 to mirror ReqToTokenPool: cuda-graph padded
        # batches default req_pool_indices to 0, and the pooling kernel shares
        # the same `max_running_req` dynamic symbol with ReqToToken, so
        # shape[0] must match.
        self._alloc_size = size + 1
        self.req_to_pool_page = torch.zeros(
            (self._alloc_size, max_pool_pages_per_req),
            dtype=torch.int32,
            device=device,
        )
        self.num_pool_pages_cpu = torch.zeros(self._alloc_size, dtype=torch.int32)

    def write(
        self,
        req_idx: int,
        pool_page_idx_start: int,
        page_ids: torch.Tensor,
    ) -> None:
        n = page_ids.numel()
        end = pool_page_idx_start + n
        self.req_to_pool_page[req_idx, pool_page_idx_start:end] = page_ids
        if end > int(self.num_pool_pages_cpu[req_idx].item()):
            self.num_pool_pages_cpu[req_idx] = end

    def get_ids(self, req_idx: int) -> torch.Tensor:
        n = int(self.num_pool_pages_cpu[req_idx].item())
        return self.req_to_pool_page[req_idx, :n]

    def num_pages(self, req_idx: int) -> int:
        return int(self.num_pool_pages_cpu[req_idx].item())

    def free(self, req_idx: int) -> None:
        self.num_pool_pages_cpu[req_idx] = 0


# ---------------------------------------------------------------------------
# The hisa pool (v3 — paged)
# ---------------------------------------------------------------------------


class HisaNSATokenToKVPool(NSATokenToKVPool):
    """NSA pool extended with a paged pool-K cache.

    Added state (all per-layer and GPU-resident):
    - ``pool_k_pages[layer][num_pool_pages_global, pool_page_size, D+4]``
      uint8: stores mean-pooled K, laid out identically to the main paged
      KV cache. One "row" per completed pool block; 64 rows per page.
    - ``pool_page_allocator`` and ``req_to_pool_page``: described above.

    No longer needed (vs v2b):
    - blocked_k scratch (replaced by TMA-friendly paged reads)
    - tail scratch (tail is written in-place into pool_k_pages)
    """

    def __init__(
        self,
        *args,
        k_block_size: int = 128,
        pool_page_size: int = POOL_PAGE_SIZE,
        max_running_requests: int = 2048,
        max_pool_pages_per_req: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.k_block_size = k_block_size
        self.pool_page_size = pool_page_size

        tokens_per_pool_page = k_block_size * pool_page_size
        # Enough pool pages to cover the whole KV-cache token budget, with a
        # small allocator slack. Per page: 8192 real tokens of context. For
        # size=353664 tokens → num_pool_pages_global = 44 pages.
        num_pool_pages_global = (
            self.size + tokens_per_pool_page - 1
        ) // tokens_per_pool_page
        # Over-provision by 50% to reduce allocator contention over mixed-size
        # requests (pool-page granularity means short requests can still burn
        # a full page each).
        num_pool_pages_global = int(num_pool_pages_global * 1.5) + 4
        self.num_pool_pages_global = num_pool_pages_global

        # pool_k_pages — same byte layout as the main KV cache.
        # 2D SoA: [num_pool_pages_global, pool_page_size * (D + 4)] uint8
        #   bytes [0, pool_page_size * D): all 64 pool rows' fp8 data concatenated
        #   bytes [pool_page_size * D, pool_page_size * (D+4)): all scales concatenated
        D = self.index_head_dim
        page_bytes = pool_page_size * (D + 4)
        self.pool_k_pages: List[torch.Tensor] = [
            torch.zeros(
                (num_pool_pages_global, page_bytes),
                dtype=torch.uint8,
                device=self.device,
            )
            for _ in range(self.layer_num)
        ]

        # Allocator + per-request page map.
        self.pool_page_allocator = HisaPoolPageAllocator(
            num_pool_pages_global, self.device
        )
        if max_pool_pages_per_req is None:
            # Per-request max = pages needed to cover the LONGEST SINGLE
            # request's context. Caller typically passes max_model_len-based
            # sizing; fall back to the whole-pool bound as a safe default.
            # NOTE: the kernel's grid y = max_pool_pages_per_req, so keeping
            # this SMALL is critical for decode perf.
            max_pool_pages_per_req = num_pool_pages_global
        self.max_pool_pages_per_req = max_pool_pages_per_req
        self.req_to_pool_page = HisaReqToPoolPagePool(
            max_running_requests, max_pool_pages_per_req, self.device
        )

        # Persistent int32 scratch for prev / new seq_lens — graph-capturable
        # dtype casts (source may be int64 in prefill path). Tiny tensors.
        # +1 to mirror ReqToTokenPool sizing convention even though these
        # scratches are sliced by batch position ([:B]) rather than indexed
        # by req_pool_idx — defensive, avoids future misuse.
        self._scratch_prev_lens_i32 = torch.zeros(
            max_running_requests + 1,
            dtype=torch.int32,
            device=self.device,
        )
        self._scratch_new_lens_i32 = torch.zeros(
            max_running_requests + 1,
            dtype=torch.int32,
            device=self.device,
        )
        # Persistent int32 scratch for the per-batch pool_page_tables slice
        # (see get_pool_page_tables): avoids a per-layer allocator hit from
        # fancy indexing. Full width; caller takes a [:B] view.
        self._scratch_pool_page_tables = torch.zeros(
            (max_running_requests + 1, max_pool_pages_per_req),
            dtype=torch.int32,
            device=self.device,
        )
        # Per-layer pool watermark: ``pool_watermark[layer, req]`` is the
        # token position up to which we've already mean-pooled K for that
        # (layer, request). On the next extend we only pool from the
        # watermark forward, instead of re-pooling [0, new_seq) every
        # time (the previous mitigation for the prefix-cache bug; see
        # A4a). Watermark resets to 0 on free, so a slot reused by a
        # cache-hit request naturally pools its full prefix range.
        # Per-layer (not shared) because layers run sequentially and
        # each layer's update_pool both reads and writes the watermark —
        # a shared watermark would be advanced by layer 0 and silently
        # skip layers 1..N-1.
        # +1 padding column to mirror ReqToTokenPool: this buffer is indexed
        # by req_pool_idx (values 1..max_running_requests; cuda-graph dummies
        # use 0), so dim-1 must be max_running_requests + 1.
        self._pool_watermark_i32 = torch.zeros(
            (self.layer_num, max_running_requests + 1),
            dtype=torch.int32,
            device=self.device,
        )

        logger.info(
            "HisaNSATokenToKVPool(v3): k_block_size=%d, pool_page_size=%d, "
            "num_pool_pages_global=%d, max_pool_pages_per_req=%d, "
            "pool_k_pages=%.1f MB/layer",
            k_block_size,
            pool_page_size,
            num_pool_pages_global,
            max_pool_pages_per_req,
            num_pool_pages_global
            * pool_page_size
            * (self.index_head_dim + 4)
            / 1024
            / 1024,
        )

    # ------------------------------------------------------------------
    # Scheduler-side alloc/free hooks
    # ------------------------------------------------------------------

    @staticmethod
    def _ceildiv(a: int, b: int) -> int:
        return (a + b - 1) // b

    def alloc_pool_pages_for_extend(
        self,
        req_pool_indices: List[int],
        prefix_lens: List[int],
        seq_lens: List[int],
    ) -> None:
        """For each request, allocate pool pages that this extend step's
        ``seq_len`` reaches. Allocation is at PAGE granularity — one page
        covers ``pool_page_size * k_block_size`` real tokens.

        Called right after ``alloc_for_extend``'s main-buffer alloc.

        Uses ``num_pool_pages_cpu[req_idx]`` (the actual count of allocated
        pool pages on this slot) as the start of the new alloc, NOT
        ``ceildiv(prefix_lens)``. This is critical for prefix-cache hits:
        a freshly-reused slot has ``num_pages_cpu == 0`` even though
        ``prefix_lens > 0``, so we (correctly) allocate fresh pool pages
        for the prefix range. The old behaviour skipped that alloc and
        left ``req_to_pool_page[req_idx, 0..prev_pages]`` pointing at the
        previous tenant's pool pages — which the "pool from 0" mitigation
        in ``_store_index_k_cache`` then wrote through, corrupting other
        requests' pool data. ``prefix_lens`` is now unused but kept in the
        signature for caller compatibility.
        """
        del prefix_lens  # see docstring
        K = self.k_block_size
        P = self.pool_page_size
        tokens_per_page = K * P
        for i, req_idx in enumerate(req_pool_indices):
            existing_pages = self.req_to_pool_page.num_pages(req_idx)
            new_pages = self._ceildiv(seq_lens[i], tokens_per_page)
            num_new = new_pages - existing_pages
            if num_new <= 0:
                continue
            new_page_ids = self.pool_page_allocator.alloc(num_new)
            if new_page_ids is None:
                raise RuntimeError(
                    f"HisaPoolPageAllocator OOM on extend: need {num_new}, "
                    f"have {self.pool_page_allocator.available_size()}"
                )
            self.req_to_pool_page.write(req_idx, existing_pages, new_page_ids)

    def alloc_pool_pages_for_decode(
        self,
        req_pool_indices: List[int],
        seq_lens_after_decode: List[int],
        token_per_req: int = 1,
    ) -> None:
        """If the decode step crosses a pool-page boundary, alloc one page."""
        K = self.k_block_size
        P = self.pool_page_size
        tokens_per_page = K * P
        for i, req_idx in enumerate(req_pool_indices):
            new_len = seq_lens_after_decode[i]
            prev_len = new_len - token_per_req
            prev_pages = self._ceildiv(prev_len, tokens_per_page)
            new_pages = self._ceildiv(new_len, tokens_per_page)
            num_new = new_pages - prev_pages
            if num_new <= 0:
                continue
            new_page_ids = self.pool_page_allocator.alloc(num_new)
            if new_page_ids is None:
                raise RuntimeError(f"HisaPoolPageAllocator OOM on decode req={req_idx}")
            self.req_to_pool_page.write(req_idx, prev_pages, new_page_ids)

    def free_req_pool_pages(self, req_pool_idx: int) -> None:
        """Release a request's pool-page IDs back to the allocator."""
        ids = self.req_to_pool_page.get_ids(req_pool_idx)
        if ids.numel() > 0:
            self.pool_page_allocator.free(ids)
        self.req_to_pool_page.free(req_pool_idx)
        # Reset watermark across all layers so a cache-hit reuse of this
        # slot pools its full prefix range from 0 (see A4b).
        self._pool_watermark_i32[:, req_pool_idx] = 0

    # ------------------------------------------------------------------
    # Store-side hook (called per layer from HisaIndexer._store_index_k_cache)
    # ------------------------------------------------------------------

    def load_extend_prev_seq_lens_from_watermark(
        self,
        layer_id: int,
        req_pool_indices: torch.Tensor,
        out_prev: torch.Tensor,
    ) -> None:
        """Fill ``out_prev[:B] = pool_watermark[layer_local, req_pool_indices]``.

        Used by the extend path of ``_store_index_k_cache`` so each layer's
        update_pool only re-pools blocks from the last-pooled position
        forward, instead of starting from 0 every time. ``out_prev`` must
        be sized at least ``[B]`` and dtype int32 (passed-through
        ``_scratch_prev_lens_i32[:B]``).
        """
        layer_local = layer_id - self.start_layer
        torch.index_select(
            self._pool_watermark_i32[layer_local],
            0,
            req_pool_indices,
            out=out_prev,
        )

    def advance_pool_watermark(
        self,
        layer_id: int,
        req_pool_indices: torch.Tensor,
        new_seq_lens: torch.Tensor,
    ) -> None:
        """Advance ``pool_watermark[layer, req] = floor(new_seq_lens, K)``.

        Called after ``update_pool_for_completed_blocks`` on the extend
        path. Floors to the K-block boundary because update_pool only
        writes COMPLETED blocks; the residual tail (seq_len % K tokens)
        is the responsibility of ``tail_only_v3``.
        """
        layer_local = layer_id - self.start_layer
        K = self.k_block_size
        new_floored = (new_seq_lens // K) * K
        # int32 indices are accepted by index_copy_ as long as the source
        # is int32 and target is int32.
        idx = req_pool_indices
        if idx.dtype != torch.int64:
            idx = idx.to(torch.int64)
        self._pool_watermark_i32[layer_local].index_copy_(0, idx, new_floored)

    def update_pool_for_completed_blocks(
        self,
        layer_id: int,
        req_to_token: torch.Tensor,  # [max_running_req, max_ctx] int32
        req_pool_indices: torch.Tensor,  # [B] int64
        prev_seq_lens: torch.Tensor,  # [B] int32
        new_seq_lens: torch.Tensor,  # [B] int32
        max_pool_per_req_grid: int,
    ) -> None:
        """Mean-pool any blocks that became full during this forward and
        write them into ``pool_k_pages[phys, slot, :D]``. Graph-capturable.

        Dispatch: tilelang vanilla for k>=paged_block_size (=64), tilelang
        grouped for k<paged_block_size (one pool block per CTA, K-row slice
        of the underlying paged block). The grouped variant requires
        ``paged_block_size % k_block_size == 0``.
        """
        kv_cache_flat = self.index_k_with_scale_buffer[layer_id - self.start_layer]
        pool_k_pages = self.pool_k_pages[layer_id - self.start_layer]
        if self.k_block_size < self.page_size:
            fp8_native_paged_mean_pooling_completed_blocks_grouped_interface(
                kv_cache_flat=kv_cache_flat,
                req_to_token=req_to_token,
                pool_page_tables=self.req_to_pool_page.req_to_pool_page,
                req_pool_indices=req_pool_indices,
                prev_seq_lens=prev_seq_lens,
                new_seq_lens=new_seq_lens,
                pool_k_pages=pool_k_pages,
                k_block_size=self.k_block_size,
                paged_block_size=self.page_size,
                pool_page_size=self.pool_page_size,
                max_pool_per_req_grid=max_pool_per_req_grid,
            )
        else:
            fp8_native_paged_mean_pooling_completed_blocks_interface(
                kv_cache_flat=kv_cache_flat,
                req_to_token=req_to_token,
                pool_page_tables=self.req_to_pool_page.req_to_pool_page,
                req_pool_indices=req_pool_indices,
                prev_seq_lens=prev_seq_lens,
                new_seq_lens=new_seq_lens,
                pool_k_pages=pool_k_pages,
                k_block_size=self.k_block_size,
                paged_block_size=self.page_size,
                pool_page_size=self.pool_page_size,
                max_pool_per_req_grid=max_pool_per_req_grid,
            )

    # ------------------------------------------------------------------
    # Decode-side helpers (used by HisaIndexer._get_topk_paged)
    # ------------------------------------------------------------------

    def get_pool_k_pages(self, layer_id: int) -> torch.Tensor:
        """``[num_pool_pages_global, pool_page_size, D + 4] uint8`` per layer."""
        return self.pool_k_pages[layer_id - self.start_layer]

    def get_pool_page_tables(self, req_pool_indices: torch.Tensor) -> torch.Tensor:
        """Return ``[B, max_pool_pages_per_req] int32`` of pool_page_ids for
        the requested per-query request indices.

        Dim-1 is always the full column cap (== allocated table width). The
        v3 block_mqa kernel masks beyond ``num_pool_pages`` via
        ``context_lens_pool``.

        Uses a persistent scratch buffer (written in-place via ``index_select``)
        so each layer avoids a fresh allocator hit. The returned slice aliases
        scratch; callers must consume it before the next ``get_pool_page_tables``
        call on the same pool (which overwrites scratch). In decode, the
        indexer calls this once per layer and passes the tensor directly into
        the kernel launch, so the lifetime is safe.
        """
        B = req_pool_indices.shape[0]
        out = self._scratch_pool_page_tables[:B]
        torch.index_select(
            self.req_to_pool_page.req_to_pool_page,
            0,
            req_pool_indices,
            out=out,
        )
        return out


# ---------------------------------------------------------------------------
# Scheduler gate helpers (no-op for non-HISA pools)
# ---------------------------------------------------------------------------


def maybe_alloc_hisa_pool_for_extend(
    kv_pool,
    req_pool_indices: list,
    prefix_lens: list,
    seq_lens: list,
) -> None:
    if isinstance(kv_pool, HisaNSATokenToKVPool):
        kv_pool.alloc_pool_pages_for_extend(req_pool_indices, prefix_lens, seq_lens)


def maybe_alloc_hisa_pool_for_decode(
    kv_pool,
    req_pool_indices: list,
    seq_lens_after_decode: list,
    token_per_req: int = 1,
) -> None:
    if isinstance(kv_pool, HisaNSATokenToKVPool):
        kv_pool.alloc_pool_pages_for_decode(
            req_pool_indices, seq_lens_after_decode, token_per_req=token_per_req
        )


def maybe_free_hisa_pool_blocks(kv_pool, req_pool_idx: int) -> None:
    if isinstance(kv_pool, HisaNSATokenToKVPool):
        kv_pool.free_req_pool_pages(req_pool_idx)
