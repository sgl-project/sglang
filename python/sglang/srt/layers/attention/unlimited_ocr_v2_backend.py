"""Unlimited-OCR v2 DASH block-sparse attention backend.

Each sparse block maps to one KV page. Prompts require ``<PAGE>`` before every
``<image>`` so Page2D coordinates match training. Prefix caching is unsupported
because resumed requests skip page-state initialization.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch
import triton

from sglang.kernels.ops.attention.unlimited_ocr_dash.attend import (
    dash_sparse_attn,
    dash_sparse_attn_decode,
)
from sglang.kernels.ops.attention.unlimited_ocr_dash.decode_meta import (
    local_union_fused,
)
from sglang.kernels.ops.attention.unlimited_ocr_dash.route import (
    SCAN_TILE,
    kbar_all_plan,
    kbar_refresh_tail_triton,
    kbar_tail_plan,
    prefill_row2req,
    route_decode_triton,
    route_prefill_triton,
    win_bounds_decode,
)
from sglang.srt.configs.model_config import get_unlimited_ocr_v2_sparse_config
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.unlimited_dash_ops.dash_router import (
    router_from_sparse_config,
)
from sglang.srt.layers.attention.unlimited_dash_ops.dash_stage0 import (
    resolve_stage0,
)
from sglang.srt.layers.attention.unlimited_dash_ops.decode_meta import (
    cos_sin_2d_fused,
    decode_allow_fused,
    decode_meta_fused,
)
from sglang.srt.layers.attention.unlimited_dash_ops.page2d_rope import (
    PageCoordState,
    PageRope2D,
    page_capacity,
)
from sglang.srt.layers.attention.unlimited_dash_ops.page_state import (
    BatchLayout,
    PageIsolationState,
)
from sglang.srt.mem_cache.memory_pool import MiniMaxSparseKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.unlimited_ocr_v2_support import document_reuse as uocr_doc_reuse

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)

# Page window is only meaningful when a tile width was resolved (SCAN_TILE).
_ENT_WINDOW = SCAN_TILE > 0

# A seeded coordinate row's stamp until its request's first step refreshes it:
# above any step count, so LRU never evicts it first.
_COORD_ROW_PINNED = 1 << 60


def _uses_static_width(*, use_decode_twin: bool, use_prefill_twin: bool) -> bool:
    return use_decode_twin or use_prefill_twin


class UnlimitedOCRV2Metadata:
    """Per-step geometry, shared by every DASH layer of one forward.

    A step is all-prefill (EXTEND / MIXED) or all-decode (DECODE), never mixed.
    """

    __slots__ = (
        "is_decode",
        "num_tokens",
        "num_reqs",
        "block_table",
        "seq_lens",
        "context_lens",
        "cu_seqlens_q",
        "max_query_len",
        "max_seq_len",
        "max_block",
        "decode_query_len",
        "slot_mapping",
        "layout",
        "q_page",
        "query_allow",
        "coords",
        # Per-step caches shared by all layers (built lazily on first use).
        "kbar_plan",
        "row2req",
        "image_allow",
        "win",
        # Fixed-address routing storage selected by the graph entry points.
        "use_decode_twin",
        "use_prefill_twin",
    )

    def __init__(self) -> None:
        # Eager decode metadata is populated in several steps.  Keep optional
        # geometry explicitly initialized so the first decode can safely take
        # the same RoPE refresh path as prefill and graph replay.
        self.coords = None
        self.q_page = None
        self.query_allow = None
        self.kbar_plan = None
        self.row2req = None
        self.image_allow = None
        self.win = None
        self.use_decode_twin = False
        self.use_prefill_twin = False


class UnlimitedOCRV2AttnBackend(AttentionBackend):
    """DASH sparse attention over two paged KV groups (main K/V + index K)."""

    # max_seq_len drives the eager block-table width (graphs pin it instead).
    needs_cpu_seq_lens = True
    # Document KV reuse whose queued edits run before each forward.
    doc_reuse: Optional[uocr_doc_reuse.DocReuse] = None

    def __init__(self, model_runner: ModelRunner):
        pool = model_runner.token_to_kv_pool
        assert isinstance(pool, MiniMaxSparseKVPool), (
            "Unlimited-OCR v2 needs the two-group sparse KV pool; got "
            f"{type(pool).__name__}"
        )
        self.kv_pool = pool
        self.token_to_kv_pool = pool
        self.req_to_token_pool = model_runner.req_to_token_pool
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.device = model_runner.device

        hf_config = model_runner.model_config.hf_config
        sparse_cfg = get_unlimited_ocr_v2_sparse_config(hf_config)
        self.block_size = int(sparse_cfg["sparse_block_size"])
        self.local_blocks = max(1, int(sparse_cfg.get("sparse_local_block", 1)))
        self.idx_head_dim = int(sparse_cfg.get("sparse_index_dim", 128))
        self.q_chunk = int(sparse_cfg.get("dash_query_chunk", 2048))
        self.stage0 = resolve_stage0(sparse_cfg)
        self.num_layers = int(model_runner.model_config.num_hidden_layers)
        self.max_context_len = int(model_runner.model_config.context_len)

        assert self.kv_pool.page_size == self.block_size, (
            f"page_size ({self.kv_pool.page_size}) must equal sparse_block_size "
            f"({self.block_size}): one sparse block is one KV page."
        )
        if self.stage0 != "head_cls":
            raise NotImplementedError(
                f"dash_stage0={self.stage0!r} is not implemented in sglang; only "
                "the trained 'head_cls' stage-0 summarization is."
            )

        # Per-layer router (each layer has its own entmax temperature / floor).
        self.routers = [
            router_from_sparse_config(sparse_cfg, lid) for lid in range(self.num_layers)
        ]
        r0 = self.routers[0]
        self.topm = r0.topm
        self.out_slots = r0.out_slots
        # local + buffer width: the attend reads local blocks out of the same row.
        self.buf_width = max(self.out_slots, 1)

        # Per-layer head_cls page summaries and index-key rotaries live here so
        # the model layers stay stateless w.r.t. the step.
        self._kbar: list[Optional[torch.Tensor]] = [None] * self.num_layers
        # head_cls per layer, recorded on first use for out-of-forward summaries.
        self._uocr_head_cls: dict[int, torch.Tensor] = {}
        self._logged_first = False
        self._init_decode_buffer_slots()
        self._init_page_state(sparse_cfg)

        # Use the checkpoint's per-page token theta, not the global YaRN theta.
        head_dim = model_runner.model_config.head_dim
        self.model_dtype = model_runner.dtype
        self.rope_main = _rope_from_cfg(sparse_cfg, head_dim, "main")
        self.rope_index = _rope_from_cfg(sparse_cfg, self.idx_head_dim, "index")
        if self.rope_main is None or self.rope_index is None:
            raise NotImplementedError(
                "Unlimited-OCR v2 requires sparse_main_rope_mode == "
                "sparse_rope_mode == 'page2d'."
            )
        # cos/sin depend only on (coords, freq table, dtype): layers and branches
        # with the same rope signature share one table per step.
        self._cos_sin_cache: dict = {}

        self.forward_metadata: Optional[UnlimitedOCRV2Metadata] = None

        # Captured grid and reduction widths must not grow with sequence length.
        self.max_block_static = max(
            1, triton.cdiv(self.max_context_len, self.block_size)
        )
        # Set by init_cuda_graph_state; until then every buffer grows on demand.
        self.cg_max_bs = 0
        self.cg_max_tokens = 0
        self._cg_ready = False

        self._init_prefill_buffer_slots()

        logger.info(
            "[Unlimited-OCR v2] DASH sparse attention: block=%d local=%d "
            "read_budget=%d (buffer_width=%d) index_dim=%d stage0=%s "
            "page_token_id=%s per_page_image_tokens=%d image_bucket=%s",
            self.block_size,
            self.local_blocks,
            r0.budget_remote,
            self.buf_width,
            self.idx_head_dim,
            self.stage0,
            self.page_state.page_token_id,
            self.page_state.per_page_image_tokens,
            self.image_bucket_constraint,
        )

    def attach_doc_reuse(self, doc_reuse: uocr_doc_reuse.DocReuse) -> None:
        """Run ``doc_reuse``'s queued KV/state edits before every forward."""
        self.doc_reuse = doc_reuse

    @torch.no_grad()
    def write_page_isolation(
        self,
        *,
        slots: torch.Tensor,
        img_page: torch.Tensor,
        marker_cum: torch.Tensor,
        img_run: torch.Tensor,
        prompt_pages: torch.Tensor,
    ) -> None:
        """Overwrite the page-isolation state of KV ``slots`` (int32 per slot)."""
        ps = self.page_state
        ps._ensure(self._index_pages() * self.block_size, slots.device)
        ps._slot_img_page[slots] = img_page
        ps._slot_marker_cum[slots] = marker_cum
        ps._slot_img_run[slots] = img_run
        ps._slot_prompt_pages[slots] = prompt_pages

    @torch.no_grad()
    def refresh_page_summaries(
        self, *, pages: torch.Tensor, toks: torch.Tensor
    ) -> None:
        """Recompute every layer's head_cls summary of KV ``pages`` from their keys.

        ``toks[i]`` is how many leading tokens of ``pages[i]`` are valid.
        """
        import math

        from sglang.srt.layers.attention.unlimited_dash_ops.dash_stage0 import (
            _summary_keys,
        )

        bs = self.block_size
        valid = torch.arange(bs, device=pages.device)[None, :] < toks[:, None]
        inv_sqrt_d = 1.0 / math.sqrt(self.idx_head_dim)
        for lid in range(self.num_layers):
            head_cls = self._uocr_head_cls.get(lid)
            if head_cls is None:
                continue
            keys = self.index_kv_view(lid)[pages].float()
            k_bar = _summary_keys(keys, valid, head_cls.float(), inv_sqrt_d)
            self._kbar_cache(lid)[pages] = k_bar

    @torch.no_grad()
    def seed_coord_row(
        self,
        *,
        first_block: torch.Tensor,
        marker: int,
        prompt_pages: int,
        counts: list[int],
    ) -> None:
        """Seed the page2d coordinate row keyed by ``first_block`` with a carry.

        The row is pinned against LRU eviction until a step of its request
        refreshes the stamp, or until ``unpin_coord_rows`` releases it.
        """
        cs = self.coord_state
        cs._ensure(first_block.device)
        key = first_block.to(torch.int64).reshape(())
        match = (cs.key == key) & (cs.key >= 0)
        prio = torch.where(cs.key >= 0, cs.stamp, torch.full_like(cs.stamp, -1))
        prio[cs._dummy] = torch.iinfo(torch.int64).max
        row = torch.where(match.any(), match.to(torch.int32).argmax(), prio.argmin())
        row = row.reshape(1)
        cs.key[row] = key
        cs.stamp[row] = _COORD_ROW_PINNED
        cs.st_marker[row] = marker
        cs.st_pp[row] = prompt_pages
        cs.cnt[row] = torch.tensor(counts, dtype=cs.cnt.dtype, device=cs.cnt.device)

    @torch.no_grad()
    def unpin_coord_rows(self, first_blocks: Optional[torch.Tensor]) -> None:
        """Make still-pinned rows of ``first_blocks`` (None: all) the oldest.

        A row its request already stepped holds a fresh stamp and is untouched.
        """
        cs = self.coord_state
        if cs.key is None:
            return
        pinned = cs.stamp >= _COORD_ROW_PINNED
        if first_blocks is not None:
            keys = first_blocks.to(device=cs.key.device, dtype=torch.int64)
            pinned &= (cs.key[:, None] == keys[None, :]).any(dim=1)
        cs.stamp.masked_fill_(pinned, 0)

    def _init_decode_buffer_slots(self) -> None:
        """declare eager scratch and decode graph buffer slots"""
        self._idx_buf: Optional[torch.Tensor] = None
        self._bias_buf: Optional[torch.Tensor] = None
        self._eager_idx_buf: Optional[torch.Tensor] = None
        self._eager_bias_buf: Optional[torch.Tensor] = None
        self._score_buf: Optional[torch.Tensor] = None
        self._score_buf_win: Optional[torch.Tensor] = None
        self._allow_mask: Optional[torch.Tensor] = None
        self._eager_score_buf: Optional[torch.Tensor] = None
        self._eager_score_buf_win: Optional[torch.Tensor] = None
        self._eager_allow_mask: Optional[torch.Tensor] = None
        # Fixed-address block table + its column index, allocated by
        # init_cuda_graph_state (None while graphs are off).
        self._bt_buf: Optional[torch.Tensor] = None
        self._bt_cols: Optional[torch.Tensor] = None
        self._seq_lens_buf: Optional[torch.Tensor] = None
        self._ctx_lens_buf: Optional[torch.Tensor] = None
        self._coord_tok_buf: Optional[torch.Tensor] = None
        self._coord_page_buf: Optional[torch.Tensor] = None
        self._qpage_buf: Optional[torch.Tensor] = None
        self._img_allow_buf: Optional[torch.Tensor] = None
        self._kbar_plan_buf: Optional[tuple] = None
        self._win_bufs: Optional[tuple] = None
        # Fixed-address cos/sin per branch, allocated by init_cuda_graph_state.
        self._cos_sin_buf: dict = {}

    def _init_prefill_buffer_slots(self) -> None:
        """Declare the plain-prefill graph buffer slots."""
        # Plain-prefill graphs use dedicated `_p_*` buffers allocated before
        # capture; over-capacity prefill remains eager.
        self._pcg_ready = False
        self._pcg_max_tokens = 0
        self._pcg_max_reqs = 0
        # Marks plain-prefill steps using their dedicated graph buffers.
        self._in_prefill_cg = False
        self._p_seq_lens_buf: Optional[torch.Tensor] = None
        self._p_ctx_lens_buf: Optional[torch.Tensor] = None
        self._p_culen_buf: Optional[torch.Tensor] = None
        self._p_bt_buf: Optional[torch.Tensor] = None
        self._p_bt_cols: Optional[torch.Tensor] = None
        self._p_cos_sin_buf: dict = {}
        self._p_idx_buf: Optional[torch.Tensor] = None
        self._p_bias_buf: Optional[torch.Tensor] = None
        self._p_score_buf: Optional[torch.Tensor] = None
        self._p_img_allow_buf: Optional[torch.Tensor] = None
        self._p_kbar_all_plan_buf: Optional[tuple] = None
        self._p_row2req_buf: Optional[torch.Tensor] = None

    def _init_page_state(self, sparse_cfg: dict) -> None:
        """Build page isolation and page2d coordinate state."""
        # Page isolation + page2d coordinates: one instance per process.
        self.page_state = PageIsolationState(sparse_cfg, self.block_size)
        # Per req_to_token row: absolute position of the first generated token,
        # written for every extend by note_prompt_lens. Unset rows never count
        # as generated, so an extend without a recorded start is all prompt.
        self._gen_start_row = torch.full(
            (int(self.req_to_token.shape[0]),),
            1 << 62,
            dtype=torch.int64,
            device=self.req_to_token.device,
        )
        # Per req_to_token row: the window's global first-page index, folded by
        # the page-axis period; added to RoPE page coordinates only.
        self._page_base_row = torch.zeros(
            (int(self.req_to_token.shape[0]),),
            dtype=torch.int64,
            device=self.req_to_token.device,
        )
        max_reqs = int(self.req_to_token.shape[0]) + 8
        self.coord_state = PageCoordState(
            sparse_cfg,
            max_reqs=max_reqs,
            max_pages=page_capacity(self.max_context_len),
        )
        self.image_bucket_constraint = bool(
            sparse_cfg.get("image_bucket_constraint", False)
        )

    # ------------------------------------------------------------------
    # Per-step buffers
    # ------------------------------------------------------------------
    def _no_alloc_while_capturing(self, what: str) -> None:
        """Reject cross-step allocations during CUDA graph capture.

        Capture sizes share a graph pool, so reused addresses could corrupt
        state embedded in earlier graphs.
        """
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"[Unlimited-OCR v2] {what} would be allocated during cuda graph "
                "capture. Every cross-step buffer must be sized in "
                "init_cuda_graph_state, at the maximum geometry of the whole "
                "capture sweep."
            )

    @staticmethod
    def _fixed_routing_views(
        idx: Optional[torch.Tensor],
        bias: Optional[torch.Tensor],
        *,
        num_tokens: int,
        capacity: int,
        owner: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if idx is None or bias is None or num_tokens > capacity:
            raise RuntimeError(
                f"[Unlimited-OCR v2] {owner} routing capacity is {capacity}, "
                f"but this step needs {num_tokens} tokens."
            )
        return idx[:num_tokens], bias[:num_tokens]

    def _routing_buffers(
        self, md: UnlimitedOCRV2Metadata, device
    ) -> tuple[torch.Tensor, ...]:
        """Select graph-owned routing storage or growable eager scratch."""
        if md.use_prefill_twin:
            return self._fixed_routing_views(
                self._p_idx_buf,
                self._p_bias_buf,
                num_tokens=md.num_tokens,
                capacity=self._pcg_max_tokens,
                owner="prefill cuda graph",
            )
        if md.use_decode_twin:
            return self._fixed_routing_views(
                self._idx_buf,
                self._bias_buf,
                num_tokens=md.num_tokens,
                capacity=self.cg_max_tokens,
                owner="decode cuda graph",
            )
        if self._eager_idx_buf is None or self._eager_idx_buf.shape[0] < md.num_tokens:
            self._no_alloc_while_capturing("eager routing buffers")
            rows = max(md.num_tokens, 1024)
            self._eager_idx_buf = torch.full(
                (rows, self.buf_width), -1, dtype=torch.int32, device=device
            )
            self._eager_bias_buf = torch.zeros(
                (rows, self.buf_width), dtype=torch.float32, device=device
            )
        return (
            self._eager_idx_buf[: md.num_tokens],
            self._eager_bias_buf[: md.num_tokens],
        )

    def _scores(self, md: UnlimitedOCRV2Metadata, width: int, device) -> torch.Tensor:
        if md.use_decode_twin:
            return self._fixed_scratch_view(
                self._score_buf,
                rows=md.num_tokens,
                width=width,
                owner="decode cuda graph score",
            )
        buf = self._eager_score_buf
        if buf is None or buf.shape[0] < md.num_tokens or buf.shape[1] < width:
            self._no_alloc_while_capturing("eager router score scratch")
            buf = torch.empty(
                max(md.num_tokens, 8),
                max(width, 64),
                dtype=torch.float32,
                device=device,
            )
            self._eager_score_buf = buf
        return buf[: md.num_tokens, :width]

    @staticmethod
    def _fixed_scratch_view(
        buf: Optional[torch.Tensor], *, rows: int, width: int, owner: str
    ) -> torch.Tensor:
        if buf is None or buf.shape[0] < rows or buf.shape[1] < width:
            shape = None if buf is None else tuple(buf.shape)
            raise RuntimeError(
                f"[Unlimited-OCR v2] {owner} capacity {shape} cannot serve "
                f"({rows}, {width})."
            )
        return buf[:rows, :width]

    def _scores_win(self, md: UnlimitedOCRV2Metadata, device) -> Optional[torch.Tensor]:
        if not _ENT_WINDOW:
            return None
        if md.use_decode_twin:
            return self._fixed_scratch_view(
                self._score_buf_win,
                rows=md.num_tokens,
                width=SCAN_TILE,
                owner="decode cuda graph window score",
            )
        buf = self._eager_score_buf_win
        if buf is None or buf.shape[0] < md.num_tokens:
            self._no_alloc_while_capturing("eager router window score scratch")
            buf = torch.empty(
                max(md.num_tokens, 8),
                SCAN_TILE,
                dtype=torch.float32,
                device=device,
            )
            self._eager_score_buf_win = buf
        return buf[: md.num_tokens]

    def _allow_buf(self, md: UnlimitedOCRV2Metadata) -> torch.Tensor:
        """Candidate-mask storage for fused decode metadata."""
        if md.use_decode_twin:
            return self._fixed_scratch_view(
                self._allow_mask,
                rows=md.num_reqs,
                width=md.max_block,
                owner="decode cuda graph candidate mask",
            )
        buf = self._eager_allow_mask
        if buf is None or buf.shape[0] < md.num_reqs or buf.shape[1] < md.max_block:
            self._no_alloc_while_capturing("eager candidate mask buffer")
            buf = torch.zeros(
                max(md.num_reqs, 8),
                max(md.max_block, 64),
                dtype=torch.int8,
                device=self.device,
            )
            self._eager_allow_mask = buf
        return buf[: md.num_reqs, : md.max_block]

    def _kbar_cache(self, layer_id: int) -> torch.Tensor:
        cache = self._kbar[layer_id]
        n_pages = self._index_pages()
        if cache is not None and cache.shape[0] == n_pages:
            return cache
        # The KV pool exists well before capture, so a rebuild means the page count
        # changed; under capture it would strand every later `k_bar[page]` write.
        self._no_alloc_while_capturing(f"head_cls summary cache (layer {layer_id})")
        cache = torch.zeros(
            n_pages, self.idx_head_dim, dtype=torch.float32, device=self.device
        )
        self._kbar[layer_id] = cache
        return cache

    # ------------------------------------------------------------------
    # KV cache views
    # ------------------------------------------------------------------
    def _index_pages(self) -> int:
        buf = self.kv_pool.get_index_k_buffer(0)
        return buf.shape[0] // self.block_size

    def index_kv_view(self, layer_id: int) -> torch.Tensor:
        """Index side cache as ``[num_pages, block_size, idx_head_dim]``."""
        buf = self.kv_pool.get_index_k_buffer(layer_id)
        return buf.view(-1, self.block_size, self.idx_head_dim)

    def main_kv_view(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Main K/V as ``[num_pages, block_size, num_kv_heads, head_dim]``."""
        k, v = self.kv_pool.get_kv_buffer(layer_id)
        heads, dim = k.shape[1], k.shape[2]
        shape = (-1, self.block_size, heads, dim)
        return k.view(shape), v.view(shape)

    # ------------------------------------------------------------------
    # page2d RoPE
    # ------------------------------------------------------------------
    def cos_sin(self, which: str, dtype: torch.dtype):
        """Return this step's main or index cos/sin tables.

        Graph replay uses fixed buffers refreshed eagerly; lazy construction is
        eager-only so warmup cannot omit RoPE kernels from capture.
        """
        md = self.forward_metadata
        if md is None or md.coords is None:
            return None
        rope = self.rope_main if which == "main" else self.rope_index
        key = (rope.signature, md.num_tokens, dtype)
        got = self._cos_sin_cache.get(key)
        if got is None:
            got = rope.cos_sin(md.coords[0], md.coords[1], dtype)
            self._cos_sin_cache[key] = got
        return got

    def _publish_cos_sin_buffers(self, md) -> None:
        n = int(md.coords[0].shape[0])
        for which in ("main", "index"):
            rope = self.rope_main if which == "main" else self.rope_index
            buf = (
                self._p_cos_sin_buf.get(which)
                if md.use_prefill_twin
                else self._cos_sin_buf.get(which)
            )
            if buf is None or buf[0].shape[0] < n:
                continue
            self._cos_sin_cache[(rope.signature, md.num_tokens, self.model_dtype)] = (
                buf[0][:n],
                buf[1][:n],
            )

    def _refresh_cos_sin(self, md) -> None:
        if md.coords is None:
            return
        if self._refresh_recorded_in_graph(md):
            self._publish_cos_sin_buffers(md)
            return
        n = int(md.coords[0].shape[0])
        for which in ("main", "index"):
            rope = self.rope_main if which == "main" else self.rope_index
            # Plain-prefill FULL graph reads cos/sin from the prefill-owned twins.
            buf = (
                self._p_cos_sin_buf.get(which)
                if md.use_prefill_twin
                else self._cos_sin_buf.get(which)
            )
            if buf is not None and buf[0].shape[0] >= n:
                # Buffered: build the table straight into the fixed-address pair.
                cv, sv = cos_sin_2d_fused(
                    rope=rope,
                    tok_pos=md.coords[0],
                    page_pos=md.coords[1],
                    dtype=self.model_dtype,
                    out_cos=buf[0],
                    out_sin=buf[1],
                )
            else:
                # Eager path (or a batch wider than the capture sweep): publish
                # the fresh tensors as-is.
                cv, sv = rope.cos_sin(md.coords[0], md.coords[1], self.model_dtype)
            self._cos_sin_cache[(rope.signature, md.num_tokens, self.model_dtype)] = (
                cv,
                sv,
            )

    def _refresh_router_state(self, md) -> None:
        """Refresh pointer-backed router inputs before graph replay.

        Rebuild plans, page masks, and windows in fixed buffers so warmup
        memoization cannot freeze capture-time contents.
        """
        if md.use_prefill_twin:
            # Materialize pointer-backed inputs before prefill graph replay.
            self._refresh_prefill_state(md)
            return
        if not md.is_decode or md.query_allow is None:
            return
        if not any(r.image_only for r in self.routers):
            return
        tail = 1 if md.decode_query_len <= 1 else 2
        md.kbar_plan = kbar_tail_plan(
            block_table=md.block_table,
            seq_lens=md.seq_lens,
            block_size=self.block_size,
            tail_pages=tail,
            device=self.device,
            out=self._kbar_plan_buf,
        )
        self._build_image_allow_decode(md)
        if _ENT_WINDOW:
            self._build_scan_window(md)

    def _refresh_prefill_state(self, md) -> None:
        """Build plain-prefill router inputs in graph-stable storage."""
        md.kbar_plan = kbar_all_plan(
            block_table=md.block_table,
            seq_lens=md.seq_lens,
            context_lens=md.context_lens,
            block_size=self.block_size,
            max_block=md.max_block,
            max_query_len=md.max_query_len,
            device=self.device,
            out=self._p_kbar_all_plan_buf,
        )
        md.row2req = prefill_row2req(
            cu_seqlens_q=md.cu_seqlens_q,
            device=self.device,
            out=self._p_row2req_buf,
        )
        if self.image_bucket_constraint and md.query_allow is not None:
            self._build_image_allow_prefill(md)

    def _block_cols(self, max_block: int) -> torch.Tensor:
        """Column indices into ``req_to_token`` for every logical page start."""
        # Prefill capture owns this tensor independently of decode capture.
        use_prefill = self._in_prefill_cg
        cols = self._p_bt_cols if use_prefill else self._bt_cols
        if cols is None or cols.shape[0] < max_block:
            self._no_alloc_while_capturing("block-table column index")
            n = max(max_block, self.max_block_static if self._cg_ready else 0)
            cols = torch.arange(n, device=self.device, dtype=torch.long)
            cols = (cols * self.block_size).clamp(max=self.req_to_token.shape[1] - 1)
            if use_prefill:
                self._p_bt_cols = cols
            else:
                self._bt_cols = cols
        return cols[:max_block]

    def _block_table(self, forward_batch: ForwardBatch, max_block: int) -> torch.Tensor:
        """Build the int32 logical-to-physical page table.

        Page-aligned KV slots yield physical pages by integer division. Graph
        paths write fixed-address storage; eager paths allocate.
        """
        cols = self._block_cols(max_block)
        rows = forward_batch.req_pool_indices.long()
        # Flattened gather reads only the requested page starts.
        row_width = self.req_to_token.shape[1]
        linear = rows[:, None] * row_width + cols[None, :]
        gathered = (
            self.req_to_token.reshape(-1)
            .gather(0, linear.reshape(-1))
            .view(rows.shape[0], cols.shape[0])
        )
        table = self._p_bt_buf if self._in_prefill_cg else self._bt_buf
        if (
            table is None
            or table.shape[0] < rows.shape[0]
            or table.shape[1] < max_block
        ):
            return torch.div(gathered, self.block_size, rounding_mode="floor").to(
                torch.int32
            )
        view = table[: rows.shape[0], :max_block]
        torch.div(gathered, self.block_size, rounding_mode="floor", out=view)
        return view

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Build metadata for an eager forward."""
        self._build_metadata(forward_batch, graph_path=False, in_capture=False)

    def init_forward_metadata_out_graph(
        self, forward_batch: ForwardBatch, in_capture: bool = False
    ):
        """Build metadata with graph-owned storage before capture or replay."""
        self._build_metadata(forward_batch, graph_path=True, in_capture=in_capture)

    def _refresh_recorded_in_graph(self, md) -> bool:
        # Only graph-path decode (fixed-address twins) defers its geometry to
        # init_forward_metadata_in_graph; eager decode never runs that hook and
        # must build q_page / coords / query_allow / block table itself.
        return md.is_decode and md.use_decode_twin and not md.use_prefill_twin

    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch):
        """Record decode metadata refresh in dependency order."""
        md = self.forward_metadata
        if md is None or not self._refresh_recorded_in_graph(md):
            return
        # Producers must precede their graph consumers.
        self._block_table(forward_batch, md.max_block)
        self._record_decode_geometry_in_graph(md, forward_batch)
        self._shift_page_coords(md, forward_batch)
        self._record_cos_sin_in_graph(md)
        decode_allow_fused(
            block_table=md.block_table,
            page_state=self.page_state,
            q_page=md.q_page,
            seq_lens=md.seq_lens,
            max_block=md.max_block,
            num_blocks=self._index_pages(),
            block_size=self.block_size,
            out=self._allow_buf(md),
        )
        self._refresh_router_state(md)

    def _bind_graph_geometry(self, md) -> None:
        if self._qpage_buf is None:
            return
        md.q_page = self._qpage_buf[: md.num_reqs]
        md.coords = (
            self._coord_tok_buf[: md.num_reqs],
            self._coord_page_buf[: md.num_reqs],
        )
        md.query_allow = self._allow_buf(md)

    def _record_decode_geometry_in_graph(self, md, forward_batch) -> None:
        decode_meta_fused(
            input_ids=forward_batch.input_ids,
            block_table=md.block_table,
            seq_lens=md.seq_lens,
            slot_mapping=md.slot_mapping,
            page_state=self.page_state,
            coord_state=self.coord_state,
            num_slots=self._index_pages() * self.block_size,
            block_size=self.block_size,
            out_q_page=self._qpage_buf,
            out_tok=self._coord_tok_buf,
            out_page=self._coord_page_buf,
        )

    def _record_cos_sin_in_graph(self, md) -> None:
        if md.coords is None:
            return
        n = self._coord_tok_buf.shape[0] if self._coord_tok_buf is not None else 0
        for which in ("main", "index"):
            rope = self.rope_main if which == "main" else self.rope_index
            buf = (
                self._p_cos_sin_buf.get(which)
                if md.use_prefill_twin
                else self._cos_sin_buf.get(which)
            )
            if buf is None or buf[0].shape[0] < n:
                continue
            cos_sin_2d_fused(
                rope=rope,
                tok_pos=md.coords[0],
                page_pos=md.coords[1],
                dtype=self.model_dtype,
                out_cos=buf[0],
                out_sin=buf[1],
            )

    def _to_static_i32(
        self, buf: Optional[torch.Tensor], src: torch.Tensor, n: int
    ) -> torch.Tensor:
        """Copy ``src[:n]`` into fixed int32 storage, or cast eagerly."""
        if buf is None or buf.shape[0] < n:
            return src[:n].to(torch.int32)
        view = buf[:n]
        view.copy_(src[:n])
        return view

    def _build_metadata(self, forward_batch, *, graph_path: bool, in_capture: bool):
        # Document KV reuse queues copies and state rebuilds for this point so
        # they run on the forward stream before any kernel reads that state.
        if not in_capture and self.doc_reuse is not None:
            self.doc_reuse.run_pending(self)
        if forward_batch.forward_mode.is_idle():
            self.forward_metadata = None
            return
        md = self._new_step_metadata(forward_batch, graph_path=graph_path)
        # An in-graph refresh reads the graph-owned decode inputs directly.
        refresh_in_graph = self._refresh_recorded_in_graph(md)
        self._bind_kv_geometry(md, forward_batch, refresh_in_graph)
        self._bind_query_geometry(md, forward_batch, refresh_in_graph)
        self._bind_page_geometry(md, forward_batch, refresh_in_graph)
        if not refresh_in_graph:
            self._shift_page_coords(md, forward_batch)
        self._cos_sin_cache.clear()
        self.forward_metadata = md
        # Python cache publication stays outside graph replay.
        self._refresh_cos_sin(md)
        if not refresh_in_graph:
            self._refresh_router_state(md)
        # The one-shot diagnostic reads coords back to the host, which capture
        # rejects outright; skip it there (an eager warmup already logged it).
        if not in_capture:
            self._log_first_step(md)

    def _new_step_metadata(
        self, forward_batch, *, graph_path: bool
    ) -> UnlimitedOCRV2Metadata:
        """Classify the step and pick the graph-owned buffer set it may use."""
        md = UnlimitedOCRV2Metadata()
        md.is_decode = forward_batch.forward_mode.is_decode_or_idle()
        md.num_tokens = int(forward_batch.input_ids.shape[0])
        # The full prefill graph's replay view pads input_ids to the captured
        # bucket; only the real tokens own KV slots.
        if forward_batch.out_cache_loc is not None:
            md.num_tokens = min(
                md.num_tokens, int(forward_batch.out_cache_loc.shape[0])
            )
        md.num_reqs = int(forward_batch.seq_lens.shape[0])
        md.use_decode_twin = graph_path and self._cg_ready and md.is_decode
        # Plain-prefill CUDA graph eligibility; such a step runs the FULL prefill
        # graph, so its kernels must read only the fixed-address `_p_*` twins.
        md.use_prefill_twin = bool(
            graph_path
            and self._pcg_ready
            and not md.is_decode
            and md.num_tokens <= self._pcg_max_tokens
            and md.num_reqs <= self._pcg_max_reqs
        )
        if graph_path and not (md.use_decode_twin or md.use_prefill_twin):
            raise RuntimeError(
                "[Unlimited-OCR v2] graph metadata requested for a step outside "
                "the captured routing-buffer geometry."
            )
        if md.use_decode_twin and (
            md.num_tokens > self.cg_max_tokens or md.num_reqs > self.cg_max_bs
        ):
            raise RuntimeError(
                "[Unlimited-OCR v2] decode cuda graph metadata exceeds its "
                "captured token or request capacity."
            )
        self._in_prefill_cg = md.use_prefill_twin
        return md

    def _bind_kv_geometry(
        self, md: UnlimitedOCRV2Metadata, forward_batch, refresh_in_graph: bool
    ) -> None:
        """bind lengths, block-table width and block table for the step"""
        seq_lens_src_buf = (
            self._p_seq_lens_buf if md.use_prefill_twin else self._seq_lens_buf
        )
        if refresh_in_graph:
            md.seq_lens = forward_batch.seq_lens[: md.num_reqs]
        else:
            md.seq_lens = self._to_static_i32(
                seq_lens_src_buf, forward_batch.seq_lens, md.num_reqs
            )
        md.slot_mapping = forward_batch.out_cache_loc
        seq_lens_cpu = forward_batch.seq_lens_cpu
        if refresh_in_graph or seq_lens_cpu is None:
            md.max_seq_len = self.max_context_len
        else:
            md.max_seq_len = int(seq_lens_cpu.max().item())
        if _uses_static_width(
            use_decode_twin=md.use_decode_twin,
            use_prefill_twin=md.use_prefill_twin,
        ):
            # Pin graph width to the engine maximum: grids and entmax reduction
            # order are fixed at capture, while device lengths mask unused columns.
            md.max_block = self.max_block_static
        else:
            md.max_block = max(1, triton.cdiv(md.max_seq_len, self.block_size))
        if refresh_in_graph:
            md.block_table = (
                self._bt_buf[: md.num_reqs, : md.max_block]
                if self._bt_buf is not None
                else self._block_table(forward_batch, md.max_block)
            )
        else:
            md.block_table = self._block_table(forward_batch, md.max_block)

    def _bind_query_geometry(
        self,
        md: UnlimitedOCRV2Metadata,
        forward_batch,
        refresh_in_graph: bool,
    ) -> None:
        """Set query lengths, context lengths and cu_seqlens for the step kind."""
        if md.is_decode:
            md.decode_query_len = max(1, md.num_tokens // max(1, md.num_reqs))
            md.max_query_len = md.decode_query_len
            if refresh_in_graph:
                md.context_lens = None
            else:
                ctx = self._ctx_lens_buf
                if ctx is None or ctx.shape[0] < md.num_reqs:
                    md.context_lens = (md.seq_lens - md.decode_query_len).clamp(min=0)
                else:
                    view = ctx[: md.num_reqs]
                    torch.sub(md.seq_lens, md.decode_query_len, out=view)
                    view.clamp_(min=0)
                    md.context_lens = view
            md.cu_seqlens_q = None
        else:
            md.decode_query_len = 0
            md.max_query_len = int(max(forward_batch.extend_seq_lens_cpu))
            if md.use_prefill_twin:
                # FULL prefill graph twins: its kernels read context_lens /
                # cu_seqlens_q by pointer, so these must be the `_p_*` buffers.
                n = md.num_reqs
                ctx_view = self._p_ctx_lens_buf[:n]
                ctx_view.copy_(forward_batch.extend_prefix_lens)
                md.context_lens = ctx_view
                cu = torch.nn.functional.pad(
                    torch.cumsum(forward_batch.extend_seq_lens, dim=0), (1, 0)
                )
                cu_view = self._p_culen_buf[: n + 1]
                cu_view.copy_(cu)
                md.cu_seqlens_q = cu_view
            else:
                md.context_lens = forward_batch.extend_prefix_lens.to(torch.int32)
                md.cu_seqlens_q = torch.nn.functional.pad(
                    torch.cumsum(forward_batch.extend_seq_lens, dim=0), (1, 0)
                ).to(torch.int32)

    def _bind_page_geometry(
        self, md: UnlimitedOCRV2Metadata, forward_batch, refresh_in_graph: bool
    ) -> None:
        """Set query pages, page2d coordinates and the candidate-page mask."""
        num_slots = self._index_pages() * self.block_size
        fused = (
            md.is_decode and md.decode_query_len == 1 and self.image_bucket_constraint
        )
        if fused:
            # Fuse page IDs and coordinates separately from the candidate mask;
            # both match the PyTorch path.
            md.layout = None
            if refresh_in_graph:
                self._bind_graph_geometry(md)
            else:
                md.q_page, md.coords = decode_meta_fused(
                    input_ids=forward_batch.input_ids,
                    block_table=md.block_table,
                    seq_lens=md.seq_lens,
                    slot_mapping=md.slot_mapping,
                    page_state=self.page_state,
                    coord_state=self.coord_state,
                    num_slots=num_slots,
                    block_size=self.block_size,
                    out_q_page=self._qpage_buf,
                    out_tok=self._coord_tok_buf,
                    out_page=self._coord_page_buf,
                )
            if not refresh_in_graph:
                md.query_allow = decode_allow_fused(
                    block_table=md.block_table,
                    page_state=self.page_state,
                    q_page=md.q_page,
                    seq_lens=md.seq_lens,
                    max_block=md.max_block,
                    num_blocks=self._index_pages(),
                    block_size=self.block_size,
                    out=self._allow_buf(md),
                )
        else:
            md.layout = self._layout(md, forward_batch)
            md.q_page = self.page_state.step(
                forward_batch.input_ids, md.layout, md.slot_mapping, num_slots
            )
            md.query_allow = None
            if self.image_bucket_constraint:
                md.query_allow = self.page_state.build_query_allow(
                    md.layout, md.q_page, md.max_block, self._index_pages()
                )
            md.coords = self.coord_state.step(forward_batch.input_ids, md.layout)

    def note_prompt_lens(
        self, rows: torch.Tensor, prompt_lens: torch.Tensor, page_bases: torch.Tensor
    ) -> None:
        """Record generation start and RoPE page base for the rows of an extend batch."""
        self._gen_start_row[rows] = prompt_lens
        mod = self.rope_main.page_mod
        # Unfolded page axes would leave the trained range; keep windows local.
        self._page_base_row[rows] = (
            page_bases % mod if mod > 0 else torch.zeros_like(page_bases)
        )

    def _shift_page_coords(self, md, forward_batch) -> None:
        """Add each request's global page base to this step's RoPE page coordinates.

        In-place on the coordinate buffer so CUDA graph replay reads the shifted
        values; page isolation and coordinate counting keep window-local pages.
        """
        if md.coords is None:
            return
        rows = forward_batch.req_pool_indices[: md.num_reqs].long()
        base = self._page_base_row[rows]
        page = md.coords[1]
        if md.is_decode and md.decode_query_len == 1:
            per_tok = base
        elif md.is_decode:
            per_tok = base.repeat_interleave(md.decode_query_len)
        else:
            per_tok = base[md.layout.req_of_token]
        per_tok = per_tok[: page.shape[0]]
        # Page 0 holds the prefix and instruction, which have no document page.
        page.add_(torch.where(page > 0, per_tok, torch.zeros_like(per_tok)))

    def _layout(self, md: UnlimitedOCRV2Metadata, forward_batch) -> BatchLayout:
        """build fixed-shape prefill/decode page geometry without host synchronization"""
        device = self.device
        tok = torch.arange(md.num_tokens, device=device, dtype=torch.int64)
        gen_start = None
        if md.is_decode:
            q_start = (
                torch.arange(md.num_reqs + 1, device=device, dtype=torch.int64)
                * md.decode_query_len
            )
            start_pos = md.seq_lens.to(torch.int64) - md.decode_query_len
            req_of_token = tok // md.decode_query_len
        else:
            q_start = md.cu_seqlens_q.to(torch.int64)
            start_pos = md.context_lens.to(torch.int64)
            req_of_token = torch.searchsorted(
                q_start[1:].contiguous(), tok, right=True
            ).clamp_(max=md.num_reqs - 1)
            rows = forward_batch.req_pool_indices[: md.num_reqs].long()
            gen_start = self._gen_start_row[rows]
        return BatchLayout(
            num_tokens=md.num_tokens,
            num_reqs=md.num_reqs,
            q_start=q_start,
            start_pos=start_pos,
            block_table=md.block_table,
            req_of_token=req_of_token,
            num_decode_tokens=md.num_tokens if md.is_decode else 0,
            gen_start=gen_start,
        )

    def _log_first_step(self, md: UnlimitedOCRV2Metadata) -> None:
        if self._logged_first or md.coords is None:
            return
        self._logged_first = True
        pages = int(md.coords[1].max().item()) + 1
        logger.info(
            "[Unlimited-OCR v2] first step: tokens=%d reqs=%d pages_seen=%d "
            "tok_pos_max=%d max_block=%d",
            md.num_tokens,
            md.num_reqs,
            pages,
            int(md.coords[0].max().item()),
            md.max_block,
        )
        if pages <= 1:
            logger.warning(
                "[Unlimited-OCR v2] this prompt carries no <PAGE> marker (id %s): "
                "every token lands in the reserved page-0 bucket, so the page "
                "axis of the RoPE is inert and the coordinates differ from "
                "training. The trained format is one bare <PAGE> before every "
                "<image>, e.g. '<PAGE><image><PAGE><image>Multi page parsing.'",
                self.page_state.page_token_id,
            )
        n_img = self.page_state._last_img_run_max
        n_img = int(n_img.item()) if n_img is not None else 0
        per = self.page_state.per_page_image_tokens
        # Marker mode derives image pages independently of this fallback width.
        if pages <= 1 and n_img and n_img % per != 0:
            logger.warning(
                "[Unlimited-OCR v2] %d image tokens is not a multiple of "
                "per_page_image_tokens=%d, and this prompt has no <PAGE> "
                "markers -> image pages are mis-split. Add the markers, or "
                "serve the crop mode the checkpoint was trained on.",
                n_img,
                per,
            )

    # ------------------------------------------------------------------
    # Routing (entmax) -- called once per layer per step
    # ------------------------------------------------------------------
    def _build_image_allow_decode(self, md: UnlimitedOCRV2Metadata) -> None:
        """Fill ``_img_allow_buf`` and publish it as ``md.image_allow``."""
        buf = self._img_allow_buf
        if (
            md.decode_query_len == 1
            and self.local_blocks > 0
            and buf is not None
            and buf.shape[0] >= md.num_reqs
            and buf.shape[1] >= md.max_block
        ):
            # Union the local window and the page mask in one kernel, writing
            # straight into the fixed-address buffer the graph reads.
            md.image_allow = local_union_fused(
                allow=md.query_allow,
                seq_lens=md.seq_lens,
                out=buf,
                max_block=md.max_block,
                block_size=self.block_size,
                local_blocks=self.local_blocks,
            )
            return
        # Built in a bool scratch, published int8 into the fixed-address buffer:
        # the router reads this by pointer, so a graph must not see a fresh alloc.
        allow = _fit_width(md.query_allow, md.max_block).clone()
        if allow.dtype != torch.bool:
            allow = allow != 0
        if self.local_blocks > 0:
            off = torch.arange(md.decode_query_len, device=self.device)
            pos = (
                md.seq_lens.to(torch.int64)[:, None]
                - md.decode_query_len
                + off[None, :]
            ).reshape(-1)
            q_blk = pos // self.block_size
            loc = torch.arange(self.local_blocks, device=self.device)
            local_idx = (q_blk[:, None] - loc[None, :]).clamp(
                min=0, max=md.max_block - 1
            )
            allow.scatter_(1, local_idx.to(torch.int64), True)
        rows = allow.shape[0]
        if buf is None or buf.shape[0] < rows or buf.shape[1] < md.max_block:
            md.image_allow = allow.to(torch.int8)
        else:
            view = buf[:rows, : md.max_block]
            view.copy_(allow)
            md.image_allow = view

    def _image_allow_decode(self, md: UnlimitedOCRV2Metadata) -> Optional[torch.Tensor]:
        if md.query_allow is None:
            return None
        if md.image_allow is None:
            self._build_image_allow_decode(md)
        return md.image_allow

    def _build_image_allow_prefill(self, md: UnlimitedOCRV2Metadata) -> None:
        """Publish the prefill candidate mask in graph-stable storage."""
        mask = _fit_width(md.query_allow, md.max_block).to(torch.int8)
        buf = self._p_img_allow_buf if self._in_prefill_cg else self._img_allow_buf
        rows = mask.shape[0]
        if buf is None or buf.shape[0] < rows or buf.shape[1] < md.max_block:
            md.image_allow = mask
            return
        view = buf[:rows, : md.max_block]
        view.copy_(mask)
        md.image_allow = view

    def _image_allow_prefill(
        self, md: UnlimitedOCRV2Metadata
    ) -> Optional[torch.Tensor]:
        if md.query_allow is None:
            return None
        if md.image_allow is None:
            self._build_image_allow_prefill(md)
        return md.image_allow

    def _build_scan_window(self, md: UnlimitedOCRV2Metadata) -> None:
        """Build a remote scan window from the page mask."""
        nd = md.num_tokens
        bufs = self._win_bufs
        if bufs is None or bufs[0].shape[0] < nd:
            lo = torch.zeros(nd, dtype=torch.int32, device=self.device)
            hi = torch.zeros(nd, dtype=torch.int32, device=self.device)
            fit = torch.zeros(nd, dtype=torch.int8, device=self.device)
        else:
            # The router reads all three by pointer, so they come from the
            # fixed-address set under cuda graphs.
            lo, hi, fit = (b[:nd] for b in bufs)
            lo.zero_()
            hi.zero_()
            fit.zero_()
        win_bounds_decode(
            allow=md.query_allow,
            seq_lens=md.seq_lens,
            lo=lo,
            hi=hi,
            fit=fit,
            max_block=md.max_block,
            block_size=self.block_size,
            local_blocks=self.local_blocks,
            decode_query_len=md.decode_query_len,
            tile=SCAN_TILE,
        )
        md.win = (lo, hi, fit)

    def _scan_window(self, md: UnlimitedOCRV2Metadata):
        if md.query_allow is None or not _ENT_WINDOW:
            return None
        if md.win is None:
            self._build_scan_window(md)
        return md.win

    def _route(
        self, layer_id: int, idx_q: torch.Tensor, head_cls: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """Route each query to indices, biases, and output width.

        Single-head rows expand over KV heads through stride-zero views.
        """
        md = self.forward_metadata
        router = self.routers[layer_id]
        kbar = self._kbar_cache(layer_id)
        kv = self.index_kv_view(layer_id)
        idx_buf, bias_buf = self._routing_buffers(md, idx_q.device)
        route = self._route_decode if md.is_decode else self._route_prefill
        width = route(
            md,
            router=router,
            k_bar=kbar,
            kv=kv,
            idx_q=idx_q,
            head_cls=head_cls,
            idx_buf=idx_buf,
            bias_buf=bias_buf,
        )
        # The kept blocks are packed at the front, so advertising min(width, W)
        # is the same read set with less attend work.
        width = max(1, min(int(width), md.max_block))
        return idx_buf[:, :width], bias_buf[:, :width], width

    def _route_decode(
        self, md, *, router, k_bar, kv, idx_q, head_cls, idx_buf, bias_buf
    ):
        """Refresh decode-tail summaries and route decode queries; return width."""
        if md.kbar_plan is None:
            tail = 1 if md.decode_query_len <= 1 else 2
            md.kbar_plan = kbar_tail_plan(
                block_table=md.block_table,
                seq_lens=md.seq_lens,
                block_size=self.block_size,
                tail_pages=tail,
                device=self.device,
                out=self._kbar_plan_buf,
            )
        kbar_refresh_tail_triton(
            kv_cache=kv,
            k_bar=k_bar,
            block_table=md.block_table,
            seq_lens=md.seq_lens,
            head_cls=head_cls,
            block_size=self.block_size,
            plan=md.kbar_plan,
        )
        img = self._image_allow_decode(md) if router.image_only else None
        win = self._scan_window(md) if img is not None else None
        return route_decode_triton(
            idx_q=idx_q,
            k_bar=k_bar,
            block_table=md.block_table,
            seq_lens=md.seq_lens,
            image_allow=img,
            out_idx=idx_buf,
            out_bias=bias_buf,
            max_block=md.max_block,
            block_size=self.block_size,
            local_blocks=self.local_blocks,
            decode_query_len=md.decode_query_len,
            idx_scale=router.idx_scale,
            prior_floor=router.prior_floor,
            sigma=router.sigma,
            topm=router.topm,
            score_buf=self._scores(md, md.max_block, idx_q.device),
            budget=router.budget_remote,
            out_slots=router.out_slots,
            bounds=None if win is None else win[:2],
            entmax_base=None if win is None else win[0],
            entmax_fit=None if win is None else win[2],
            entmax_width=SCAN_TILE if _ENT_WINDOW else 0,
            score_buf_win=(
                self._scores_win(md, idx_q.device) if win is not None else None
            ),
        )

    def _route_prefill(
        self, md, *, router, k_bar, kv, idx_q, head_cls, idx_buf, bias_buf
    ):
        """refresh summaries and route prefill queries; return width"""
        if md.kbar_plan is None:
            # full prefill graphs own their routing storage; eager prefill allocates
            if md.use_prefill_twin:
                twins = self._p_kbar_all_plan_buf
                row2req_out = self._p_row2req_buf
            else:
                twins = None
                row2req_out = None
            md.kbar_plan = kbar_all_plan(
                block_table=md.block_table,
                seq_lens=md.seq_lens,
                context_lens=md.context_lens,
                block_size=self.block_size,
                max_block=md.max_block,
                max_query_len=md.max_query_len,
                device=self.device,
                out=twins,
            )
            md.row2req = prefill_row2req(
                cu_seqlens_q=md.cu_seqlens_q,
                device=self.device,
                out=row2req_out,
            )
        kbar_refresh_tail_triton(
            kv_cache=kv,
            k_bar=k_bar,
            block_table=md.block_table,
            seq_lens=md.seq_lens,
            head_cls=head_cls,
            block_size=self.block_size,
            plan=md.kbar_plan,
        )
        img = self._image_allow_prefill(md) if router.image_only else None
        width, _, _ = route_prefill_triton(
            idx_q=idx_q,
            k_bar=k_bar,
            block_table=md.block_table,
            seq_lens=md.seq_lens,
            cu_seqlens_q=md.cu_seqlens_q,
            context_lens=md.context_lens,
            row2req=md.row2req,
            image_allow=img,
            out_idx=idx_buf,
            out_bias=bias_buf,
            max_block=md.max_block,
            block_size=self.block_size,
            local_blocks=self.local_blocks,
            idx_scale=router.idx_scale,
            prior_floor=router.prior_floor,
            sigma=router.sigma,
            topm=router.topm,
            budget=router.budget_remote,
            q_chunk=self.q_chunk,
            out_slots=router.out_slots,
            score_buf=self._prefill_score_scratch(md),
        )
        return width

    def _prefill_score_scratch(self, md) -> Optional[torch.Tensor]:
        """Graph-owned prefill router score scratch, or None to allocate."""
        if md.use_prefill_twin and self._p_score_buf is not None:
            return self._p_score_buf
        return None

    # ------------------------------------------------------------------
    # Attend
    # ------------------------------------------------------------------
    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        """Prefill/extend entry point; routes to the shared DASH attend path."""
        return self._forward(q, k, v, layer, forward_batch, save_kv_cache, kwargs)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        """Decode entry point; routes to the shared DASH attend path."""
        return self._forward(q, k, v, layer, forward_batch, save_kv_cache, kwargs)

    def _forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool,
        kwargs: dict,
    ):
        idx_q = kwargs["idx_q"]
        idx_k = kwargs["idx_k"]
        head_cls = kwargs["head_cls"]
        layer_id = layer.layer_id
        if layer_id not in self._uocr_head_cls:
            self._uocr_head_cls[layer_id] = head_cls
        md = self.forward_metadata
        num_tokens = q.shape[0]
        out = q.new_empty((num_tokens, layer.tp_q_head_num * layer.v_head_dim))

        if md is None:
            return out.zero_()

        if save_kv_cache:
            self.kv_pool.set_kv_buffer(layer, forward_batch.out_cache_loc, k, v)
            self.kv_pool.set_index_k_buffer(layer, forward_batch.out_cache_loc, idx_k)

        topk_idx, blk_bias, _ = self._route(layer_id, idx_q, head_cls)
        num_kv_heads = layer.tp_k_head_num
        # Single-head router -> stride-0 view over the kv heads.
        di = topk_idx.unsqueeze(0).expand(num_kv_heads, -1, -1)
        db = blk_bias.unsqueeze(0).expand(num_kv_heads, -1, -1)
        k_cache, v_cache = self.main_kv_view(layer_id)
        q_view = q.view(num_tokens, layer.tp_q_head_num, layer.head_dim)
        out_view = out.view(num_tokens, layer.tp_q_head_num, layer.v_head_dim)

        if md.is_decode:
            dash_sparse_attn_decode(
                q_view,
                k_cache,
                v_cache,
                di,
                db,
                md.block_table,
                md.seq_lens,
                num_kv_heads,
                layer.scaling,
                out_view,
                md.decode_query_len,
                local_blocks=self.local_blocks,
                sparse_block_size=self.block_size,
                # Narrowing the emitted slots must not change the split-K
                # partition (it decides the accumulation order).
                chunk_width=min(self.topm, self.buf_width),
                # Allocate split-K scratch at maximum row/chunk geometry before
                # capture; shrinking batches may need more chunks.
                cg_rows=self.cg_max_bs if md.use_decode_twin else None,
            )
        else:
            dash_sparse_attn(
                q_view,
                k_cache,
                v_cache,
                di,
                db,
                md.block_table,
                md.cu_seqlens_q,
                md.seq_lens,
                md.context_lens,
                md.max_query_len,
                num_kv_heads,
                layer.scaling,
                out_view,
                local_blocks=self.local_blocks,
                sparse_block_size=self.block_size,
            )
        return out

    # ------------------------------------------------------------------
    # CUDA graph
    # ------------------------------------------------------------------
    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        """Preallocate cross-step graph buffers at maximum geometry.

        Metadata updates eagerly, but captured addresses and grids are immutable.
        Use ``max_block_static`` and reject any missed allocation during capture.
        """
        self.cg_max_bs = int(max_bs)
        self.cg_max_tokens = int(max_num_tokens)
        w = self.max_block_static
        dev = self.device

        self._bt_cols = None
        self._bt_buf = torch.zeros((self.cg_max_bs, w), dtype=torch.int32, device=dev)
        # Causal bounds use fixed int32 storage so captured pointers cannot retain
        # the padding sentinel.
        self._seq_lens_buf = torch.zeros(
            (self.cg_max_bs,), dtype=torch.int32, device=dev
        )
        self._ctx_lens_buf = torch.zeros(
            (self.cg_max_bs,), dtype=torch.int32, device=dev
        )
        self._idx_buf = torch.full(
            (self.cg_max_tokens, self.buf_width), -1, dtype=torch.int32, device=dev
        )
        self._bias_buf = torch.zeros(
            (self.cg_max_tokens, self.buf_width), dtype=torch.float32, device=dev
        )
        self._score_buf = torch.empty(
            (self.cg_max_tokens, w), dtype=torch.float32, device=dev
        )
        self._score_buf_win = (
            torch.empty(
                (self.cg_max_tokens, SCAN_TILE), dtype=torch.float32, device=dev
            )
            if _ENT_WINDOW
            else None
        )
        self._allow_mask = torch.zeros(
            (self.cg_max_tokens, w), dtype=torch.int8, device=dev
        )
        # `_image_allow_decode` unions the local window into a COPY of query_allow;
        # the router reads that copy by pointer, so it needs a stable address.
        self._img_allow_buf = torch.zeros(
            (self.cg_max_tokens, w), dtype=torch.int8, device=dev
        )
        # decode_meta_fused's outputs: page id per query, and the (tok_pos,
        # page_pos) pair the rope kernels read by pointer.
        self._qpage_buf = torch.zeros((self.cg_max_bs,), dtype=torch.int32, device=dev)
        self._coord_tok_buf = torch.zeros(
            (self.cg_max_tokens,), dtype=torch.int64, device=dev
        )
        self._coord_page_buf = torch.zeros(
            (self.cg_max_tokens,), dtype=torch.int64, device=dev
        )
        # kbar tail-refresh plan: tail_pages is at most 2 per request.
        self._kbar_plan_buf = (
            torch.zeros((self.cg_max_bs * 2,), dtype=torch.int64, device=dev),
            torch.zeros((self.cg_max_bs * 2,), dtype=torch.int32, device=dev),
        )
        self._win_bufs = (
            torch.zeros((self.cg_max_tokens,), dtype=torch.int32, device=dev),
            torch.zeros((self.cg_max_tokens,), dtype=torch.int32, device=dev),
            torch.zeros((self.cg_max_tokens,), dtype=torch.int8, device=dev),
        )
        # page2d cos/sin, one pair per branch (main head_dim, index
        # sparse_index_dim); `_refresh_cos_sin` refills them eagerly before replay.
        self._cos_sin_buf = self._alloc_cos_sin_pairs(self.cg_max_tokens, dev)
        # Allocate summaries against the final KV-pool size before capture.
        for lid in range(self.num_layers):
            self._kbar_cache(lid)
        self._cg_ready = True
        logger.info(
            "[Unlimited-OCR v2] cuda graph buffers: max_bs=%d max_tokens=%d "
            "block_table_width=%d (context_len=%d / block=%d) scan_tile=%d",
            self.cg_max_bs,
            self.cg_max_tokens,
            w,
            self.max_context_len,
            self.block_size,
            SCAN_TILE if _ENT_WINDOW else 0,
        )

    def _alloc_cos_sin_pairs(self, rows: int, dev) -> dict:
        """Zeroed (cos, sin) storage per rope branch, ``rows`` tokens each."""
        return {
            which: tuple(
                torch.zeros((rows, rope.dim), dtype=self.model_dtype, device=dev)
                for _ in range(2)
            )
            for which, rope in (("main", self.rope_main), ("index", self.rope_index))
        }

    def get_cuda_graph_seq_len_fill_value(self):
        """Padding fill value for CUDA-graph seq_lens rows.

        Padded rows get req_pool_indices = 0 and out_cache_loc = 0, so a length
        of 1 makes them read the reserved page-0 sink and write slot 0.
        """
        return 1

    def init_prefill_cuda_graph_state(self, max_num_tokens: int, max_req_slots: int):
        """Preallocate fixed storage for full-prefill graphs.

        Prefill and decode storage have separate lifetimes. Size token and
        request axes by their capture maxima; over-capacity prefill is eager.
        """
        self._pcg_max_tokens = int(max_num_tokens)
        self._pcg_max_reqs = int(max_req_slots)
        w = self.max_block_static
        dev = self.device
        t = self._pcg_max_tokens
        r = self._pcg_max_reqs

        # Geometry tensors (request axis).
        self._p_seq_lens_buf = torch.zeros((r,), dtype=torch.int32, device=dev)
        self._p_ctx_lens_buf = torch.zeros((r,), dtype=torch.int32, device=dev)
        self._p_culen_buf = torch.zeros((r + 1,), dtype=torch.int32, device=dev)
        self._p_bt_buf = torch.zeros((r, w), dtype=torch.int32, device=dev)
        # kbar_all_plan / prefill_row2req twins. plan rows = R * (w + 1) bounds
        # width <= cdiv(max_query_len, block) + 1; row2req spans the token axis.
        plan_len = r * (w + 1)
        self._p_kbar_all_plan_buf = (
            torch.zeros((plan_len,), dtype=torch.int64, device=dev),
            torch.zeros((plan_len,), dtype=torch.int32, device=dev),
        )
        self._p_row2req_buf = torch.zeros((t,), dtype=torch.int32, device=dev)
        # Router output + score scratch (token axis).
        self._p_idx_buf = torch.full(
            (t, self.buf_width), -1, dtype=torch.int32, device=dev
        )
        self._p_bias_buf = torch.zeros(
            (t, self.buf_width), dtype=torch.float32, device=dev
        )
        self._p_score_buf = torch.empty(
            (min(self.q_chunk, t), w), dtype=torch.float32, device=dev
        )
        self._p_img_allow_buf = torch.zeros((t, w), dtype=torch.int8, device=dev)
        # page2d cos/sin (token axis), one pair per branch.
        self._p_cos_sin_buf = self._alloc_cos_sin_pairs(t, dev)
        self._p_bt_cols = None  # lazily built column index, prefill-owned
        self._pcg_ready = True
        logger.info(
            "[Unlimited-OCR v2] prefill cuda graph twins: max_tokens=%d req_slots=%d "
            "block_table_width=%d",
            t,
            r,
            w,
        )


def _fit_width(mask: torch.Tensor, width: int) -> torch.Tensor:
    """Pad / trim a mask's trailing dim to exactly ``width``."""
    w = int(mask.shape[-1])
    if w < width:
        return torch.nn.functional.pad(mask, (0, width - w), value=False)
    return mask[..., :width]


def _rope_from_cfg(cfg: dict, dim: int, which: str) -> Optional[PageRope2D]:
    """Build the page2d rotary for the ``main`` / ``index`` branch, None for 1D."""
    key = "sparse_main_rope_mode" if which == "main" else "sparse_rope_mode"
    if str(cfg.get(key, "1d")).lower() != "page2d":
        return None
    span = int(cfg.get("sparse_rope_page_span", 0) or 0)
    if span > 0:
        raise NotImplementedError(
            f"sparse_rope_page_span={span} (span-normalised token axis) is not "
            "implemented; the checkpoint would run with wrong token coordinates."
        )
    # Token axis theta is sparse_rope_tok_theta, NOT rope_theta: the yarn base
    # (5e5) is sized for 128k global positions while tok_pos spans one page.
    tok_theta = float(cfg.get("sparse_rope_tok_theta", 10000.0))
    page_theta = float(cfg.get("sparse_rope_page_theta", 256.0))
    page_pairs = int(cfg.get("sparse_rope_page_pairs", 8))
    page_mod = int(cfg.get("sparse_rope_page_mod", 0) or 0)
    freq_key = (
        "sparse_main_rope_tok_freq_mode"
        if which == "main"
        else "sparse_rope_tok_freq_mode"
    )
    # Training defaults differ per branch: main "truncate", index "resample".
    default_mode = "truncate" if which == "main" else "resample"
    return PageRope2D(
        dim,
        tok_theta,
        page_theta,
        page_pairs,
        tok_freq_mode=str(cfg.get(freq_key, default_mode)),
        page_mod=page_mod,
    )
