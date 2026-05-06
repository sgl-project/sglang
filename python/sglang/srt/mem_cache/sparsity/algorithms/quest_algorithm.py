"""
Quest sparse attention algorithm.

This implementation follows the Quest paper's bounding-box estimation for
query-aware page selection.  For each KV page it maintains per-dimension
min/max of keys and uses them to upper-bound attention scores without
materializing full dot products.

Designed to be driven directly by :class:`HiSparseCoordinator` in its
``quest`` mode, mirroring the NSA-hisparse hot path:

  forward(layer):
      topk = quest.retrieve_topk(q, layer_id, req_pool_indices, seq_lens)
      page_table = coord.swap_in_selected_pages(... topk ...)
      attention(q, kv, page_table)

  prepare_for_decode (between forwards, on backup stream):
      coord._eager_backup_previous_token(...)
        → quest.update_decode_representations(layer, req_indices, k_loc) per layer
        → quest.maybe_finalize_decode_representations(req_indices) once all layers done
      coord.map_last_loc_to_buffer(...)

  admit_request_into_staging:
      quest.update_prefill_representations(layer, req_idx, k_buffer, prefill_indices) per layer

The orphan ``SparseCoordinator`` / ``BaseSparseAlgorithm`` interfaces are
intentionally NOT used; this class is standalone.
"""

import logging
import os
from typing import Optional, Tuple

import torch

from sglang.srt.mem_cache.sparsity.algorithms.quest_decode_bounds_kernel import (
    quest_decode_bounds,
)
from sglang.srt.mem_cache.sparsity.algorithms.quest_prefill_bounds_kernel import (
    quest_prefill_bounds,
)
from sglang.srt.mem_cache.sparsity.algorithms.quest_score_kernel import quest_score

logger = logging.getLogger(__name__)

# Toggle for the Triton fused score kernel.  Default on; set
# QUEST_DISABLE_TRITON_SCORE=1 to fall back to the PyTorch op chain (for
# comparison + debug).
_USE_TRITON_SCORE = os.environ.get("QUEST_DISABLE_TRITON_SCORE", "0") != "1"

# Toggle for the fused all-layer prefill-bounds kernel.  Default on; set
# QUEST_DISABLE_FUSED_PREFILL_BOUNDS=1 to fall back to the per-layer Python
# loop (for parity testing).
_USE_FUSED_PREFILL_BOUNDS = (
    os.environ.get("QUEST_DISABLE_FUSED_PREFILL_BOUNDS", "0") != "1"
)

# Toggle for the fused all-layer decode-bounds kernel.  Default on; set
# QUEST_DISABLE_FUSED_DECODE_BOUNDS=1 to fall back to the per-layer Python
# loop.
_USE_FUSED_DECODE_BOUNDS = (
    os.environ.get("QUEST_DISABLE_FUSED_DECODE_BOUNDS", "0") != "1"
)

# Sentinels for running min/max so the first observed K initialises correctly.
# Using bf16 finfo because the running buffers themselves are bf16.
_BF16_POS_INF = torch.finfo(torch.bfloat16).max
_BF16_NEG_INF = torch.finfo(torch.bfloat16).min


class QuestAlgorithm:
    """Quest page-wise sparse attention with bounds storage indexed by
    (layer, req_pool_idx, logical_page_in_req).

    Storage budget (bf16):
      * page bounds: ``2 × num_layers × max_reqs × max_pages_per_req × kv_heads × head_dim``
      * running bounds (one in-flight page per req): ``2 × num_layers × max_reqs × kv_heads × head_dim``
    """

    def __init__(self, top_k: int, page_size: int, device: torch.device):
        if page_size <= 0:
            raise ValueError(f"quest page_size must be > 0, got {page_size}")
        if top_k % page_size != 0:
            raise ValueError(
                f"top_k ({top_k}) must be divisible by quest_page_size ({page_size}); "
                "Quest emits whole pages worth of token positions."
            )
        self.top_k = top_k
        self.page_size = page_size
        self.top_k_pages = top_k // page_size
        self.device = device

        # Set in init_storage()
        self.start_layer: Optional[int] = None
        self.end_layer: Optional[int] = None
        self.num_layers: Optional[int] = None
        self.max_reqs: Optional[int] = None
        self.max_pages_per_req: Optional[int] = None
        self.kv_heads: Optional[int] = None
        self.head_dim: Optional[int] = None

        # Per-page bounds, keyed by (layer_offset, req, page_in_req).
        # Combined into ``page_k_bounds`` (last axis indexes [min, max])
        # so a single gather brings both during retrieve_topk.  Invalid
        # pages carry sentinel values (+inf for min, -inf for max) so the
        # Quest score formula naturally yields -inf — no separate valid bit.
        self.page_k_bounds: Optional[torch.Tensor] = None
        self.page_k_min: Optional[torch.Tensor] = None  # view of page_k_bounds[..., 0]
        self.page_k_max: Optional[torch.Tensor] = None  # view of page_k_bounds[..., 1]

        # Step-level cached state, populated by ``prepare_step`` (called once
        # per decode step by the backend) and read by every layer's
        # ``retrieve_topk``.  Avoids ~6 redundant ops per layer in the
        # captured graph (~14 ms saved at 48 layers in the bench config).
        self._step_last_valid_buf: Optional[torch.Tensor] = None
        self._step_recent_positions_buf: Optional[torch.Tensor] = None
        self._step_short_layout_buf: Optional[torch.Tensor] = None
        self._step_is_short_buf: Optional[torch.Tensor] = None
        self._step_actual_lens_buf: Optional[torch.Tensor] = None
        # Constant arange tensors used by prepare_step.
        self._recent_offsets_const: Optional[torch.Tensor] = None
        self._all_positions_const: Optional[torch.Tensor] = None
        self._page_offsets_const: Optional[torch.Tensor] = None

        # Running min/max for the page currently being filled by decode.
        self.running_k_min: Optional[torch.Tensor] = None
        self.running_k_max: Optional[torch.Tensor] = None
        # Per-request scalars on device (so coordinator can update batched).
        self.running_token_count: Optional[torch.Tensor] = None
        self.running_page_idx: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------ setup

    def init_storage(
        self,
        start_layer: int,
        end_layer: int,
        max_reqs: int,
        max_context_len: int,
        kv_heads: int,
        head_dim: int,
    ) -> None:
        num_layers = end_layer - start_layer
        max_pages_per_req = (max_context_len + self.page_size - 1) // self.page_size

        self.start_layer = start_layer
        self.end_layer = end_layer
        self.num_layers = num_layers
        self.max_reqs = max_reqs
        self.max_pages_per_req = max_pages_per_req
        self.kv_heads = kv_heads
        self.head_dim = head_dim

        # Combined per-page bounds tensor: last axis indexes [min, max].
        # Sentinel-initialised: invalid pages have min=+inf, max=-inf so the
        # Quest score formula (which picks q*max for q≥0 and q*min for q<0)
        # naturally yields -inf for them.  This drops the separate page_valid
        # tensor and the in_range mask in retrieve_topk — fewer kernels per
        # layer (cuda-graph-replay overhead is what limits Mode 2 perf).
        bounds_shape = (num_layers, max_reqs, max_pages_per_req, kv_heads, head_dim, 2)
        self.page_k_bounds = torch.empty(
            bounds_shape, dtype=torch.bfloat16, device=self.device,
        )
        # Initialise: [..., 0] = +inf (min sentinel), [..., 1] = -inf (max sentinel).
        self.page_k_bounds[..., 0].fill_(_BF16_POS_INF)
        self.page_k_bounds[..., 1].fill_(_BF16_NEG_INF)
        # Convenience views (no-op slices, no kernel work).  Read-only by
        # convention; assignments must go through ``_write_page_bounds``.
        self.page_k_min = self.page_k_bounds[..., 0]
        self.page_k_max = self.page_k_bounds[..., 1]

        running_shape = (num_layers, max_reqs, kv_heads, head_dim)
        self.running_k_min = torch.full(
            running_shape, _BF16_POS_INF, dtype=torch.bfloat16, device=self.device
        )
        self.running_k_max = torch.full(
            running_shape, _BF16_NEG_INF, dtype=torch.bfloat16, device=self.device
        )
        self.running_token_count = torch.zeros(
            max_reqs, dtype=torch.int32, device=self.device
        )
        self.running_page_idx = torch.zeros(
            max_reqs, dtype=torch.int32, device=self.device
        )

        # Total static memory:
        bounds_mb = (
            bounds_shape[0] * bounds_shape[1] * bounds_shape[2]
            * bounds_shape[3] * bounds_shape[4] * bounds_shape[5] * 2
        ) / (1024 ** 2)
        running_mb = (
            2 * running_shape[0] * running_shape[1] * running_shape[2]
            * running_shape[3] * 2
        ) / (1024 ** 2)
        logger.info(
            "Quest storage initialised: %d layers × %d reqs × %d pages/req × "
            "%d KV heads × %d dim (page=%d, top_k=%d) → bounds %.1f MB, running %.1f MB",
            num_layers, max_reqs, max_pages_per_req, kv_heads, head_dim,
            self.page_size, self.top_k, bounds_mb, running_mb,
        )

        # Step-level state buffers (sized for max_reqs as upper bound on bs).
        self._step_last_valid_buf = torch.zeros(
            (max_reqs, 1), dtype=torch.int64, device=self.device,
        )
        self._step_recent_positions_buf = torch.zeros(
            (max_reqs, self.page_size), dtype=torch.int32, device=self.device,
        )
        self._step_short_layout_buf = torch.zeros(
            (max_reqs, self.top_k), dtype=torch.int32, device=self.device,
        )
        self._step_is_short_buf = torch.zeros(
            (max_reqs, 1), dtype=torch.bool, device=self.device,
        )
        self._step_actual_lens_buf = torch.zeros(
            (max_reqs,), dtype=torch.int32, device=self.device,
        )
        # Constants used by prepare_step + retrieve_topk.
        self._recent_offsets_const = torch.arange(
            self.page_size, device=self.device, dtype=torch.int64,
        ).unsqueeze(0)  # [1, page_size]
        self._all_positions_const = torch.arange(
            self.top_k, device=self.device, dtype=torch.int64,
        ).unsqueeze(0)  # [1, top_k]
        self._page_offsets_const = torch.arange(
            self.page_size, device=self.device, dtype=torch.int64,
        ).view(1, 1, -1)  # [1, 1, page_size]
        self._page_idx_const = torch.arange(
            self.max_pages_per_req, device=self.device, dtype=torch.int64,
        )  # [max_pages_per_req] — used by retrieve_topk to mask OOB pages

    # ---------------------------------------------------------- prefill update

    def update_prefill_representations(
        self,
        layer_id: int,
        req_pool_idx: int,
        k_buffer: torch.Tensor,
        prefill_indices: torch.Tensor,
    ) -> None:
        """Compute bounds for ``req``'s prefill K and seed running buffers.

        Args:
          layer_id: model layer (absolute).  Will be offset by ``start_layer``.
          req_pool_idx: integer request slot.
          k_buffer: ``[pool_size, kv_heads, head_dim]`` the layer's K buffer.
          prefill_indices: ``[prefill_len]`` device addresses for the
            request's prefill tokens, in token order.

        Only fully-valid pages contribute to ``page_k_min/max``; the partial
        last page (if any) seeds ``running_k_min/max`` so decode picks up
        where prefill left off.

        Per-request scalar counters (``running_token_count``,
        ``running_page_idx``) are only set on the last layer call to avoid
        multiple writes; the coordinator must invoke this method for every
        layer in ``[start_layer, end_layer)``.
        """
        layer_offset = layer_id - self.start_layer
        prefill_len = int(prefill_indices.shape[0])

        num_full_pages = prefill_len // self.page_size
        if num_full_pages > 0:
            full_count = num_full_pages * self.page_size
            full_indices = prefill_indices[:full_count]
            full_k = k_buffer[full_indices]
            paged = full_k.view(
                num_full_pages, self.page_size, self.kv_heads, self.head_dim
            )
            self.page_k_min[layer_offset, req_pool_idx, :num_full_pages] = (
                paged.amin(dim=1).to(torch.bfloat16)
            )
            self.page_k_max[layer_offset, req_pool_idx, :num_full_pages] = (
                paged.amax(dim=1).to(torch.bfloat16)
            )

        partial_count = prefill_len - num_full_pages * self.page_size
        if partial_count > 0:
            partial_indices = prefill_indices[num_full_pages * self.page_size :]
            partial_k = k_buffer[partial_indices]
            self.running_k_min[layer_offset, req_pool_idx] = (
                partial_k.amin(dim=0).to(torch.bfloat16)
            )
            self.running_k_max[layer_offset, req_pool_idx] = (
                partial_k.amax(dim=0).to(torch.bfloat16)
            )
        # else: running buffers remain at (+inf, -inf), correct for an empty page.

        if layer_id == self.end_layer - 1:
            # Per-request scalars: written once per admit, on the last layer.
            self.running_token_count[req_pool_idx] = partial_count
            self.running_page_idx[req_pool_idx] = num_full_pages

    def update_prefill_representations_fused(
        self,
        req_pool_idx: int,
        k_data_ptrs: torch.Tensor,
        prefill_indices: torch.Tensor,
    ) -> None:
        """All-layer fused variant of :meth:`update_prefill_representations`.

        Computes per-page K bounds for ALL Quest layers in ONE Triton kernel
        launch, replacing num_layers iterations of the per-layer Python op
        chain (gather + reshape + amin/amax + cast + slice-store).  Used by
        the hisparse coordinator's staging admit path; the per-layer method
        remains for parity testing and the per-layer extend hook in the
        quest_only backend.

        Behaviour mirrors the per-layer method:
          * full pages → page_k_min/max (via page_k_bounds backing tensor)
          * partial last page → running_k_min/max
          * running_token_count + running_page_idx scalars updated once

        Args:
          req_pool_idx: int — request slot to write into.
          k_data_ptrs: ``[num_total_layers]`` uint64 — pointer to each
            layer's K buffer (from ``MHATokenToKVPool.k_data_ptrs``).
          prefill_indices: ``[prefill_len]`` int — token positions in the
            K buffer for this request, in order.
        """
        if not _USE_FUSED_PREFILL_BOUNDS:
            raise RuntimeError(
                "update_prefill_representations_fused called but "
                "QUEST_DISABLE_FUSED_PREFILL_BOUNDS=1 is set"
            )
        if prefill_indices.numel() == 0:
            return

        quest_prefill_bounds(
            k_data_ptrs=k_data_ptrs,
            prefill_indices=prefill_indices,
            page_k_bounds=self.page_k_bounds,
            running_k_min=self.running_k_min,
            running_k_max=self.running_k_max,
            layer_offset_start=self.start_layer,
            req_pool_idx=req_pool_idx,
            page_size=self.page_size,
        )

        prefill_len = int(prefill_indices.shape[0])
        num_full_pages = prefill_len // self.page_size
        partial_count = prefill_len - num_full_pages * self.page_size

        self.running_token_count[req_pool_idx] = partial_count
        self.running_page_idx[req_pool_idx] = num_full_pages

    # ----------------------------------------------------------- decode update

    def update_decode_representations(
        self,
        layer_id: int,
        req_indices: torch.Tensor,
        k_buffer: torch.Tensor,
        device_locs: torch.Tensor,
    ) -> None:
        """Accumulate the just-decoded token's K into running min/max.

        Args:
          layer_id: model layer (absolute).
          req_indices: ``[num_reqs]`` int64 request slots.
          k_buffer: ``[pool_size, kv_heads, head_dim]`` the layer's K buffer.
          device_locs: ``[num_reqs]`` int physical addresses of each
            request's just-decoded token in ``k_buffer``.

        Counter advancement (``running_token_count`` / ``running_page_idx``)
        is performed exactly once per decode step by
        :meth:`maybe_finalize_decode_representations`, called by the coordinator after
        all layers' :meth:`update_decode_representations` have run.
        """
        layer_offset = layer_id - self.start_layer
        new_k = k_buffer[device_locs].to(torch.bfloat16)

        cur_min = self.running_k_min[layer_offset, req_indices]
        cur_max = self.running_k_max[layer_offset, req_indices]
        self.running_k_min[layer_offset, req_indices] = torch.minimum(cur_min, new_k)
        self.running_k_max[layer_offset, req_indices] = torch.maximum(cur_max, new_k)

    def update_decode_representations_fused(
        self,
        k_data_ptrs: torch.Tensor,
        req_indices: torch.Tensor,
        device_locs: torch.Tensor,
    ) -> None:
        """All-layer fused variant of :meth:`update_decode_representations`.

        Replaces the num_layers iterations of the per-layer Python op chain
        (gather + minimum/maximum + slice-store) with ONE Triton kernel that
        processes every Quest layer in one launch.

        Args:
          k_data_ptrs: ``[num_total_layers]`` uint64 — pointer to each
            layer's K buffer (from ``MHATokenToKVPool.k_data_ptrs``).
          req_indices: ``[num_active_reqs]`` int64.
          device_locs: ``[num_active_reqs]`` int — physical address of each
            req's just-decoded token in the K buffer.
        """
        if not _USE_FUSED_DECODE_BOUNDS:
            raise RuntimeError(
                "update_decode_representations_fused called but "
                "QUEST_DISABLE_FUSED_DECODE_BOUNDS=1 is set"
            )
        if req_indices.numel() == 0:
            return
        quest_decode_bounds(
            k_data_ptrs=k_data_ptrs,
            device_locs=device_locs,
            req_indices=req_indices,
            running_k_min=self.running_k_min,
            running_k_max=self.running_k_max,
            layer_offset_start=self.start_layer,
        )

    def maybe_finalize_decode_representations(self, req_indices: torch.Tensor) -> None:
        """Advance counters; finalize any page whose count just hit page_size.

        Must be called once per decode step, AFTER
        :meth:`update_decode_representations` has run for every layer in
        ``[start_layer, end_layer)``.

        For each request whose ``running_token_count`` was ``page_size - 1``
        (i.e., this step's update completed a page):
          * copy ``running_k_min/max`` into ``page_k_min/max[*, req, page_idx]``
            for every layer (this OVERWRITES the sentinel — that's how the
            page becomes "valid" for retrieve_topk's score formula)
          * reset running buffers to (+inf, -inf) so the next page's first
            token initialises correctly
          * advance ``running_page_idx[req]`` by 1

        For all requests in ``req_indices``, ``running_token_count`` is
        incremented (mod ``page_size``).
        """
        counts = self.running_token_count[req_indices]
        will_complete = counts == (self.page_size - 1)

        # Run unconditionally on the device — when no req is completing,
        # ``completing_reqs`` is empty and the advanced-indexing assignments
        # below are no-ops.  This avoids a host-device sync from
        # ``will_complete.any().item()`` on the critical decode-step path
        # (the sync was the dominant per-step cost in cuda-graph replay).
        completing_reqs = req_indices[will_complete]
        page_indices = self.running_page_idx[completing_reqs].to(torch.int64)
        self.page_k_min[:, completing_reqs, page_indices] = (
            self.running_k_min[:, completing_reqs]
        )
        self.page_k_max[:, completing_reqs, page_indices] = (
            self.running_k_max[:, completing_reqs]
        )
        self.running_k_min[:, completing_reqs] = _BF16_POS_INF
        self.running_k_max[:, completing_reqs] = _BF16_NEG_INF
        self.running_page_idx[completing_reqs] = (
            self.running_page_idx[completing_reqs] + 1
        )

        # Increment counters for ALL reqs, wrap to 0 on completion.
        self.running_token_count[req_indices] = (counts + 1) % self.page_size

    # ------------------------------------------------------------- invalidate

    # ----------------------------------------------------------- step prep

    def prepare_step(self, seq_lens: torch.Tensor) -> None:
        """Compute step-level state once and write to pre-allocated buffers.

        Backend must call this in ``init_forward_metadata`` (and the cuda-graph
        capture/replay hooks) — once per step, BEFORE any layer's
        ``retrieve_topk``.  ``retrieve_topk`` reads from the buffers' stable
        addresses, avoiding ~6 redundant ops per layer (×48 layers in the
        bench config = ~14 ms saved).
        """
        bs = seq_lens.shape[0]
        if bs > self._step_last_valid_buf.shape[0]:
            raise RuntimeError(
                f"prepare_step bs={bs} > max_reqs={self._step_last_valid_buf.shape[0]}"
            )
        seq_lens_i64 = seq_lens.to(torch.int64)
        last_valid = (seq_lens_i64 - 1).clamp(min=0).unsqueeze(1)  # [bs, 1]
        self._step_last_valid_buf[:bs].copy_(last_valid, non_blocking=True)

        recent_start = (seq_lens_i64 - self.page_size).clamp(min=0).unsqueeze(1)
        recent_positions = torch.minimum(
            recent_start + self._recent_offsets_const, last_valid,
        )  # [bs, page_size]
        self._step_recent_positions_buf[:bs].copy_(
            recent_positions.to(torch.int32), non_blocking=True,
        )

        short_layout = torch.minimum(
            self._all_positions_const, last_valid,
        ).to(torch.int32)  # [bs, top_k]
        self._step_short_layout_buf[:bs].copy_(short_layout, non_blocking=True)

        is_short = (seq_lens_i64 <= self.top_k).unsqueeze(1)
        self._step_is_short_buf[:bs].copy_(is_short, non_blocking=True)

        actual_lens = torch.minimum(
            seq_lens_i64, torch.full_like(seq_lens_i64, self.top_k),
        ).to(torch.int32)
        self._step_actual_lens_buf[:bs].copy_(actual_lens, non_blocking=True)

    @property
    def page_valid(self) -> torch.Tensor:
        """Backwards-compat shim: derive validity from sentinel.

        A page is "valid" iff its k_max (any element) is strictly greater
        than the bf16 -inf sentinel.  Same shape as the legacy page_valid
        tensor: ``[num_layers, max_reqs, max_pages_per_req]``, dtype bool.

        This is for tests that introspect Quest state.  The retrieve_topk
        hot path does NOT call this — it reads page_k_bounds directly and
        relies on the sentinel score (-inf) to filter invalid pages.
        """
        # ``page_k_max`` is a view of page_k_bounds[..., 1].
        # Reduce over (kv_heads, head_dim) — a page is valid iff at least
        # one element exceeds the sentinel; we use a strict > to be safe
        # against bf16 underflow on real bounds.
        return (self.page_k_max > _BF16_NEG_INF).any(dim=(-1, -2))

    def invalidate_request(self, req_pool_idx: int) -> None:
        """Reset all per-request state when a request is freed.

        Required because page bounds storage is shared across requests via
        the (req_pool_idx, page_in_req) addressing scheme; without this,
        a new request landing in the same slot would read stale bounds
        and topk would pick stale pages.
        """
        # Reset to sentinel: min=+inf, max=-inf → score = -inf for the slot.
        self.page_k_min[:, req_pool_idx, :].fill_(_BF16_POS_INF)
        self.page_k_max[:, req_pool_idx, :].fill_(_BF16_NEG_INF)
        self.running_k_min[:, req_pool_idx] = _BF16_POS_INF
        self.running_k_max[:, req_pool_idx] = _BF16_NEG_INF
        self.running_token_count[req_pool_idx] = 0
        self.running_page_idx[req_pool_idx] = 0

    # ------------------------------------------------------------- retrieval

    def retrieve_topk(
        self,
        queries: torch.Tensor,
        layer_id: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return top-k token positions for sparse attention.

        Args:
          queries: ``[bs, q_heads, head_dim]`` (or ``[bs, hidden]`` flattened).
          layer_id: absolute layer id.
          req_pool_indices: ``[bs]`` int64.
          seq_lens: ``[bs]`` int32 or int64.  Each entry is the current
            sequence length for that request.

        Returns:
          ``(token_positions, actual_lens)``:
            * ``token_positions``: ``[bs, top_k]`` int32, only the first
              ``actual_lens[i]`` entries per row are meaningful (the
              backend must pack via ``actual_lens`` to avoid duplicate
              over-weighting in softmax).
            * ``actual_lens``: ``[bs]`` int32, ``min(seq_len_i, top_k)``.

        Layout per request (long sequences, ``seq_len >= top_k``):
            slots ``[0, top_k - page_size)``: top-``(top_k_pages - 1)``
              best-scoring full pages, expanded to token positions.
            slots ``[top_k - page_size, top_k)``: most-recent
              ``page_size`` token positions ("recent window"), which
              always covers the partial last page plus possibly a tail
              of the most-recent full page. This guarantees the model
              can attend to its just-generated context.

        Layout per request (short sequences, ``seq_len < top_k``):
            slots ``[0, seq_len)``: positions ``0, 1, ..., seq_len - 1``
              (i.e. dense). Slots beyond ``seq_len`` are unused (caller
              must respect ``actual_lens``).

        Why the recent-window: Quest's bounding-box scoring only ranks
        fully-completed pages (per-page bounds aren't valid until the
        page fills up). The partial last page — holding the most recent
        ~page_size tokens, which transformer decode heavily relies on —
        would otherwise never be selected, costing the model its just-
        generated context for the entire fill-up cycle.
        """
        layer_offset = layer_id - self.start_layer
        bs = queries.shape[0]

        # Read step-level cached state (set by prepare_step before this step).
        recent_positions = self._step_recent_positions_buf[:bs]  # [bs, page_size] int32
        short_layout = self._step_short_layout_buf[:bs]        # [bs, top_k] int32
        is_short = self._step_is_short_buf[:bs]                # [bs, 1] bool
        actual_lens = self._step_actual_lens_buf[:bs]          # [bs] int32

        # Align query shape to KV heads.
        if queries.dim() == 2:
            hidden = queries.shape[1]
            if hidden % self.head_dim != 0:
                raise ValueError(
                    f"Query hidden {hidden} not divisible by head_dim {self.head_dim}"
                )
            q_heads = hidden // self.head_dim
            q = queries.view(bs, q_heads, self.head_dim)
        elif queries.dim() == 3:
            q = queries
        else:
            raise ValueError(f"Unsupported query shape {queries.shape}")

        if q.shape[1] != self.kv_heads:
            if q.shape[1] % self.kv_heads != 0:
                raise ValueError(
                    f"q_heads {q.shape[1]} not divisible by kv_heads {self.kv_heads}"
                )
            group = q.shape[1] // self.kv_heads
            q = q.view(bs, self.kv_heads, group, self.head_dim).mean(dim=2)

        # Quest criticality: per-page upper bound on Σ q·k under the
        # bounding box [k_min, k_max].  Invalid pages → -inf (sentinel).
        if _USE_TRITON_SCORE:
            # Fused Triton kernel: gather + where + multiply + sum in ONE
            # kernel launch.  Replaces ~7 PyTorch ops, saving cuda graph
            # replay overhead.  Q must be float32 + contiguous.
            q_f32 = q.contiguous().float()
            scores = quest_score(
                q_f32, self.page_k_bounds, req_pool_indices, layer_offset,
            )  # [bs, max_pages_per_req] float32
        else:
            # Reference PyTorch path (for debug + correctness comparison).
            k_bounds = self.page_k_bounds[layer_offset, req_pool_indices]
            q_unsq = q.unsqueeze(1)
            k_chosen = torch.where(
                (q_unsq >= 0).unsqueeze(-1),
                k_bounds[..., 1:2],
                k_bounds[..., 0:1],
            ).squeeze(-1)
            scores = (q_unsq.float() * k_chosen.float()).sum(dim=(2, 3))

        # Long-sequence layout:
        #   first (top_k - page_size) slots: top (top_k_pages - 1) page expansions
        #   last page_size slots: recent window (cached in prepare_step)
        select_pages = self.top_k_pages - 1  # may be 0 if top_k == page_size
        if select_pages > 0:
            # Defensive: mask scores for pages beyond the request's actual page
            # count to -inf so topk cannot pick stale state from a previous
            # occupant of the (req_pool_idx, page_in_req) slot.  page_k_bounds
            # is shared across requests via slot reuse, and invalidate_request
            # is the only mechanism keeping it clean — if any free path drops
            # the invalidation (or bf16 sentinel rounds badly under the score
            # formula), we'd otherwise silently pick OOB pages and read
            # garbage K/V.  Cost: ~3 ops/layer; arange constant is cached.
            num_pages_per_req = (seq_lens.to(torch.int64) // self.page_size).unsqueeze(1)
            page_idx = self._page_idx_const[: scores.shape[1]].unsqueeze(0)
            scores = scores.masked_fill(page_idx >= num_pages_per_req, float("-inf"))
            topk_pages = torch.topk(scores, k=select_pages, dim=1).indices  # [bs, select_pages] int64
            # Expand pages to token positions.  No need to clamp — for
            # long-seq (where this layout is used), num_full_pages ≥
            # top_k_pages so topk picks valid in-range pages.
            select_positions = (
                topk_pages.unsqueeze(2) * self.page_size + self._page_offsets_const
            ).reshape(bs, select_pages * self.page_size)
        else:
            select_positions = torch.empty(
                (bs, 0), dtype=torch.int64, device=self.device
            )

        long_layout = torch.cat([select_positions, recent_positions], dim=1)  # [bs, top_k]

        token_positions = torch.where(is_short, short_layout, long_layout)

        return token_positions.to(torch.int32), actual_lens
