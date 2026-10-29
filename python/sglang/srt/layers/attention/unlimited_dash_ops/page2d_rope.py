"""Two-axis ``(tok_pos, page_pos)`` RoPE with streaming coordinates.

Page positions align prompt images with response text; page zero holds non-page
tokens. Token positions count within each page. ``PageCoordState`` reproduces
batched training coordinates in incremental flat inference steps.
"""

from __future__ import annotations

import math

import torch

from sglang.srt.layers.attention.unlimited_dash_ops.page_state import (
    generated_mask,
    image_token_mask,
)

# Prompt tokens of the cheapest page: a marker plus the "tiny" preset's
# (8 + 1) * 8 + 1 image tokens (multimodal/processors/unlimited_ocr.py).
MIN_PAGE_TOKENS = 74


def page_capacity(context_len: int) -> int:
    """Page buckets enough for any prompt that fits ``context_len`` tokens.

    Page 0 holds the prefix and instruction; generated pages reuse prompt
    buckets, so no admissible request reaches the clamp to the last bucket.
    """
    return max(1, int(context_len)) // MIN_PAGE_TOKENS + 2


class PageRope2D:
    """Frequency tables matching the training-time 2D indexer rotary.

    ``inv_tok`` pairs precede ``inv_page``, preserving ``rotate_half`` pairing;
    ``page_mod`` folds only the RoPE page axis, coordinate state keeps raw pages.
    """

    def __init__(
        self,
        dim: int,
        base_tok: float,
        base_page: float,
        page_pairs: int,
        tok_freq_mode: str = "resample",
        page_mod: int = 0,
    ) -> None:
        half = dim // 2
        page_pairs = int(page_pairs)
        if not (0 < page_pairs < half):
            raise ValueError(
                f"[Unlimited-OCR v2] page2d RoPE page_pairs={page_pairs} must be in (0, {half}); dim={dim}"
            )
        n_tok = half - page_pairs
        ar = torch.arange(0, n_tok, dtype=torch.int64).float()
        if str(tok_freq_mode).lower() == "truncate":
            inv_tok = 1.0 / (base_tok ** (ar * 2 / dim))
        else:
            inv_tok = 1.0 / (base_tok ** (ar / n_tok))
        inv_page = 1.0 / (
            base_page
            ** (torch.arange(0, page_pairs, dtype=torch.int64).float() / page_pairs)
        )
        self.dim = dim
        self.n_tok = n_tok
        self.page_pairs = page_pairs
        self.base_tok = float(base_tok)
        self.base_page = float(base_page)
        self.tok_freq_mode = str(tok_freq_mode).lower()
        self.page_mod = int(page_mod or 0)
        self.inv_tok = inv_tok
        self.inv_page = inv_page
        self._dev_cache: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}

    def _freqs(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        got = self._dev_cache.get(device)
        if got is None:
            got = (self.inv_tok.to(device), self.inv_page.to(device))
            self._dev_cache[device] = got
        return got

    @property
    def signature(self) -> tuple:
        """State affecting ``cos_sin`` besides coordinates."""
        return (
            self.dim,
            self.n_tok,
            self.page_pairs,
            self.base_tok,
            self.base_page,
            self.tok_freq_mode,
            self.page_mod,
        )

    @property
    def page_mono_range(self) -> float:
        """Monotone (unambiguous) page distance of the page axis."""
        if self.page_mod > 0:
            # Folded axis: report the fold period; "monotone range" is meaningless.
            return float(self.page_mod)
        return self.base_page ** ((self.page_pairs - 1) / self.page_pairs) * math.pi

    @property
    def tok_mono_range(self) -> float:
        """Monotone (unambiguous) token distance of the token axis."""
        return math.pi / float(self.inv_tok[-1])

    @torch.no_grad()
    def cos_sin(
        self,
        tok_pos: torch.Tensor,
        page_pos: torch.Tensor,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``tok_pos`` / ``page_pos`` [T] int -> cos/sin [T, dim] (fp32 math)."""
        inv_tok, inv_page = self._freqs(tok_pos.device)
        with torch.autocast(device_type=tok_pos.device.type, enabled=False):
            t = tok_pos.float()[:, None]
            if self.page_mod > 0:
                # int64 remainder, like training (float modulo would drift).
                page_pos = torch.remainder(page_pos.to(torch.int64), self.page_mod)
            p = page_pos.float()[:, None]
            ang = torch.cat((t * inv_tok[None, :], p * inv_page[None, :]), dim=-1)
            emb = torch.cat((ang, ang), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype), sin.to(dtype)


class PageCoordState:
    """Stream ``(tok_pos, page_pos)`` for flat decode-first input.

    State tracks marker count, prompt page count, and per-page token offsets
    across chunked prefill and decode. Rows are keyed by first physical KV block
    and reused by LRU; prefix caching is unsupported.
    """

    def __init__(
        self,
        sparse_cfg: dict,
        max_reqs: int,
        max_pages: int,
        max_spec_slots: int = 1,
    ) -> None:
        if max_pages <= 0:
            raise ValueError(f"max_pages must be positive, got {max_pages}")
        self.image_token_id = int(sparse_cfg.get("image_token_id", 128815))
        pt = sparse_cfg.get("page_token_id", None)
        self.page_token_id = int(pt) if pt is not None else None
        self.max_pages = int(max_pages)
        self.max_reqs = int(max_reqs)
        # Snapshots rewind rejected speculative tokens on the next step.
        self.max_slots = max(1, int(max_spec_slots))
        # One scratch row (index == max_reqs) absorbs masked-out writes.
        self._dummy = self.max_reqs
        self.key: torch.Tensor | None = None
        self.stamp: torch.Tensor | None = None
        self.st_marker: torch.Tensor | None = None
        self.st_pp: torch.Tensor | None = None
        self.cnt: torch.Tensor | None = None
        # Snapshot of the LAST committed step per row, so the next step can undo
        # a rejected tail (see _rewind_rejected).
        self.sp_pos0: torch.Tensor | None = None
        self.sp_n: torch.Tensor | None = None
        self.sp_marker: torch.Tensor | None = None
        self.sp_pp: torch.Tensor | None = None
        self.sp_pl: torch.Tensor | None = None
        self.sp_marker0: torch.Tensor | None = None
        self.sp_pp0: torch.Tensor | None = None
        self.sp_cnt0: torch.Tensor | None = None
        self._step = 0
        # Device-side step counter + cached scratch-row index (see _rows).
        self._step_t: torch.Tensor | None = None
        self._dummy_t: torch.Tensor | None = None

    def _step_counter(self) -> torch.Tensor:
        """0-dim int64 CUDA counter, incremented by a kernel (capture-safe)."""
        if self._step_t is None:
            self._step_t = torch.zeros((), dtype=torch.int64, device=self.key.device)
        self._step_t += 1
        self._step += 1
        return self._step_t

    def _dummy_idx(self, device: torch.device) -> torch.Tensor:
        if self._dummy_t is None:
            self._dummy_t = torch.full(
                (1,), self._dummy, dtype=torch.int64, device=device
            )
        return self._dummy_t

    def _ensure(self, device: torch.device) -> None:
        if self.key is not None:
            return
        n = self.max_reqs + 1
        self.key = torch.full((n,), -1, dtype=torch.int64, device=device)
        self.stamp = torch.zeros(n, dtype=torch.int64, device=device)
        self.st_marker = torch.zeros(n, dtype=torch.int32, device=device)
        self.st_pp = torch.zeros(n, dtype=torch.int32, device=device)
        self.cnt = torch.zeros((n, self.max_pages), dtype=torch.int32, device=device)
        w = self.max_slots
        self.sp_pos0 = torch.full((n,), -1, dtype=torch.int64, device=device)
        self.sp_n = torch.zeros(n, dtype=torch.int32, device=device)
        self.sp_marker = torch.zeros((n, w), dtype=torch.int32, device=device)
        self.sp_pp = torch.zeros((n, w), dtype=torch.int32, device=device)
        self.sp_pl = torch.full((n, w), -1, dtype=torch.int32, device=device)
        self.sp_marker0 = torch.zeros(n, dtype=torch.int32, device=device)
        self.sp_pp0 = torch.zeros(n, dtype=torch.int32, device=device)
        self.sp_cnt0 = torch.zeros(
            (n, self.max_pages), dtype=torch.int32, device=device
        )

    @torch.no_grad()
    def _rows(self, layout) -> torch.Tensor:
        """Match the first physical block to a state row or allocate by LRU.

        Device-side writes and counters keep CUDA graph replay safe.
        """
        step = self._step_counter()
        keys = layout.block_table[:, 0].to(torch.int64)
        fresh = layout.start_pos.to(torch.int64) <= 0
        live = self.key[None, :] >= 0
        eq = (keys[:, None] == self.key[None, :]) & live
        matched = eq.any(dim=1)
        row = eq.to(torch.int32).argmax(dim=1)
        # Reset a live key in place at position zero; allocating a new row could
        # duplicate the key and expose stale coordinate state.
        alloc = ~matched
        reset = fresh & matched
        need = alloc | reset

        # Refresh kept rows' stamps BEFORE picking LRU victims so a row in use
        # this step cannot be evicted; a reset row is in use too, hence ``alloc``.
        dummy = torch.full_like(row, self._dummy)
        keep_row = torch.where(alloc, dummy, row).to(torch.int64)
        self.stamp.index_put_((keep_row,), step.expand(keep_row.shape))

        # Free rows first (stamp -1), then oldest stamp; the scratch row last.
        prio = torch.where(self.key >= 0, self.stamp, torch.full_like(self.stamp, -1))
        prio.index_fill_(0, self._dummy_idx(prio.device), torch.iinfo(torch.int64).max)
        order = torch.argsort(prio)
        slot = (torch.cumsum(alloc.to(torch.int64), dim=0) - 1).clamp(
            min=0, max=self.max_reqs - 1
        )
        victim = order[slot].to(row.dtype)

        target = torch.where(alloc, victim, row)
        # Warmup batches may repeat an all-zero key; fold duplicates onto the
        # first row so the state table remains unique.
        pos = torch.arange(target.shape[0], device=target.device)
        earlier = (keys[:, None] == keys[None, :]) & (pos[None, :] < pos[:, None])
        first = earlier.to(torch.int32).argmax(dim=1).to(torch.int64)
        target = torch.where(earlier.any(dim=1), target.gather(0, first), target)
        w = torch.where(need, target, dummy).to(torch.int64)
        self.key.index_put_((w,), keys)
        self.stamp.index_put_((w,), step.expand(w.shape))
        self.st_marker.index_fill_(0, w, 0)
        self.st_pp.index_fill_(0, w, 0)
        self.cnt.index_fill_(0, w, 0)
        # Masked writes may set the scratch key; invalidate it before matching.
        self.key.index_fill_(0, self._dummy_idx(self.key.device), -1)
        # A reset row has no previous step to rewind to.
        self.sp_pos0.index_fill_(0, w, -1)
        return target.to(torch.int64)

    @torch.no_grad()
    def _rewind_rejected(self, layout, row: torch.Tensor) -> None:
        """Rewind running coordinates to the accepted speculative prefix.

        Snapshots are position-anchored; only verify-width steps are snapshotted.
        """
        pos0 = self.sp_pos0[row]  # [R] int64
        n_prev = self.sp_n[row].to(torch.int64)
        acc = (layout.start_pos.to(torch.int64) - pos0).clamp(min=0)
        hit = (pos0 >= 0) & (acc < n_prev)
        # Rows without a snapshot (pos0 == -1) can produce a huge `acc`; clamp the
        # slot math into range -- their writes are masked out below anyway.
        acc = acc.clamp(max=self.max_slots)
        j = (acc - 1).clamp(min=0)[:, None]
        # carry as of the last ACCEPTED token; acc == 0 -> the pre-step carry.
        gm = self.sp_marker[row].gather(1, j).squeeze(1)
        gp = self.sp_pp[row].gather(1, j).squeeze(1)
        pos_marker = torch.where(acc > 0, gm, self.sp_marker0[row])
        pos_pp = torch.where(acc > 0, gp, self.sp_pp0[row])
        # cnt = pre-step cnt + the page buckets of the accepted tokens only.
        pl_prev = self.sp_pl[row]  # [R, W]
        slots = torch.arange(self.max_slots, device=row.device)
        live = (slots[None, :] < acc[:, None]) & (pl_prev >= 0)
        cnt_new = self.sp_cnt0[row].clone()
        cnt_new.scatter_add_(
            1, pl_prev.clamp(min=0).to(torch.int64), live.to(cnt_new.dtype)
        )
        # Masked-out rows write into the scratch row (index == max_reqs).
        w = torch.where(hit, row, torch.full_like(row, self._dummy))
        self.st_marker.index_put_((w,), pos_marker)
        self.st_pp.index_put_((w,), pos_pp)
        self.cnt.index_put_((w,), cnt_new)

    @torch.no_grad()
    def _snapshot(
        self,
        layout,
        row,
        req,
        starts,
        q_len,
        tok_idx,
        marker_cum,
        pp,
        pl,
        st_marker_in,
        st_pp_in,
        cnt_in,
    ) -> None:
        """Snapshot verify-width rows for a possible next-step rewind.

        Writes go through a scratch slot to avoid data-dependent indexing during
        CUDA Graph capture; wider prompt rows invalidate old snapshots.
        """
        W = self.max_slots
        R = int(layout.num_reqs)
        device = pl.device
        dummy = torch.full_like(row, self._dummy)
        narrow = q_len <= W
        jj = tok_idx - starts[req]
        keep_tok = narrow[req] & (jj < W)
        sink = torch.full_like(jj, R * W)
        dest = torch.where(keep_tok, req * W + jj.clamp(max=W - 1), sink)
        n_buf = R * W + 1
        buf_m = torch.zeros(n_buf, dtype=torch.int32, device=device)
        buf_p = torch.zeros(n_buf, dtype=torch.int32, device=device)
        buf_l = torch.full((n_buf,), -1, dtype=torch.int32, device=device)
        buf_m.index_put_((dest,), marker_cum)
        buf_p.index_put_((dest,), pp)
        buf_l.index_put_((dest,), pl.to(torch.int32))
        wsave = torch.where(narrow, row, dummy)
        self.sp_marker.index_put_((wsave,), buf_m[: R * W].view(R, W))
        self.sp_pp.index_put_((wsave,), buf_p[: R * W].view(R, W))
        self.sp_pl.index_put_((wsave,), buf_l[: R * W].view(R, W))
        self.sp_marker0.index_put_((wsave,), st_marker_in)
        self.sp_pp0.index_put_((wsave,), st_pp_in)
        self.sp_cnt0.index_put_((wsave,), cnt_in)
        self.sp_n.index_put_((wsave,), q_len.to(torch.int32))
        self.sp_pos0.index_put_((wsave,), layout.start_pos.to(torch.int64))
        wide = torch.where(narrow, dummy, row)
        self.sp_pos0.index_put_((wide,), torch.full_like(row, -1))

    @torch.no_grad()
    def step(
        self, input_ids: torch.Tensor, layout
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Coordinates of this step's tokens: ``(tok_pos, page_pos)`` [T] int64.

        ``layout`` is ``page_state.BatchLayout`` (duck-typed in the tests).
        """
        t = int(layout.num_tokens)
        if t <= 0:
            return None
        device = input_ids.device
        self._ensure(device)
        ids = input_ids[:t]
        req = layout.req_of_token.to(torch.int64)
        starts = layout.q_start[:-1].to(torch.int64)
        q_len = layout.q_start[1:].to(torch.int64) - starts
        row = self._rows(layout)
        # Undo the tokens the previous step committed but the verifier rejected.
        # Must run before the carries below are read.
        self._rewind_rejected(layout, row)
        st_marker_in = self.st_marker[row]
        st_pp_in = self.st_pp[row]
        cnt_in = self.cnt[row]

        is_img = image_token_mask(ids, self.image_token_id)
        if self.page_token_id is None:
            is_mark = torch.zeros_like(is_img)
        else:
            is_mark = ids == self.page_token_id
        marker_cum, pp = _marker_and_prompt_pages(
            is_img=is_img,
            is_mark=is_mark,
            req=req,
            starts=starts,
            st_marker_in=st_marker_in,
            st_pp_in=st_pp_in,
        )
        tok_idx = torch.arange(t, device=device, dtype=torch.int64)
        pl = _page_of_tokens(
            layout=layout,
            req=req,
            tok_idx=tok_idx,
            is_img=is_img,
            marker_cum=marker_cum,
            pp=pp,
            max_pages=self.max_pages,
        )
        flat = req * self.max_pages + pl
        tok = cnt_in.reshape(-1)[flat] + _rank_in_bucket(flat, tok_idx)

        self._commit_carry(
            layout=layout,
            row=row,
            q_len=q_len,
            tok_idx=tok_idx,
            flat=flat,
            marker_cum=marker_cum,
            pp=pp,
            st_marker_in=st_marker_in,
            st_pp_in=st_pp_in,
            cnt_in=cnt_in,
        )
        self._snapshot(
            layout,
            row,
            req,
            starts,
            q_len,
            tok_idx,
            marker_cum,
            pp,
            pl,
            st_marker_in,
            st_pp_in,
            cnt_in,
        )
        return tok.to(torch.int64), pl

    def _commit_carry(
        self,
        *,
        layout,
        row: torch.Tensor,
        q_len: torch.Tensor,
        tok_idx: torch.Tensor,
        flat: torch.Tensor,
        marker_cum: torch.Tensor,
        pp: torch.Tensor,
        st_marker_in: torch.Tensor,
        st_pp_in: torch.Tensor,
        cnt_in: torch.Tensor,
    ) -> None:
        """Persist each request's marker / prompt-page carry and page counts."""
        pages = self.max_pages
        last = (layout.q_start[1:].to(torch.int64) - 1).clamp(min=0)
        empty = q_len <= 0
        # Padding past the last request maps onto it; it must not count.
        real = (tok_idx < layout.q_start[-1]).to(torch.int32)
        totals = (
            torch.zeros(q_len.shape[0] * pages, dtype=torch.int32, device=flat.device)
            .index_add_(0, flat, real)
            .view(-1, pages)
        )
        keep = empty[:, None]
        self.cnt[row] = torch.where(keep, cnt_in, cnt_in + totals)
        self.st_marker[row] = torch.where(empty, st_marker_in, marker_cum[last])
        self.st_pp[row] = torch.where(empty, st_pp_in, pp[last])


def _marker_and_prompt_pages(
    *,
    is_img: torch.Tensor,
    is_mark: torch.Tensor,
    req: torch.Tensor,
    starts: torch.Tensor,
    st_marker_in: torch.Tensor,
    st_pp_in: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-request segmented marker count and prompt-page count, with carry-in."""
    zero32 = torch.zeros((), dtype=torch.int32, device=is_img.device)
    c_m = torch.cumsum(is_mark, dim=0, dtype=torch.int32)
    prev_idx = (starts - 1).clamp(min=0)
    base_m = torch.where(starts > 0, c_m[prev_idx], zero32)
    marker_cum = c_m - base_m[req] + st_marker_in[req]
    big = 1 << 24
    off = req * big
    x = (
        torch.where(is_img, marker_cum, torch.zeros_like(marker_cum)).to(torch.int64)
        + off
    )
    pp = (torch.cummax(x, dim=0).values - off).to(torch.int32)
    pp = torch.maximum(pp, st_pp_in[req])
    return marker_cum, pp


def _page_of_tokens(
    *,
    layout,
    req: torch.Tensor,
    tok_idx: torch.Tensor,
    is_img: torch.Tensor,
    marker_cum: torch.Tensor,
    pp: torch.Tensor,
    max_pages: int,
) -> torch.Tensor:
    """[T] int64 page bucket of every token, clamped to ``max_pages - 1``."""
    # Prompt pages use marker counts; generated pages restart after prompt pages.
    is_gen = generated_mask(layout, tok_idx, req)
    in_prefill = (~is_gen) & (is_img | (marker_cum > pp))
    resp = (marker_cum - pp).clamp(min=0)
    # Extra generated <PAGE> markers stay on the last real page, as in training.
    resp = torch.minimum(resp, pp)
    page = torch.where(in_prefill, marker_cum, resp).clamp(min=0, max=max_pages - 1)
    return page.to(torch.int64)


def _rank_in_bucket(flat: torch.Tensor, tok_idx: torch.Tensor) -> torch.Tensor:
    """Rank of each token inside its (request, page) bucket, in token order."""
    # A stable sort ranks in O(tokens); a dense [tokens, pages] count table
    # grows with the page capacity and ran out of memory at large ones.
    order = torch.sort(flat, stable=True).indices
    sorted_flat = flat[order]
    head = torch.ones_like(sorted_flat, dtype=torch.bool)
    head[1:] = sorted_flat[1:] != sorted_flat[:-1]
    seg_start = torch.cummax(
        torch.where(head, tok_idx, torch.zeros_like(tok_idx)), dim=0
    ).values
    return torch.empty_like(tok_idx).scatter_(0, order, tok_idx - seg_start)
