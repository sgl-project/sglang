"""Page-aware image-bucket constraints for incremental inference.

Training-compatible page IDs persist per physical KV slot across prefill and
decode. Routing selects image buckets from the query page and falls back to the
full domain on no match.
"""

from __future__ import annotations

from typing import Optional

import msgspec
import torch

from sglang.srt.runtime_context import get_buffer

_INT16_MAX = 32767

# Multimodal placeholders are rewritten above this value; text markers are not.
_MM_PAD_SHIFT_VALUE = 1_000_000


def image_token_mask(ids: torch.Tensor, image_token_id: int) -> torch.Tensor:
    """``True`` where a token is an image placeholder (sentinel or literal id)."""
    return (ids >= _MM_PAD_SHIFT_VALUE) | (ids == image_token_id)


def _stable(name: str, t: torch.Tensor) -> torch.Tensor:
    """Copy ``t`` into a fixed-address buffer and return the buffer."""
    key = f"uocr_page_state:{name}:{tuple(t.shape)}:{t.dtype}:{t.device}"
    buf = get_buffer(key, lambda: torch.empty_like(t))
    buf.copy_(t)
    return buf


class BatchLayout(msgspec.Struct):
    """Per-step, per-request geometry shared by the indexer / main metadata."""

    num_tokens: int
    num_reqs: int
    # [num_reqs + 1] int64: offset of each request's tokens in the flat array.
    q_start: torch.Tensor
    # [num_reqs] int64: absolute KV position of each request's first new token.
    start_pos: torch.Tensor
    # [num_reqs, max_blocks] int64/int32: logical -> physical block ids.
    block_table: torch.Tensor
    # [num_tokens] int64: request row of every flat token.
    req_of_token: torch.Tensor
    # Prompt and generated tokens use different page-numbering domains.
    num_decode_tokens: int = 0
    # [num_reqs] int64: absolute position of each request's first generated
    # token. An extend can carry generated tokens too (a retracted request
    # re-prefills prompt + output); None means an extend is all prompt.
    gen_start: Optional[torch.Tensor] = None


def generated_mask(
    layout: BatchLayout, tok_idx: torch.Tensor, req: torch.Tensor
) -> torch.Tensor:
    """[num_tokens] bool: which tokens of this step are generated (not prompt)."""
    is_gen = tok_idx < int(layout.num_decode_tokens)
    if layout.gen_start is None:
        return is_gen
    abs_pos = layout.start_pos[req] + (tok_idx - layout.q_start[req])
    return is_gen | (abs_pos >= layout.gen_start[req])


def _num_tokens_tail(num_tokens: int, device: torch.device) -> torch.Tensor:
    """Build the CUDA-resident ``q_start`` sentinel without a host copy."""
    return torch.full((1,), num_tokens, device=device, dtype=torch.int64)


def _req_of_token(q_start: torch.Tensor, num_tokens: int) -> torch.Tensor:
    """Token -> request row, shape-static (no data-dependent output shape)."""
    tok = torch.arange(num_tokens, device=q_start.device, dtype=torch.int64)
    return torch.searchsorted(q_start[1:].contiguous(), tok, right=True).clamp_(
        max=q_start.shape[0] - 2
    )


def build_layout_from_indexer(index_md) -> BatchLayout | None:
    """Layout from index KV-cache group metadata (``decode`` / ``prefill``)."""
    num_tokens = int(index_md.num_actual_tokens)
    num_reqs = int(index_md.seq_lens.shape[0])
    if num_tokens <= 0 or num_reqs <= 0:
        return None
    nd = int(index_md.num_decodes)
    device = index_md.seq_lens.device
    starts: list[torch.Tensor] = []
    start_pos: list[torch.Tensor] = []
    tables: list[torch.Tensor] = []
    if nd > 0:
        d = index_md.decode
        if d is None:
            return None
        dql = int(d.decode_query_len)
        starts.append(torch.arange(nd, device=device, dtype=torch.int64) * dql)
        start_pos.append(d.seq_lens.to(torch.int64) - dql)
        tables.append(d.block_table)
    if int(index_md.num_prefills) > 0:
        p = index_md.prefill
        if p is None:
            return None
        starts.append(
            p.cu_seqlens_q[:-1].to(torch.int64) + int(index_md.num_decode_tokens)
        )
        start_pos.append(p.context_lens.to(torch.int64))
        tables.append(p.block_table)
    if not starts:
        return None
    q_start = torch.cat(starts + [_num_tokens_tail(num_tokens, device)])
    return BatchLayout(
        num_tokens=num_tokens,
        num_reqs=num_reqs,
        q_start=q_start,
        start_pos=torch.cat(start_pos),
        block_table=torch.cat(tables, dim=0).to(torch.int64),
        req_of_token=_req_of_token(q_start, num_tokens),
        num_decode_tokens=int(index_md.num_decode_tokens) if nd > 0 else 0,
    )


def build_layout_from_main(main_md) -> BatchLayout | None:
    """Layout from main KV-cache group metadata (``decode`` / ``prefill``)."""
    num_tokens = int(main_md.num_actual_tokens)
    num_reqs = int(main_md.seq_lens.shape[0])
    if num_tokens <= 0 or num_reqs <= 0:
        return None
    nd = int(main_md.num_decodes)
    device = main_md.seq_lens.device
    starts: list[torch.Tensor] = []
    start_pos: list[torch.Tensor] = []
    tables: list[torch.Tensor] = []
    if nd > 0:
        d = main_md.decode
        if d is None:
            return None
        dql = int(d.decode_query_len)
        starts.append(torch.arange(nd, device=device, dtype=torch.int64) * dql)
        start_pos.append(d.seq_lens.to(torch.int64) - dql)
        tables.append(d.block_table)
    if int(main_md.num_prefills) > 0:
        p = main_md.prefill
        if p is None:
            return None
        starts.append(
            p.cu_seqlens_q[:-1].to(torch.int64) + int(main_md.num_decode_tokens)
        )
        start_pos.append(p.context_lens.to(torch.int64))
        tables.append(p.block_table)
    if not starts:
        return None
    q_start = torch.cat(starts + [_num_tokens_tail(num_tokens, device)])
    return BatchLayout(
        num_tokens=num_tokens,
        num_reqs=num_reqs,
        q_start=q_start,
        start_pos=torch.cat(start_pos),
        block_table=torch.cat(tables, dim=0).to(torch.int64),
        req_of_token=_req_of_token(q_start, num_tokens),
        num_decode_tokens=int(main_md.num_decode_tokens) if nd > 0 else 0,
    )


class PageIsolationState:
    """Persistent per-slot page bookkeeping + per-query block allow masks."""

    def __init__(self, sparse_cfg: dict, block_size: int) -> None:
        self.block_size = int(block_size)
        self.image_token_id = int(sparse_cfg.get("image_token_id", 128815))
        pt = sparse_cfg.get("page_token_id", None)
        self.page_token_id = int(pt) if pt is not None else None
        self.per_page_image_tokens = max(
            1, int(sparse_cfg.get("per_page_image_tokens", 273))
        )
        # Persistent per-physical-slot state (allocated on first step).
        self._slot_img_page: torch.Tensor | None = None
        self._slot_marker_cum: torch.Tensor | None = None
        self._slot_img_run: torch.Tensor | None = None
        self._slot_prompt_pages: torch.Tensor | None = None
        self._num_slots = 0
        self._last_img_run_max: torch.Tensor | None = None

    def _ensure(self, num_slots: int, device: torch.device) -> None:
        if self._slot_img_page is not None and self._num_slots >= num_slots:
            return
        self._slot_img_page = torch.full(
            (num_slots,), -1, dtype=torch.int32, device=device
        )
        self._slot_marker_cum = torch.zeros(num_slots, dtype=torch.int32, device=device)
        self._slot_img_run = torch.zeros(num_slots, dtype=torch.int32, device=device)
        self._slot_prompt_pages = torch.zeros(
            num_slots, dtype=torch.int32, device=device
        )
        self._num_slots = num_slots

    def _carry(self, layout: BatchLayout) -> tuple[torch.Tensor, ...]:
        """Per-request carry-in read off the slot of the previous token."""
        prev_pos = layout.start_pos - 1
        valid = prev_pos >= 0
        p = prev_pos.clamp(min=0)
        bs = self.block_size
        blk = (p // bs).clamp(max=layout.block_table.shape[1] - 1)
        phys = layout.block_table.gather(1, blk[:, None]).squeeze(1)
        slot = (phys * bs + (p % bs)).clamp(min=0, max=self._num_slots - 1)
        slot = torch.where(valid, slot, torch.zeros_like(slot))
        zero = torch.zeros_like(slot, dtype=torch.int32)
        return (
            torch.where(valid, self._slot_marker_cum[slot], zero),
            torch.where(valid, self._slot_img_run[slot], zero),
            torch.where(valid, self._slot_prompt_pages[slot], zero),
        )

    @torch.no_grad()
    def step(
        self,
        input_ids: torch.Tensor,
        layout: BatchLayout,
        slot_mapping: torch.Tensor,
        num_slots: int,
    ) -> torch.Tensor:
        """Update the per-slot state from this step's tokens.

        Returns ``q_page[num_tokens]`` int32: the page id of every query token
        of this step (-1 == page unknown -> query stays unconstrained).
        """
        self._ensure(num_slots, input_ids.device)
        T = layout.num_tokens
        ids = input_ids[:T]
        req = layout.req_of_token
        is_img = image_token_mask(ids, self.image_token_id)
        if self.page_token_id is None:
            is_mark = torch.zeros_like(is_img)
        else:
            is_mark = ids == self.page_token_id

        prev_m, prev_i, prev_pp = self._carry(layout)

        # torch.cumsum promotes integer inputs to int64 unless dtype is pinned;
        # keep everything int32 so the per-slot scatter dtypes match.
        starts = layout.q_start[:-1]
        c_m = torch.cumsum(is_mark, dim=0, dtype=torch.int32)
        c_i = torch.cumsum(is_img, dim=0, dtype=torch.int32)
        zero32 = torch.zeros((), dtype=torch.int32, device=ids.device)
        prev_idx = (starts - 1).clamp(min=0)
        base_m = torch.where(starts > 0, c_m[prev_idx], zero32)
        base_i = torch.where(starts > 0, c_i[prev_idx], zero32)
        marker_cum = c_m - base_m[req] + prev_m[req]
        img_run = c_i - base_i[req] + prev_i[req]

        # Image page: marker mode when a <PAGE> marker precedes this image token
        # (training's per-row marker_mode), else fixed per_page bucketing.
        img_page = torch.where(
            marker_cum > 0,
            marker_cum - 1,
            (img_run - 1) // self.per_page_image_tokens,
        )
        img_page = torch.where(is_img, img_page, torch.full_like(img_page, -1))

        # prompt_pages = marker count as of the last image token seen so far
        # (segment cummax with per-request carry-in; `big` keeps segments apart).
        big = 1 << 24
        off = req * big
        x = (
            torch.where(is_img, marker_cum, torch.zeros_like(marker_cum)).to(
                torch.int64
            )
            + off
        )
        pp = (torch.cummax(x, dim=0).values - off).to(torch.int32)
        pp = torch.maximum(pp, prev_pp[req])

        # Prompt pages use raw marker numbers; generated pages restart at zero.
        tok_idx = torch.arange(T, device=ids.device, dtype=torch.int64)
        is_decode_tok = generated_mask(layout, tok_idx, req)
        neg1 = torch.full_like(marker_cum, -1)
        raw_page = marker_cum - 1
        prefill_page = torch.where(marker_cum > pp, raw_page, neg1)
        resp_page = marker_cum - 1 - pp
        resp_page = torch.where(resp_page >= 0, resp_page, neg1)
        text_page = torch.where(is_decode_tok, resp_page, prefill_page)
        q_page = torch.where(is_img, img_page, text_page)

        # Used only to diagnose an incompatible ``per_page_image_tokens`` value.
        self._last_img_run_max = img_run.max() if T > 0 else None

        sl = slot_mapping[:T].to(torch.int64).clamp(min=0, max=num_slots - 1)
        self._slot_img_page[sl] = img_page
        self._slot_marker_cum[sl] = marker_cum
        self._slot_img_run[sl] = img_run
        self._slot_prompt_pages[sl] = pp

        # Reused partial blocks retain stale image pages outside the live tail.
        bs_i = self.block_size
        q_len_r = layout.q_start[1:] - layout.q_start[:-1]
        end_pos = layout.start_pos + q_len_r  # [R] seq end
        last_blk = ((end_pos - 1).clamp(min=0) // bs_i).clamp(
            min=0, max=layout.block_table.shape[1] - 1
        )
        phys = layout.block_table.gather(1, last_blk[:, None]).squeeze(1)
        off_b = torch.arange(bs_i, device=ids.device, dtype=torch.int64)
        rem = end_pos % bs_i
        stale = (off_b[None, :] >= rem[:, None]) & (rem > 0)[:, None]
        tail = (phys[:, None] * bs_i + off_b[None, :]).clamp(min=0, max=num_slots - 1)
        # Clear stale slots with shape-static gather/scatter operations so this
        # path remains safe during CUDA graph capture.
        flat = tail.reshape(-1)
        st = stale.reshape(-1)
        cur_p = self._slot_img_page.gather(0, flat)
        self._slot_img_page.scatter_(
            0, flat, torch.where(st, torch.full_like(cur_p, -1), cur_p)
        )
        return _stable("q_page", q_page)

    def block_page_bounds(self, num_blocks: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Per physical block: (min, max) image page id, -1 == no image token.

        Pages are contiguous runs far longer than one block
        (``per_page_image_tokens`` >> ``sparse_block_size``), so a block holds at
        most two adjacent pages and ``min <= p <= max`` is an exact test.
        """
        bs = self.block_size
        pages = self._slot_img_page[: num_blocks * bs].view(num_blocks, bs)
        blk_max = pages.amax(dim=1)
        big = torch.full_like(pages, _INT16_MAX)
        blk_min = torch.where(pages >= 0, pages, big).amin(dim=1)
        blk_min = torch.where(blk_max >= 0, blk_min, torch.full_like(blk_min, -1))
        return blk_min.to(torch.int16), blk_max.to(torch.int16)

    def _block_page_rows(
        self,
        layout: BatchLayout,
        q_page: torch.Tensor,
        max_block: int,
        num_blocks: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-query ``(is_image_block, same_page, open_row)`` block tables."""
        blk_min, blk_max = self.block_page_bounds(num_blocks)
        bt = layout.block_table
        w = min(max_block, int(bt.shape[1]))
        R = layout.num_reqs
        device = q_page.device
        bmin_r = torch.full((R, max_block), -1, dtype=torch.int16, device=device)
        bmax_r = torch.full((R, max_block), -1, dtype=torch.int16, device=device)
        idx = bt[:, :w].clamp(min=0, max=num_blocks - 1)
        bmin_r[:, :w] = blk_min[idx]
        bmax_r[:, :w] = blk_max[idx]

        req = layout.req_of_token
        bmin_q = bmin_r[req]
        bmax_q = bmax_r[req]
        qp = q_page.to(torch.int16)[:, None]
        img_blk = bmin_q >= 0
        same = img_blk & (bmin_q <= qp) & (qp <= bmax_q) & (qp >= 0)
        open_row = (qp < 0) | ~same.any(dim=-1, keepdim=True)
        return img_blk, same, open_row

    @torch.no_grad()
    def build_query_allow(
        self,
        layout: BatchLayout,
        q_page: torch.Tensor,
        max_block: int,
        num_blocks: int,
    ) -> torch.Tensor:
        """Build the query-page candidate mask, with an open-row fallback."""
        img_blk, same, open_row = self._block_page_rows(
            layout, q_page, max_block, num_blocks
        )
        del img_blk
        return _stable("query_allow", same | open_row)
