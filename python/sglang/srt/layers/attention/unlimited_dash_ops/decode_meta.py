"""Backend adapters feeding DASH decode-metadata kernels from runtime state.

The kernels take tensors only; these adapters own the page / coordinate state
objects and keep the backend's call signatures.
"""

from __future__ import annotations

import torch

from sglang.kernels.ops.attention.unlimited_ocr_dash.decode_meta import (
    dash_decode_allow,
    dash_decode_meta,
    page2d_cos_sin,
)
from sglang.srt.layers.attention.unlimited_dash_ops.page2d_rope import (
    PageCoordState,
    PageRope2D,
)
from sglang.srt.layers.attention.unlimited_dash_ops.page_state import (
    PageIsolationState,
)


class _LayoutShim:
    """The two fields ``PageCoordState._rows`` reads, without a full BatchLayout."""

    __slots__ = ("block_table", "start_pos")

    def __init__(self, block_table, seq_lens, block_size):
        self.block_table = block_table
        # `_rows` keys on "does this request start at position 0" to detect a
        # fresh request; on decode seq_lens is always >= 1 past that.
        self.start_pos = seq_lens.to(torch.int64) - 1


@torch.no_grad()
def decode_meta_fused(
    *,
    input_ids: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    slot_mapping: torch.Tensor,
    page_state: PageIsolationState,
    coord_state: PageCoordState,
    num_slots: int,
    block_size: int,
    out_q_page: torch.Tensor | None = None,
    out_tok: torch.Tensor | None = None,
    out_page: torch.Tensor | None = None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Fused page state and coordinates for one-token decode.

    Caller-provided outputs preserve addresses during CUDA Graph replay.
    """
    device = input_ids.device
    page_state._ensure(num_slots, device)
    coord_state._ensure(device)
    row = coord_state._rows(_LayoutShim(block_table, seq_lens, block_size))
    result = dash_decode_meta(
        input_ids=input_ids,
        block_table=block_table,
        seq_lens=seq_lens,
        slot_mapping=slot_mapping,
        coord_rows=row,
        slot_img_page=page_state._slot_img_page,
        slot_marker_cum=page_state._slot_marker_cum,
        slot_img_run=page_state._slot_img_run,
        slot_prompt_pages=page_state._slot_prompt_pages,
        coord_marker=coord_state.st_marker,
        coord_prompt_pages=coord_state.st_pp,
        coord_page_counts=coord_state.cnt,
        image_token_id=page_state.image_token_id,
        page_token_id=page_state.page_token_id,
        per_page_image_tokens=page_state.per_page_image_tokens,
        num_slots=num_slots,
        max_pages=coord_state.max_pages,
        block_size=block_size,
        out_q_page=out_q_page,
        out_tok=out_tok,
        out_page=out_page,
    )
    # The one-shot diagnostic in the backend reads this; the fused path has no
    # cheap way to produce it, and it only gates a warning.
    page_state._last_img_run_max = None
    return result


@torch.no_grad()
def decode_allow_fused(
    *,
    block_table: torch.Tensor,
    page_state: PageIsolationState,
    q_page: torch.Tensor,
    seq_lens: torch.Tensor,
    max_block: int,
    num_blocks: int,
    block_size: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Page-isolation candidate mask ``[R, max_block]`` int8 (1 = candidate)."""
    return dash_decode_allow(
        block_table=block_table,
        slot_img_page=page_state._slot_img_page,
        q_page=q_page,
        seq_lens=seq_lens,
        max_block=max_block,
        num_blocks=num_blocks,
        block_size=block_size,
        out=out,
    )


@torch.no_grad()
def cos_sin_2d_fused(
    *,
    rope: PageRope2D,
    tok_pos: torch.Tensor,
    page_pos: torch.Tensor,
    dtype: torch.dtype,
    out_cos: torch.Tensor,
    out_sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Write a Page2D rotary's cos/sin table into caller-owned buffers."""
    inv_tok, inv_page = rope._freqs(tok_pos.device)
    return page2d_cos_sin(
        inv_tok=inv_tok,
        inv_page=inv_page,
        tok_pos=tok_pos,
        page_pos=page_pos,
        dim=int(rope.dim),
        page_mod=int(rope.page_mod),
        dtype=dtype,
        out_cos=out_cos,
        out_sin=out_sin,
    )
