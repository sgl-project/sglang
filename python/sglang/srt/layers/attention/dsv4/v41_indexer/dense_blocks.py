"""The candidate scheme carried by dense block ids: a source publishes the ids of
its best blocks, a consumer takes its top-k among them."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, List, Optional, Tuple

import msgspec
import torch

from sglang.kernels.ops.attention.dsv4.candidate_blocks import (
    amax_topk_blocks,
    candidate_block_mask,
    select_candidate_block_ids,
    topk_among_blocks,
)

from .scoring import (
    DeepGEMMPrefillData,
    PagedDecodeScores,
    decode_scores,
    get_deep_gemm_prefill_data,
    prefill_requests,
    score_tiles,
    select_decode,
    write_decode,
    write_prefill,
)
from .types import (
    CandidateMetadata,
    DecodeInputs,
    PrefillInputs,
    get_tail_row_indices,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool


class BlockIds(CandidateMetadata, msgspec.Struct):
    # [rows, n] int32 block ids of each query row, unordered, -1 padded; n is the
    # largest kept count of the batch, at most topk_blocks
    blocks: torch.Tensor
    # prefill: query rows of each request in row order, for the late-layer tail
    rows_per_request: Optional[List[int]] = None
    # Hopper decode materializes this once at the source and reuses it in consumers.
    decode_mask: Optional[torch.Tensor] = None

    def tail(self, rows_per_request: List[int]) -> BlockIds:
        assert self.rows_per_request is not None, "prefill block ids missing"
        rows = get_tail_row_indices(
            full_rows_per_request=self.rows_per_request,
            tail_rows_per_request=rows_per_request,
            device=self.blocks.device,
        )
        return BlockIds(blocks=self.blocks[rows], rows_per_request=rows_per_request)


class DenseBlocksBackend:
    def __init__(
        self,
        *,
        token_to_kv_pool: DeepSeekV4TokenToKVPool,
        req_to_token: torch.Tensor,
        candidate_topk_blocks: int,
        candidate_block_size: int,
        use_deep_gemm_prefill: bool,
    ) -> None:
        self.token_to_kv_pool = token_to_kv_pool
        self.req_to_token = req_to_token
        self.topk_blocks = candidate_topk_blocks
        self.block_size = candidate_block_size
        self.use_deep_gemm_prefill = use_deep_gemm_prefill

    def publish_prefill(self, inputs: PrefillInputs):
        if self.use_deep_gemm_prefill:
            return self._deep_gemm_publish_prefill(inputs)
        return self._torch_publish_prefill(inputs)

    def consume_prefill(
        self,
        inputs: PrefillInputs,
        published: Optional[BlockIds],
    ) -> None:
        if self.use_deep_gemm_prefill:
            self._deep_gemm_consume_prefill(inputs, published)
        else:
            self._torch_consume_prefill(inputs, published)

    def publish_decode(self, inputs: DecodeInputs):
        d = decode_scores(
            inputs=inputs,
            token_to_kv_pool=self.token_to_kv_pool,
            req_to_token=self.req_to_token,
        )
        if d is None:
            return None
        if isinstance(d, PagedDecodeScores):
            assert self.block_size == 8
            blocks = amax_topk_blocks(
                d.scores, d.lens, (d.lens + 7) // 8, self.topk_blocks
            )
            published = BlockIds(
                blocks=blocks,
                decode_mask=candidate_block_mask(blocks, d.lmax, self.block_size),
            )
        else:
            published = BlockIds(
                blocks=select_candidate_block_ids(
                    d.scores,
                    d.lens[:, None],
                    topk_blocks=self.topk_blocks,
                    block_size=self.block_size,
                )
            )
        select_decode(inputs, d, inputs.indexer.index_topk)
        return published

    def consume_decode(
        self,
        inputs: DecodeInputs,
        published: Optional[BlockIds],
    ) -> None:
        d = decode_scores(
            inputs=inputs,
            token_to_kv_pool=self.token_to_kv_pool,
            req_to_token=self.req_to_token,
            candidate_mask=published.decode_mask if published is not None else None,
        )
        if d is None:
            return
        assert published is not None and published.blocks.shape[0] == d.bs
        if isinstance(d, PagedDecodeScores):
            assert published.decode_mask is not None
            select_decode(inputs, d, inputs.indexer.index_topk)
            return
        k = min(inputs.indexer.index_topk, d.lmax)
        idx = topk_among_blocks(
            d.scores, d.lens, published.blocks, k, block_size=self.block_size
        )
        write_decode(inputs, d, idx.masked_fill(idx < 0, d.lmax))

    # ---------- DeepGEMM: flattened-K scores ----------

    def _deep_gemm_publish_prefill(self, inputs: PrefillInputs):
        inputs.reset_outputs()
        data = get_deep_gemm_prefill_data(inputs, self.req_to_token)
        if data is None:
            return None
        selected, blocks = _publish_prefill_blocks(
            data=data,
            kv=self.token_to_kv_pool.get_low_ratio_index_k_fp4(
                inputs.layer_id, data.k_slots
            ),
            topk=inputs.indexer.index_topk,
            topk_blocks=self.topk_blocks,
            block_size=self.block_size,
        )
        data.write_selection(selected, inputs)
        return BlockIds(blocks=blocks, rows_per_request=data.rows_per_request)

    def _deep_gemm_consume_prefill(
        self,
        inputs: PrefillInputs,
        published: Optional[BlockIds],
    ) -> None:
        inputs.reset_outputs()
        data = get_deep_gemm_prefill_data(inputs, self.req_to_token)
        if data is None:
            return
        assert published is not None
        selected = _consume_prefill_blocks(
            data=data,
            kv=self.token_to_kv_pool.get_low_ratio_index_k_fp4(
                inputs.layer_id, data.k_slots
            ),
            topk=inputs.indexer.index_topk,
            blocks=published.blocks,
            block_size=self.block_size,
        )
        data.write_selection(selected, inputs)

    # ---------- torch: per-request scores ----------

    def _torch_publish_prefill(self, inputs: PrefillInputs):
        picked = []  # (query rows, their block ids) per scored chunk
        for request, chunks in prefill_requests(
            inputs=inputs,
            token_to_kv_pool=self.token_to_kv_pool,
            req_to_token=self.req_to_token,
        ):
            for chunk in chunks:
                picked.append(
                    (
                        chunk.tok,
                        select_candidate_block_ids(
                            chunk.scores,
                            chunk.lens[:, None],
                            topk_blocks=self.topk_blocks,
                            block_size=self.block_size,
                        ),
                    )
                )
                idx = chunk.scores.topk(request.k, dim=-1, sorted=False).indices
                write_prefill(inputs, request, chunk, idx)
        width = max((ids.shape[1] for _, ids in picked), default=0)
        blocks = torch.full(
            (inputs.positions.shape[0], width),
            -1,
            dtype=torch.int32,
            device=inputs.positions.device,
        )
        for rows, ids in picked:
            blocks[rows, : ids.shape[1]] = ids
        return BlockIds(blocks=blocks, rows_per_request=inputs.rows_per_request)

    def _torch_consume_prefill(
        self,
        inputs: PrefillInputs,
        published: Optional[BlockIds],
    ) -> None:
        assert published is not None
        for request, chunks in prefill_requests(
            inputs=inputs,
            token_to_kv_pool=self.token_to_kv_pool,
            req_to_token=self.req_to_token,
        ):
            for chunk in chunks:
                idx = topk_among_blocks(
                    chunk.scores,
                    chunk.lens,
                    published.blocks[chunk.tok],
                    request.k,
                    block_size=self.block_size,
                )
                write_prefill(
                    inputs, request, chunk, idx.masked_fill(idx < 0, request.lc)
                )


def _publish_prefill_blocks(
    *,
    data: DeepGEMMPrefillData,
    kv: Tuple[torch.Tensor, torch.Tensor],
    topk: int,
    topk_blocks: int,
    block_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    from sglang.kernels.ops.attention.dsv4 import topk_transform_ragged_v2

    selected = data.empty_selection(topk)
    width = min(topk_blocks, -(-max(data.lens_per_request, default=0) // block_size))
    blocks = torch.full(
        (data.num_rows, width), -1, dtype=torch.int32, device=selected.device
    )
    for tile, logits in score_tiles(
        data,
        kv,
        width_align=math.lcm(4, block_size),
        scratch_bytes=lambda w: _publish_scratch_bytes(w, block_size, topk_blocks),
    ):
        _publish_tile_blocks(
            data=data,
            tile=tile,
            logits=logits,
            blocks=blocks[tile],
            topk_blocks=topk_blocks,
            block_size=block_size,
        )
        topk_transform_ragged_v2(
            logits,
            data.compress_lens[tile],
            out_offsets=data.request_starts[tile],
            out_indices=selected[tile],
        )
        # Free this tile's logits before the generator scores the next one.
        del logits
    return selected, blocks


def _publish_tile_blocks(
    *,
    data: DeepGEMMPrefillData,
    tile: slice,
    logits: torch.Tensor,
    blocks: torch.Tensor,
    topk_blocks: int,
    block_size: int,
) -> None:
    lens = data.compress_lens[tile]
    for rows, lc in _requests_in_tile(data, tile):
        # Through the request's last block: -inf past each row's length is the
        # padding select_candidate_block_ids would otherwise add in a copy.
        width = -(-lc // block_size) * block_size
        scores = logits[rows, :width]
        scores.masked_fill_(
            torch.arange(width, device=logits.device)[None, :] >= lens[rows, None],
            -torch.inf,
        )
        ids = select_candidate_block_ids(
            logits=scores,
            compress_lens=lens[rows, None],
            topk_blocks=topk_blocks,
            block_size=block_size,
        )
        blocks[rows, : ids.shape[1]] = ids


def _consume_prefill_blocks(
    *,
    data: DeepGEMMPrefillData,
    kv: Tuple[torch.Tensor, torch.Tensor],
    topk: int,
    blocks: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    selected = data.empty_selection(topk)
    scratch = (_consume_scratch_bytes(blocks.shape[1], block_size, topk), 0)
    for tile, logits in score_tiles(
        data, kv, width_align=block_size, scratch_bytes=lambda w: scratch
    ):
        _consume_tile_blocks(
            data=data,
            tile=tile,
            logits=logits,
            blocks=blocks[tile],
            block_size=block_size,
            out=selected[tile],
        )
        # Free this tile's logits before the generator scores the next one.
        del logits
    return selected


def _consume_tile_blocks(
    *,
    data: DeepGEMMPrefillData,
    tile: slice,
    logits: torch.Tensor,
    blocks: torch.Tensor,
    block_size: int,
    out: torch.Tensor,
) -> None:
    lens = data.compress_lens[tile]
    starts = data.request_starts[tile]
    for rows, lc in _requests_in_tile(data, tile):
        # Columns past a row's length hold garbage; topk_among_blocks drops them.
        # The slice is block-aligned (the tile is), so the op gathers in place.
        positions = topk_among_blocks(
            logits[rows, : -(-lc // block_size) * block_size],
            lens[rows],
            blocks[rows],
            out.shape[1],
            block_size=block_size,
        )
        out[rows] = torch.where(positions >= 0, positions + starts[rows, None], -1).to(
            torch.int32
        )


def _publish_scratch_bytes(
    width: int, block_size: int, topk_blocks: int
) -> Tuple[int, int]:
    """Scratch of ``_publish_tile_blocks`` for a tile ``width`` wide, as bytes a
    row and bytes a tile. A row: the bool causal mask (a byte a column), the fp32
    block maxima, their masked copy and the newest-block mask (9 bytes a block),
    and the top-k over the blocks with its sort (32 bytes a kept block). A tile:
    one request's int64 column or block indices at a time (8 bytes a column)."""
    num_blocks = -(-width // block_size)
    per_row = width + 9 * num_blocks + 32 * min(topk_blocks, num_blocks)
    return per_row, 8 * width


def _consume_scratch_bytes(num_blocks: int, block_size: int, topk: int) -> int:
    """Per-row scratch of ``_consume_tile_blocks``, whatever the tile width:
    ``topk_among_blocks``'s gathered candidates and the two bool masks its drop
    mask is built from (6 bytes a candidate position), its int64 block ids,
    clamped ids, room and their temporaries (40 bytes a block), and the int64
    picks with their transforms (96 bytes a pick)."""
    return num_blocks * (6 * block_size + 40) + 96 * topk


def _requests_in_tile(data: DeepGEMMPrefillData, tile: slice):
    start = 0
    for num_rows, lc in zip(data.rows_per_request, data.lens_per_request):
        end = start + num_rows
        begin, stop = max(start, tile.start), min(end, tile.stop)
        if begin < stop and lc > 0:
            yield slice(begin - tile.start, stop - tile.start), lc
        start = end
