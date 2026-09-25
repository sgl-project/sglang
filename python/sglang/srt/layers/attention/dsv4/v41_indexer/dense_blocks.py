"""The candidate scheme carried by dense block ids.

A source scores its whole context, keeps its own top-k and publishes the ids of
its best blocks; a consumer scores its whole context too, gathers the scores of
the published blocks only and takes its top-k among them. Scores come from
DeepGEMM's flattened-K logits or from torch, per backend configuration; the
block math is the torch one in ``candidate_blocks``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

import msgspec
import torch

from sglang.kernels.ops.attention.dsv4.candidate_blocks import (
    get_tail_row_indices,
    select_candidate_block_ids,
    topk_among_blocks,
)

from .scoring import (
    DeepGEMMPrefillData,
    decode_scores,
    get_deep_gemm_prefill_data,
    get_flat_index_k,
    prefill_requests,
    score_tiles,
    write_decode,
    write_prefill,
)
from .types import (
    CandidateMetadata,
    DecodeInputs,
    PrefillInputs,
    Selection,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool


class BlockIds(CandidateMetadata, msgspec.Struct):
    # [rows, n] int32 block ids of each query row, unordered, -1 padded; n is the
    # largest kept count of the batch, at most topk_blocks
    blocks: torch.Tensor
    # prefill: query rows of each request in row order, for the late-layer tail
    rows_per_request: Optional[List[int]] = None

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

    def publish_prefill(self, inputs: PrefillInputs, out: Selection):
        if self.use_deep_gemm_prefill:
            return self._deep_gemm_publish_prefill(inputs, out)
        return self._torch_publish_prefill(inputs, out)

    def consume_prefill(
        self,
        inputs: PrefillInputs,
        published: Optional[BlockIds],
        out: Selection,
    ) -> None:
        if self.use_deep_gemm_prefill:
            self._deep_gemm_consume_prefill(inputs, published, out)
        else:
            self._torch_consume_prefill(inputs, published, out)

    def publish_decode(self, inputs: DecodeInputs, out: Selection):
        d = self._decode_scores(inputs, out)
        if d is None:
            return None
        published = BlockIds(
            blocks=select_candidate_block_ids(
                d.scores,
                d.lens[:, None],
                topk_blocks=self.topk_blocks,
                block_size=self.block_size,
            )
        )
        k = min(inputs.indexer.index_topk, d.lmax)
        idx = d.scores.topk(k, dim=-1, sorted=False).indices
        write_decode(out, d, idx)
        return published

    def consume_decode(
        self,
        inputs: DecodeInputs,
        published: Optional[BlockIds],
        out: Selection,
    ) -> None:
        d = self._decode_scores(inputs, out)
        if d is None:
            return
        assert published is not None and published.blocks.shape[0] == d.bs
        k = min(inputs.indexer.index_topk, d.lmax)
        idx = topk_among_blocks(
            d.scores, d.lens, published.blocks, k, block_size=self.block_size
        )
        write_decode(out, d, idx.masked_fill(idx < 0, d.lmax))

    # ---------- DeepGEMM: flattened-K scores ----------

    def _deep_gemm_publish_prefill(self, inputs: PrefillInputs, out: Selection):
        out.reset()
        data = get_deep_gemm_prefill_data(inputs, self.req_to_token)
        if data is None:
            return None
        selected, blocks = _publish_prefill_blocks(
            data=data,
            kv=self._get_flat_index_k(inputs, data),
            topk=inputs.indexer.index_topk,
            topk_blocks=self.topk_blocks,
            block_size=self.block_size,
        )
        data.write_selection(selected=selected, out=out)
        return BlockIds(blocks=blocks, rows_per_request=data.rows_per_request)

    def _deep_gemm_consume_prefill(
        self,
        inputs: PrefillInputs,
        published: Optional[BlockIds],
        out: Selection,
    ) -> None:
        out.reset()
        data = get_deep_gemm_prefill_data(inputs, self.req_to_token)
        if data is None:
            return
        assert published is not None
        selected = _consume_prefill_blocks(
            data=data,
            kv=self._get_flat_index_k(inputs, data),
            topk=inputs.indexer.index_topk,
            blocks=published.blocks,
            block_size=self.block_size,
        )
        data.write_selection(selected=selected, out=out)

    def _get_flat_index_k(self, inputs: PrefillInputs, data: DeepGEMMPrefillData):
        return get_flat_index_k(
            data=data, token_to_kv_pool=self.token_to_kv_pool, layer_id=inputs.layer_id
        )

    # ---------- torch: per-request scores ----------

    def _torch_publish_prefill(self, inputs: PrefillInputs, out: Selection):
        picked = []  # (query rows, their block ids) per scored chunk
        for request, chunks in self._prefill_requests(inputs, out):
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
                write_prefill(out, request, chunk, idx)
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
        out: Selection,
    ) -> None:
        assert published is not None
        for request, chunks in self._prefill_requests(inputs, out):
            for chunk in chunks:
                idx = topk_among_blocks(
                    chunk.scores,
                    chunk.lens,
                    published.blocks[chunk.tok],
                    request.k,
                    block_size=self.block_size,
                )
                write_prefill(out, request, chunk, idx.masked_fill(idx < 0, request.lc))

    def _prefill_requests(self, inputs: PrefillInputs, out: Selection):
        return prefill_requests(
            inputs=inputs,
            out=out,
            token_to_kv_pool=self.token_to_kv_pool,
            req_to_token=self.req_to_token,
        )

    def _decode_scores(self, inputs: DecodeInputs, out: Selection):
        return decode_scores(
            inputs=inputs,
            out=out,
            token_to_kv_pool=self.token_to_kv_pool,
            req_to_token=self.req_to_token,
        )


def _publish_prefill_blocks(
    *,
    data: DeepGEMMPrefillData,
    kv: Tuple[torch.Tensor, torch.Tensor],
    topk: int,
    topk_blocks: int,
    block_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The source layer's own plain top-k, plus the ids of each row's best blocks:
    ``[rows, min(topk_blocks, ceil(max lc / block_size))]`` int32, -1 padded."""
    from sglang.kernels.ops.attention.dsv4 import topk_transform_ragged_v2

    selected = data.empty_selection(topk)
    width = min(topk_blocks, -(-max(data.lens_per_request, default=0) // block_size))
    blocks = torch.full(
        (data.num_rows, width), -1, dtype=torch.int32, device=selected.device
    )
    for tile, logits in score_tiles(data, kv, width_align=4):
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
        scores = logits[rows, :lc]
        scores.masked_fill_(
            torch.arange(lc, device=logits.device)[None, :] >= lens[rows, None],
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
    """A consumer's top-k among the published blocks of its dense scores, as
    flattened-K columns, ``-1`` padded, unordered."""
    selected = data.empty_selection(topk)
    for tile, logits in score_tiles(data, kv, width_align=4):
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
        positions = topk_among_blocks(
            logits[rows, :lc],
            lens[rows],
            blocks[rows],
            out.shape[1],
            block_size=block_size,
        )
        out[rows] = torch.where(positions >= 0, positions + starts[rows, None], -1).to(
            torch.int32
        )


def _requests_in_tile(data: DeepGEMMPrefillData, tile: slice):
    """``(rows, lc)`` of each request with a visible position and rows in
    ``tile``: its rows within the tile and its context length."""
    start = 0
    for num_rows, lc in zip(data.rows_per_request, data.lens_per_request):
        end = start + num_rows
        begin, stop = max(start, tile.start), min(end, tile.stop)
        if begin < stop and lc > 0:
            yield slice(begin - tile.start, stop - tile.start), lc
        start = end
