"""The candidate scheme carried by dense block ids: a source publishes the ids of
its best blocks, a consumer takes its top-k among them."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

import msgspec
import torch

from sglang.kernels.ops.attention.dsv4.candidate_blocks import (
    amax_topk_blocks,
    candidate_block_mask,
    candidate_row_lens,
    candidate_topk,
    select_candidate_block_ids,
    select_candidate_block_ids_triton,
    topk_among_blocks,
)
from sglang.kernels.ops.attention.dsv4.candidate_table import CANDIDATE_BLOCK_SIZE
from sglang.kernels.ops.attention.dsv4.index_logits import FLAT_LOGITS_WIDTH_ALIGN
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_platform

from .scoring import (
    DeepGEMMPrefillData,
    PagedDecodeScores,
    decode_scores,
    dense_prefill_topk,
    get_deep_gemm_prefill_data,
    prefill_requests,
    score_tiles,
    select_decode,
    select_prefill_block_ids,
    topk_ragged,
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

# A compact DeepGEMM call packs one contiguous K range per query. Bound the
# repeated K, including requests in the batch with disjoint candidate sets.
_MAX_COMPACT_PREFILL_KEYS = 1 << 16


class BlockIds(CandidateMetadata, msgspec.Struct):
    # [rows, n] int32 block ids of each query row, unordered, -1 padded; n is the
    # largest kept count of the batch, at most topk_blocks. Hopper FP8 prefill
    # sorts valid IDs once at publish time, ahead of the padding.
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
        use_deep_gemm_publish: bool,
        use_deep_gemm_consume: bool,
        use_sm90_deep_gemm_consume: bool = False,
    ) -> None:
        self.token_to_kv_pool = token_to_kv_pool
        self.req_to_token = req_to_token
        self.topk_blocks = candidate_topk_blocks
        self.block_size = candidate_block_size
        self.use_deep_gemm_publish = use_deep_gemm_publish
        self.use_deep_gemm_consume = use_deep_gemm_consume
        self.use_sm90_deep_gemm_consume = use_sm90_deep_gemm_consume

    def publish_prefill(self, inputs: PrefillInputs):
        if self.use_deep_gemm_publish:
            return self._deep_gemm_publish_prefill(inputs)
        return self._torch_publish_prefill(inputs)

    def consume_prefill(
        self,
        inputs: PrefillInputs,
        published: Optional[BlockIds],
    ) -> None:
        if self.use_deep_gemm_consume:
            self._deep_gemm_consume_prefill(inputs, published)
            return
        if (
            self.use_sm90_deep_gemm_consume
            and published is not None
            and published.blocks.shape[1] > 0
            and inputs.seq_lens_cpu is not None
        ):
            candidate_width = published.blocks.shape[1] * self.block_size
            context_width = max(inputs.seq_lens_cpu, default=0) // inputs.compress_ratio
            compact = 2 * candidate_width <= context_width
            if (
                not compact
                or published.blocks.shape[0] * candidate_width
                <= _MAX_COMPACT_PREFILL_KEYS
            ):
                self._sm90_consume_prefill(inputs, published, compact=compact)
                return
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
        data = get_deep_gemm_prefill_data(inputs)
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
            fast_block_topk=self.use_sm90_deep_gemm_consume,
        )
        data.write_selection(selected, inputs)
        return BlockIds(blocks=blocks, rows_per_request=data.rows_per_request)

    def _deep_gemm_consume_prefill(
        self,
        inputs: PrefillInputs,
        published: Optional[BlockIds],
    ) -> None:
        inputs.reset_outputs()
        data = get_deep_gemm_prefill_data(inputs)
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

    def _sm90_consume_prefill(
        self, inputs: PrefillInputs, published: BlockIds, *, compact: bool
    ) -> None:
        inputs.reset_outputs()
        data = get_deep_gemm_prefill_data(inputs)
        if data is None:
            return
        blocks = published.blocks[: data.num_rows]
        if compact:
            selected = data.empty_selection(inputs.indexer.index_topk)
            positions, lens = _pack_candidate_positions(
                blocks, data.compress_lens, self.block_size
            )
            columns = data.request_starts[:, None] + positions
            slots = data.k_slots[columns.clamp_max(data.k_slots.numel() - 1)]
            kv = self.token_to_kv_pool.get_low_ratio_index_k_fp4(
                inputs.layer_id, slots.flatten()
            )
            width = positions.shape[1]
            packed = msgspec.structs.replace(
                data,
                request_starts=torch.arange(
                    data.num_rows, device=positions.device, dtype=torch.int32
                )
                * width,
                lens_per_request=[width],
                compress_lens=lens,
            )
            for tile, logits in score_tiles(packed, kv):
                _select_candidate_positions(
                    logits,
                    positions[tile],
                    lens[tile],
                    data.request_starts[tile],
                    selected[tile],
                )
            data.write_selection(selected, inputs)
            return

        kv = self.token_to_kv_pool.get_low_ratio_index_k_fp4(
            inputs.layer_id, data.k_slots
        )
        if max(data.lens_per_request, default=0) <= blocks.shape[1] * self.block_size:
            # The source selected every visible block. Reuse full top-k directly.
            dense_prefill_topk(data, kv, out=inputs.out_raw_indices[: data.num_rows])
            data.write_page_indices(inputs)
            return
        selected = data.empty_selection(inputs.indexer.index_topk)
        for tile, logits in score_tiles(data, kv):
            positions, lens = _pack_candidate_positions(
                blocks[tile], data.compress_lens[tile], self.block_size
            )
            compact_scores = logits.gather(1, positions.long())
            _select_candidate_positions(
                compact_scores,
                positions,
                lens,
                data.request_starts[tile],
                selected[tile],
            )
        data.write_selection(selected, inputs)

    # ---------- torch: per-request scores ----------

    def _torch_publish_prefill(self, inputs: PrefillInputs):
        picked = []  # (query rows, their block ids) per scored chunk
        select_blocks = (
            select_candidate_block_ids_triton
            if envs.SGLANG_OPT_DSV41_TRITON_PREFILL_SELECTION.get()
            and get_platform().is_sm90
            else select_candidate_block_ids
        )
        for request, chunks in prefill_requests(
            inputs=inputs,
            token_to_kv_pool=self.token_to_kv_pool,
        ):
            for chunk in chunks:
                picked.append(
                    (
                        chunk.tok,
                        select_blocks(
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
        if published is None:  # a DeepGEMM publish with no visible position
            inputs.reset_outputs()
            return
        for request, chunks in prefill_requests(
            inputs=inputs,
            token_to_kv_pool=self.token_to_kv_pool,
            candidate_blocks=published.blocks,
            candidate_block_size=self.block_size,
        ):
            for chunk in chunks:
                if chunk.candidate_only:
                    idx = candidate_topk(
                        chunk.scores,
                        published.blocks[chunk.tok],
                        request.k,
                        self.block_size,
                    )
                else:
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
    fast_block_topk: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    selected = data.empty_selection(topk)
    width = min(topk_blocks, -(-max(data.lens_per_request, default=0) // block_size))
    blocks = torch.full(
        (data.num_rows, width), -1, dtype=torch.int32, device=selected.device
    )
    zero_offsets = torch.zeros_like(data.compress_lens) if fast_block_topk else None
    for tile, logits in score_tiles(data, kv):
        if fast_block_topk and block_size == CANDIDATE_BLOCK_SIZE and width:
            lens = data.compress_lens[tile]
            nblocks, _ = candidate_row_lens(lens, width)
            select_prefill_block_ids(
                logits, lens, nblocks, blocks[tile], offsets=zero_offsets[tile]
            )
        else:
            _publish_tile_blocks(
                data=data,
                tile=tile,
                logits=logits,
                blocks=blocks[tile],
                topk_blocks=topk_blocks,
                block_size=block_size,
            )
        topk_ragged(
            logits,
            data.compress_lens[tile],
            out=selected[tile],
            offsets=data.request_starts[tile],
        )
        # Free this tile's logits before the generator scores the next one.
        del logits
    if fast_block_topk and block_size == CANDIDATE_BLOCK_SIZE and width:
        padding = torch.iinfo(torch.int32).max
        blocks = blocks.masked_fill_(blocks < 0, padding).sort(dim=-1).values
        blocks.masked_fill_(blocks == padding, -1)
    return selected, blocks


def _pack_candidate_positions(
    blocks: torch.Tensor, lens: torch.Tensor, block_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand the publisher's sorted block IDs, with padding at the end."""
    offsets = torch.arange(block_size, device=blocks.device, dtype=blocks.dtype)
    positions = (blocks[:, :, None] * block_size + offsets).flatten(1)
    valid = (positions >= 0) & (positions < lens[:, None])
    return positions.masked_fill(~valid, 0), valid.sum(dim=1, dtype=torch.int32)


def _select_candidate_positions(
    logits: torch.Tensor,
    positions: torch.Tensor,
    lens: torch.Tensor,
    starts: torch.Tensor,
    out: torch.Tensor,
) -> None:
    topk_ragged(logits, lens, out=out, offsets=torch.zeros_like(lens))
    columns = positions.gather(1, out.clamp_min(0).long()) + starts[:, None]
    out.copy_(columns.masked_fill(out < 0, -1))


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
    selected = data.empty_selection(topk)
    # the block-aligned slices of `_consume_tile_blocks` stay within the logits width
    assert FLAT_LOGITS_WIDTH_ALIGN % block_size == 0
    for tile, logits in score_tiles(data, kv):
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


def _requests_in_tile(data: DeepGEMMPrefillData, tile: slice):
    start = 0
    for num_rows, lc in zip(data.rows_per_request, data.lens_per_request):
        end = start + num_rows
        begin, stop = max(start, tile.start), min(end, tile.stop)
        if begin < stop and lc > 0:
            yield slice(begin - tile.start, stop - tile.start), lc
        start = end
