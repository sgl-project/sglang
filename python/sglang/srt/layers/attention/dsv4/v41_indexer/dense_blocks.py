"""The candidate scheme carried by dense block ids: a source publishes the ids of
its best blocks, a consumer takes its top-k among them."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

import msgspec
import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.dsv4.candidate_blocks import (
    amax_topk_blocks,
    candidate_block_mask,
    candidate_topk,
    select_candidate_block_ids,
    topk_among_blocks,
)
from sglang.kernels.ops.attention.dsv4.index_logits import FLAT_LOGITS_WIDTH_ALIGN
from sglang.srt.layers.attention.dsv4.metadata import expand_index_page_table
from sglang.srt.utils import is_xpu

from .scoring import (
    DeepGEMMPrefillData,
    PagedDecodeScores,
    _index_q_and_weights,
    decode_scores,
    get_deep_gemm_decode_data,
    get_deep_gemm_prefill_data,
    get_index_k_cache,
    prefill_requests,
    score_tiles,
    select_decode,
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


@triton.jit
def _xpu_write_candidate_selection(
    INDICES,
    PAGE_TABLE,
    OUT_PAGES,
    OUT_RAW,
    TOPK: tl.constexpr,
    PAGE_STRIDE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    OUT_PAGE_ROW_STRIDE: tl.constexpr,
    OUT_PAGE_COL_STRIDE: tl.constexpr,
    OUT_RAW_ROW_STRIDE: tl.constexpr,
    OUT_RAW_COL_STRIDE: tl.constexpr,
    WRITE_PAGES: tl.constexpr,
    WRITE_RAW: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.program_id(1) * TILE + tl.arange(0, TILE)
    indices = tl.load(INDICES + row * TOPK + columns, columns < TOPK, -1)
    if WRITE_PAGES:
        page = tl.load(
            PAGE_TABLE + row * PAGE_STRIDE + indices // PAGE_SIZE,
            (columns < TOPK) & (indices >= 0),
            0,
        )
        slots = page * PAGE_SIZE + indices % PAGE_SIZE
        tl.store(
            OUT_PAGES + row * OUT_PAGE_ROW_STRIDE + columns * OUT_PAGE_COL_STRIDE,
            tl.where(indices >= 0, slots, -1),
            columns < TOPK,
        )
    if WRITE_RAW:
        tl.store(
            OUT_RAW + row * OUT_RAW_ROW_STRIDE + columns * OUT_RAW_COL_STRIDE,
            indices,
            columns < TOPK,
        )


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
        if is_xpu():
            return self._xpu_publish_prefill(inputs)
        if self.use_deep_gemm_prefill:
            return self._deep_gemm_publish_prefill(inputs)
        return self._torch_publish_prefill(inputs)

    def consume_prefill(
        self,
        inputs: PrefillInputs,
        published: Optional[BlockIds],
    ) -> None:
        if is_xpu():
            assert published is not None
            self._xpu_consume_prefill(inputs, published)
            return
        if self.use_deep_gemm_prefill:
            self._deep_gemm_consume_prefill(inputs, published)
        else:
            self._torch_consume_prefill(inputs, published)

    def publish_decode(self, inputs: DecodeInputs):
        if is_xpu():
            return self._xpu_decode(inputs, None)
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
        if is_xpu():
            assert published is not None
            self._xpu_decode(inputs, published)
            return
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

    def _xpu_paged_rows(
        self,
        q,
        scales,
        weights,
        lengths,
        cache,
        page_table,
        max_len,
        page_size,
        out_pages,
        out_raw,
        published_blocks,
    ):
        from sgl_kernel import (
            fp4_paged_mqa_logits,
            topk_transform,
            topk_transform_ragged,
        )

        rows = q.shape[0]
        result = None
        if published_blocks is None:
            result = torch.full(
                (rows, min(self.topk_blocks, triton.cdiv(max_len, self.block_size))),
                -1,
                dtype=torch.int32,
                device=q.device,
            )
        if rows == 0 or max_len == 0:
            return result
        topk = out_raw.shape[1] if out_raw is not None else out_pages.shape[1]
        rows_per_chunk = max(1, (64 << 20) // max(max_len * 4, 1))
        for start in range(0, rows, rows_per_chunk):
            end = min(start + rows_per_chunk, rows)
            chunk_lengths = lengths[start:end].to(torch.int32).contiguous()
            chunk_pages = page_table[start:end].to(torch.int32).contiguous()
            scores = fp4_paged_mqa_logits(
                (q[start:end], scales[start:end]),
                cache,
                weights[start:end].float(),
                chunk_lengths,
                chunk_pages,
                None,
                max_len,
                clean_logits=False,
            )
            if result is not None:
                visible = (
                    torch.arange(max_len, device=q.device)[None, :]
                    < chunk_lengths[:, None]
                )
                scores = scores.masked_fill(~visible, -torch.inf)
                result[start:end] = select_candidate_block_ids(
                    scores, chunk_lengths[:, None], self.topk_blocks, self.block_size
                )
                if out_pages is not None:
                    topk_transform(
                        scores,
                        chunk_lengths,
                        chunk_pages,
                        out_pages[start:end],
                        page_size,
                        out_raw[start:end] if out_raw is not None else None,
                    )
                else:
                    assert out_raw is not None
                    topk_transform_ragged(
                        scores,
                        chunk_lengths,
                        out_raw[start:end],
                        torch.zeros_like(chunk_lengths),
                    )
                continue

            selected = topk_among_blocks(
                scores,
                chunk_lengths,
                published_blocks[start:end],
                topk,
                block_size=self.block_size,
            )
            if topk:
                page_out = out_pages[start:end] if out_pages is not None else None
                raw_out = out_raw[start:end] if out_raw is not None else None
                _xpu_write_candidate_selection[(end - start, triton.cdiv(topk, 256))](
                    selected,
                    chunk_pages,
                    page_out,
                    raw_out,
                    topk,
                    chunk_pages.stride(0),
                    page_size,
                    page_out.stride(0) if page_out is not None else 0,
                    page_out.stride(1) if page_out is not None else 0,
                    raw_out.stride(0) if raw_out is not None else 0,
                    raw_out.stride(1) if raw_out is not None else 0,
                    page_out is not None,
                    raw_out is not None,
                    256,
                )
        return result

    def _xpu_publish_prefill(self, inputs: PrefillInputs):
        return self._xpu_prefill_impl(inputs, None)

    def _xpu_consume_prefill(self, inputs: PrefillInputs, published: BlockIds) -> None:
        self._xpu_prefill_impl(inputs, published)

    def _xpu_prefill_impl(self, inputs: PrefillInputs, published: Optional[BlockIds]):
        inputs.reset_outputs()
        assert inputs.seq_lens_cpu is not None and inputs.rows_per_request is not None
        rows = inputs.positions.numel()
        max_len = max(
            (length // inputs.compress_ratio for length in inputs.seq_lens_cpu),
            default=0,
        )
        if published is None:
            blocks = torch.full(
                (rows, min(self.topk_blocks, triton.cdiv(max_len, self.block_size))),
                -1,
                dtype=torch.int32,
                device=inputs.positions.device,
            )
        else:
            assert published.blocks.shape[0] == rows
            blocks = published.blocks
        if rows == 0 or max_len == 0:
            if published is None:
                return BlockIds(blocks=blocks, rows_per_request=inputs.rows_per_request)
            return None
        q, scales, weights = _index_q_and_weights(
            indexer=inputs.indexer,
            x=inputs.x,
            q_lora=inputs.q_lora,
            freqs_cis=inputs.freqs_cis,
            positions=inputs.positions,
        )
        lengths = ((inputs.positions + 1) // inputs.compress_ratio).to(torch.int32)
        page_size = self.token_to_kv_pool.get_index_k_page_size(inputs.compress_ratio)
        pages = expand_index_page_table(
            inputs.kv_page_table,
            full_page_size=self.token_to_kv_pool.page_size,
            compress_ratio=inputs.compress_ratio,
            index_page_size=page_size,
        )
        cache = get_index_k_cache(
            token_to_kv_pool=self.token_to_kv_pool,
            layer_id=inputs.layer_id,
            page_size=page_size,
        )
        start = 0
        for request, request_rows in enumerate(inputs.rows_per_request):
            end = start + request_rows
            if request_rows:
                published_chunk = None if published is None else blocks[start:end]
                result = self._xpu_paged_rows(
                    q[start:end].unsqueeze(1),
                    scales[start:end].unsqueeze(1),
                    weights[start:end],
                    lengths[start:end],
                    cache,
                    pages[request : request + 1].expand(request_rows, -1),
                    max_len,
                    page_size,
                    (
                        inputs.out_page_indices[start:end]
                        if inputs.out_page_indices is not None
                        else None
                    ),
                    inputs.out_raw_indices[start:end],
                    published_chunk,
                )
                if published is None:
                    blocks[start:end].copy_(result)
            start = end
        assert start == rows
        if published is None:
            return BlockIds(blocks=blocks, rows_per_request=inputs.rows_per_request)
        return None

    def _xpu_decode(self, inputs: DecodeInputs, published: Optional[BlockIds]):
        inputs.reset_outputs()
        metadata = inputs.paged_metadata
        rows = inputs.positions.numel()
        if published is not None:
            assert published.blocks.shape[0] == rows
        if rows == 0 or metadata.max_compressed_seq_len == 0:
            if published is None:
                return BlockIds(
                    blocks=torch.empty(
                        (rows, 0), dtype=torch.int32, device=inputs.positions.device
                    )
                )
            return None
        data = get_deep_gemm_decode_data(inputs, self.token_to_kv_pool)
        n = metadata.rows_per_request
        pages = metadata.page_table[::n].repeat_interleave(n, dim=0).contiguous()
        blocks = self._xpu_paged_rows(
            data.q_fp4,
            data.q_sf,
            data.weights,
            metadata.compressed_seq_lens.reshape(-1),
            data.k_cache,
            pages,
            metadata.max_compressed_seq_len,
            metadata.compressed_page_size,
            inputs.out_page_indices,
            None,
            None if published is None else published.blocks,
        )
        return BlockIds(blocks=blocks) if published is None else None

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

    # ---------- torch: per-request scores ----------

    def _torch_publish_prefill(self, inputs: PrefillInputs):
        picked = []  # (query rows, their block ids) per scored chunk
        for request, chunks in prefill_requests(
            inputs=inputs,
            token_to_kv_pool=self.token_to_kv_pool,
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
) -> Tuple[torch.Tensor, torch.Tensor]:
    selected = data.empty_selection(topk)
    width = min(topk_blocks, -(-max(data.lens_per_request, default=0) // block_size))
    blocks = torch.full(
        (data.num_rows, width), -1, dtype=torch.int32, device=selected.device
    )
    for tile, logits in score_tiles(data, kv):
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
