"""The candidate scheme carried by a DeepGEMM sparse table (SM100).

A source publishes the block ids it kept plus the pool slots and the schedule
``fp8_fp4_paged_sparse_mqa_logits`` needs for them, and a consumer scores only
those blocks straight out of the index-K pool.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

import msgspec
import torch

from sglang.kernels.ops.attention.dsv4.candidate_blocks import (
    amax_topk_blocks,
    candidate_row_lens,
    get_tail_row_indices,
)
from sglang.kernels.ops.attention.dsv4.candidate_table import (
    CANDIDATE_BLOCK_SIZE,
    amax8_varlen,
    build_sparse_indexer_schedule,
    sort_candidate_blocks,
    sparse_logits,
    topk_transform_sparse,
)
from sglang.kernels.ops.attention.dsv4.topk import (
    topk_transform_paged_v2,
    topk_transform_ragged_v2,
)
from sglang.srt.layers.attention.dsv4.indexer import (
    deep_gemm_fp4_paged_mqa_logits,
    topk_transform_paged_from_metadata,
)
from sglang.srt.layers.attention.dsv4.metadata import expand_index_page_table

from .scoring import (
    DeepGEMMDecodeData,
    DeepGEMMPrefillData,
    get_deep_gemm_decode_data,
    get_deep_gemm_prefill_data,
    get_flat_index_k,
    get_index_k_cache,
    score_tiles,
)
from .types import (
    CandidateMetadata,
    DecodeInputs,
    PrefillInputs,
    Selection,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool


class _SparseTable(CandidateMetadata, msgspec.Struct):
    # [rows, topk_blocks] int32: ascending logical block ids, valid for the first
    # min(topk_blocks, ceil(seq_len / 8)) entries of a row; DeepGEMM reads only those
    blocks: torch.Tensor
    # DeepGEMM's schedule metadata (uint8) for them
    schedule: torch.Tensor
    # [rows, topk_blocks] int32: the same blocks as pool slots / 8, so a consumer's
    # top-k maps column j of the sparse row to slot phys_blocks[b, j // 8] * 8 + j % 8
    # with the plain page-table transform at page size 8
    phys_blocks: torch.Tensor
    # [rows] int32: length of each row of the sparse logits: the published blocks
    # laid out block by block, the newest possibly partial (`candidate_row_lens`)
    valid_lens: torch.Tensor
    # recorded on the side stream once the fields above are complete
    ready: torch.cuda.Event


class _SparsePrefillTable(_SparseTable):
    """Published by a prefill chunk's candidate source: one row per query token.
    The inputs stay attached so the late-layer tail can rebuild the table for
    its rows (the DeepGEMM schedule is per row set, it cannot be sliced)."""

    compress_lens: torch.Tensor  # [rows] int32
    page_table: torch.Tensor  # [rows, index pages] int32
    request_ids: torch.Tensor  # [rows] int32
    q_dtype: torch.dtype
    page_size: int  # index-K pool page size (slots)
    rows_per_request: List[int]  # rows of each request, in row order
    topk_blocks: int

    def tail(self, rows_per_request: List[int]) -> _SparsePrefillTable:
        rows = get_tail_row_indices(
            full_rows_per_request=self.rows_per_request,
            tail_rows_per_request=rows_per_request,
            device=self.blocks.device,
        )
        compress_lens = self.compress_lens[rows].contiguous()
        _, valid_lens = candidate_row_lens(compress_lens, self.topk_blocks)
        return _build_prefill_table(
            blocks=self.blocks[rows].contiguous(),
            compress_lens=compress_lens,
            page_table=self.page_table[rows].contiguous(),
            page_size=self.page_size,
            request_ids=self.request_ids[rows].contiguous(),
            rows_per_request=rows_per_request,
            q_dtype=self.q_dtype,
            valid_lens=valid_lens,
            topk_blocks=self.topk_blocks,
        )


class SparseTableBackend:
    def __init__(
        self,
        *,
        token_to_kv_pool: DeepSeekV4TokenToKVPool,
        req_to_token: torch.Tensor,
        page_size: int,
        candidate_topk_blocks: int,
        candidate_block_size: int,
    ) -> None:
        assert candidate_block_size == CANDIDATE_BLOCK_SIZE
        self.token_to_kv_pool = token_to_kv_pool
        self.req_to_token = req_to_token
        self.page_size = page_size
        self.topk_blocks = candidate_topk_blocks
        self.alt_stream = torch.cuda.Stream()
        self._cached_row_ids: Optional[torch.Tensor] = None
        # captured graphs keep reading the buffers they saw
        self._retired_cached_row_ids: list = []

    def publish_prefill(self, inputs: PrefillInputs, out: Selection):
        out.reset()
        data = self._get_prefill_data(inputs)
        if data is None:
            return None
        selected = data.empty_selection(inputs.indexer.index_topk)
        index_page_size = self.token_to_kv_pool.get_index_k_page_size(
            inputs.compress_ratio
        )
        table = publish_prefill_table(
            data=data,
            kv=get_flat_index_k(
                data=data,
                token_to_kv_pool=self.token_to_kv_pool,
                layer_id=inputs.layer_id,
            ),
            index_page_table=expand_index_page_table(
                inputs.kv_page_table[: data.num_rows],
                full_page_size=self.page_size,
                compress_ratio=inputs.compress_ratio,
                index_page_size=index_page_size,
            ).contiguous(),
            index_page_size=index_page_size,
            topk_blocks=self.topk_blocks,
            out_positions=selected,
        )
        data.write_selection(selected=selected, out=out)
        return table

    def consume_prefill(
        self,
        inputs: PrefillInputs,
        published: Optional[_SparsePrefillTable],
        out: Selection,
    ) -> None:
        out.reset()
        # Scores only the published blocks, straight out of the pool.
        data = self._get_prefill_data(inputs)
        if data is None:
            return
        assert published is not None
        selected = data.empty_selection(inputs.indexer.index_topk)
        select_prefill_table(
            table=published,
            data=data,
            k_cache=get_index_k_cache(
                token_to_kv_pool=self.token_to_kv_pool,
                layer_id=inputs.layer_id,
                page_size=self.token_to_kv_pool.get_index_k_page_size(
                    inputs.compress_ratio
                ),
            ),
            out_positions=selected,
        )
        data.write_selection(selected=selected, out=out)

    def publish_decode(self, inputs: DecodeInputs, out: Selection):
        """The source layer's own plain top-k, plus the table for the consumers."""
        data = get_deep_gemm_decode_data(inputs, self.token_to_kv_pool)
        metadata = inputs.paged_metadata
        seq_lens = metadata.compressed_seq_lens.reshape(-1)
        if isinstance(metadata.deep_gemm_metadata, list):
            return self._publish_decode_chunked(inputs, data, seq_lens, out)
        logits = deep_gemm_fp4_paged_mqa_logits(
            (data.q_fp4, data.q_sf),
            data.k_cache,
            data.weights,
            metadata.compressed_seq_lens,
            metadata.page_table,
            metadata.deep_gemm_metadata,
            metadata.max_compressed_seq_len,
        )
        main_stream = torch.cuda.current_stream()
        self.alt_stream.wait_stream(main_stream)
        # The block-selection chain reads logits after the main stream moves on.
        logits.record_stream(self.alt_stream)
        # TODO(candidate): one kernel for both selections below (dense logits read once)
        topk_transform_paged_v2(
            logits,
            seq_lens,
            metadata.page_table,
            out.page_indices,
            metadata.compressed_page_size,
            metadata.topk_metadata,
            out_raw_indices=out.raw_indices,
        )
        # per row: block count for the block top-k, sparse-row length for the consumers
        with torch.cuda.stream(self.alt_stream):
            nblocks, row_valid_lens = candidate_row_lens(seq_lens, self.topk_blocks)
            blocks = amax_topk_blocks(logits, seq_lens, nblocks, self.topk_blocks)
            # in place: ascending, INT32_MAX padded, plus the blocks as pool slots / 8
            phys_blocks = sort_candidate_blocks(
                blocks,
                seq_lens,
                metadata.page_table,
                metadata.compressed_page_size,
            )
            schedule = build_sparse_indexer_schedule(
                blocks,
                seq_lens,
                metadata.page_table,
                metadata.compressed_page_size,
                data.q_fp4.dtype,
                self._get_request_ids(inputs, data.q_fp4.shape[0], blocks.device),
            )
            ready = torch.cuda.Event()
            ready.record(self.alt_stream)
            return _SparseTable(
                blocks=blocks,
                schedule=schedule,
                phys_blocks=phys_blocks,
                valid_lens=row_valid_lens,
                ready=ready,
            )

    def _publish_decode_chunked(
        self,
        inputs: DecodeInputs,
        data: DeepGEMMDecodeData,
        seq_lens: torch.Tensor,
        out: Selection,
    ) -> _SparseTable:
        """Publish an eager forward whose dense logits are bounded by row chunks.
        It stays on the current stream so each chunk's full logits can be released
        before the next; CUDA-graph metadata always carries one schedule."""
        metadata = inputs.paged_metadata
        block_chunks = []
        phys_block_chunks = []
        valid_len_chunks = []
        topk_plans = metadata.topk_metadata_chunks
        assert not metadata.use_topk_v2 or topk_plans is not None
        for chunk_idx, (rows, plan) in enumerate(metadata.row_chunks()):
            logits = deep_gemm_fp4_paged_mqa_logits(
                (data.q_fp4[rows], data.q_sf[rows]),
                data.k_cache,
                data.weights[rows],
                metadata.compressed_seq_lens[rows],
                metadata.page_table[rows],
                plan,
                metadata.max_compressed_seq_len,
            )
            topk_transform_paged_from_metadata(
                logits,
                metadata,
                out.page_indices,
                out.raw_indices,
                rows=rows,
                topk_metadata=(
                    topk_plans[chunk_idx] if topk_plans is not None else None
                ),
            )
            chunk_seq_lens = seq_lens[rows]
            nblocks, row_valid_lens = candidate_row_lens(
                chunk_seq_lens, self.topk_blocks
            )
            blocks = amax_topk_blocks(logits, chunk_seq_lens, nblocks, self.topk_blocks)
            phys_blocks = sort_candidate_blocks(
                blocks,
                chunk_seq_lens,
                metadata.page_table[rows],
                metadata.compressed_page_size,
            )
            block_chunks.append(blocks)
            phys_block_chunks.append(phys_blocks)
            valid_len_chunks.append(row_valid_lens)

        blocks = torch.cat(block_chunks)
        schedule = build_sparse_indexer_schedule(
            blocks,
            seq_lens,
            metadata.page_table,
            metadata.compressed_page_size,
            data.q_fp4.dtype,
            self._get_request_ids(inputs, data.q_fp4.shape[0], blocks.device),
        )
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream())
        return _SparseTable(
            blocks=blocks,
            schedule=schedule,
            phys_blocks=torch.cat(phys_block_chunks),
            valid_lens=torch.cat(valid_len_chunks),
            ready=ready,
        )

    def consume_decode(
        self,
        inputs: DecodeInputs,
        published: Optional[_SparseTable],
        out: Selection,
    ) -> None:
        assert published is not None
        data = get_deep_gemm_decode_data(inputs, self.token_to_kv_pool)
        torch.cuda.current_stream().wait_event(published.ready)
        logits = sparse_logits(
            data.q_fp4,
            data.q_sf,
            data.k_cache,
            data.weights.to(torch.bfloat16),
            published.schedule,
            published.blocks.shape[1],
        )
        assert out.raw_indices is None
        topk_transform_sparse(
            logits, published.valid_lens, published.phys_blocks, out.page_indices
        )

    def _get_prefill_data(self, inputs: PrefillInputs) -> Optional[DeepGEMMPrefillData]:
        return get_deep_gemm_prefill_data(inputs, self.req_to_token)

    def _get_request_ids(self, inputs: DecodeInputs, rows: int, device: torch.device):
        if inputs.is_verify:
            return inputs.req_rows[:rows].to(torch.int32).contiguous()
        buf = self._cached_row_ids
        if buf is None or buf.numel() < rows:
            assert not torch.cuda.is_current_stream_capturing(), (
                f"row-id buffer grows to {rows} rows inside a CUDA graph capture; "
                "warm up with the largest row count first"
            )
            if buf is not None:
                self._retired_cached_row_ids.append(buf)
            size = max(8192, -(-rows // 8192) * 8192)
            buf = torch.arange(size, dtype=torch.int32, device=device)
            self._cached_row_ids = buf
        return buf[:rows]


def _build_prefill_table(
    *,
    blocks: torch.Tensor,
    compress_lens: torch.Tensor,
    page_table: torch.Tensor,
    page_size: int,
    request_ids: torch.Tensor,
    rows_per_request: List[int],
    q_dtype: torch.dtype,
    valid_lens: torch.Tensor,
    topk_blocks: int,
) -> _SparsePrefillTable:
    # in place: ascending, INT32_MAX padded, plus the blocks as pool slots / 8
    phys_blocks = sort_candidate_blocks(blocks, compress_lens, page_table, page_size)
    schedule = build_sparse_indexer_schedule(
        blocks, compress_lens, page_table, page_size, q_dtype, request_ids
    )
    ready = torch.cuda.Event()
    ready.record()
    return _SparsePrefillTable(
        blocks=blocks,
        schedule=schedule,
        phys_blocks=phys_blocks,
        valid_lens=valid_lens,
        ready=ready,
        compress_lens=compress_lens,
        page_table=page_table,
        request_ids=request_ids,
        q_dtype=q_dtype,
        page_size=page_size,
        rows_per_request=rows_per_request,
        topk_blocks=topk_blocks,
    )


# TODO(dark): support fusion of publish + topk of publish layer
def publish_prefill_table(
    *,
    data: DeepGEMMPrefillData,
    kv: Tuple[torch.Tensor, torch.Tensor],
    index_page_table: torch.Tensor,
    index_page_size: int,
    topk_blocks: int,
    out_positions: torch.Tensor,
) -> _SparsePrefillTable:
    """The source layer's own top-``k`` (``k = out_positions.shape[1]``) as
    flattened-K columns, ``-1`` padded, unordered, and the chunk's table: both
    from one pass over its dense scores, tile by tile. ``index_page_table`` is
    ``[rows, index pages]`` at ``index_page_size`` slots."""
    rows = data.num_rows
    device = data.q_fp4.device
    nblocks, valid_lens = candidate_row_lens(data.compress_lens, topk_blocks)
    blocks = torch.empty(rows, topk_blocks, dtype=torch.int32, device=device)
    # the block keys read the score rows through 32-byte vectors
    for tile, logits in score_tiles(data, kv, width_align=8):
        lens = data.compress_lens[tile]
        topk_transform_ragged_v2(
            logits,
            lens,
            out_offsets=data.request_starts[tile],
            out_indices=out_positions[tile],
        )
        # keys past a row's block count stay unset; the top-k reads a row
        # up to its block count only (v2 wants the stride a multiple of 4)
        width_blocks = (
            logits.shape[1] + CANDIDATE_BLOCK_SIZE - 1
        ) // CANDIDATE_BLOCK_SIZE
        keys = logits.new_empty(logits.shape[0], (width_blocks + 3) // 4 * 4)
        amax8_varlen(logits, lens, out=keys)
        topk_transform_ragged_v2(
            keys,
            nblocks[tile],
            out_offsets=torch.zeros(logits.shape[0], dtype=torch.int32, device=device),
            out_indices=blocks[tile],
        )
    request_ids = torch.repeat_interleave(
        torch.arange(len(data.rows_per_request), dtype=torch.int32, device=device),
        torch.tensor(data.rows_per_request, dtype=torch.int64, device=device),
        output_size=rows,
    )
    return _build_prefill_table(
        blocks=blocks,
        compress_lens=data.compress_lens,
        page_table=index_page_table,
        page_size=index_page_size,
        request_ids=request_ids,
        rows_per_request=data.rows_per_request,
        q_dtype=data.q_fp4.dtype,
        valid_lens=valid_lens,
        topk_blocks=topk_blocks,
    )


def select_prefill_table(
    *,
    table: _SparsePrefillTable,
    data: DeepGEMMPrefillData,
    k_cache: torch.Tensor,
    out_positions: torch.Tensor,
) -> None:
    """A consumer's top-``k`` over the published blocks, in the layout
    ``publish_prefill_table`` writes; ``k_cache`` is the layer's index-K pool,
    ``[pages, page_size, 1, 68]`` uint8."""
    rows, heads = data.q_sf.shape
    logits = sparse_logits(
        data.q_fp4.view(rows, 1, heads, 64),
        data.q_sf.view(rows, 1, heads),
        k_cache,
        data.weights.to(torch.bfloat16),
        table.schedule,
        table.blocks.shape[1],
    )
    # request-relative compressed positions, -1 padded
    topk_transform_sparse(logits, table.valid_lens, table.blocks, out_positions)
    out_positions.add_(torch.where(out_positions >= 0, data.request_starts[:, None], 0))
