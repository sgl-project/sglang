"""The candidate scheme carried by a DeepGEMM sparse table (SM100): a source
publishes its blocks with the schedule ``fp8_fp4_paged_sparse_mqa_logits`` needs,
a consumer scores only those blocks out of the index-K pool."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

import msgspec
import torch

from sglang.kernels.ops.attention.dsv4.candidate_blocks import (
    amax8_varlen,
    amax_topk_blocks,
    candidate_row_lens,
)
from sglang.kernels.ops.attention.dsv4.candidate_table import (
    CANDIDATE_BLOCK_SIZE,
    build_sparse_indexer_schedule,
    sort_candidate_blocks,
)
from sglang.kernels.ops.attention.dsv4.index_logits import (
    deep_gemm_fp4_paged_mqa_logits,
    sparse_logits,
)
from sglang.kernels.ops.attention.dsv4.topk import (
    topk_transform_paged_v2,
    topk_transform_ragged_v2,
    topk_transform_sparse,
)
from sglang.srt.layers.attention.dsv4.indexer import topk_transform_paged_from_metadata
from sglang.srt.layers.attention.dsv4.metadata import expand_index_page_table
from sglang.srt.utils.common import async_h2d

from .scoring import (
    DeepGEMMDecodeData,
    DeepGEMMPrefillData,
    get_deep_gemm_decode_data,
    get_deep_gemm_prefill_data,
    get_index_k_cache,
    score_tiles,
)
from .types import (
    CandidateMetadata,
    DecodeInputs,
    PrefillInputs,
    RowShard,
    get_tail_row_indices,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool


class _SparseTable(CandidateMetadata, msgspec.Struct):
    # [rows, topk_blocks] int32: ascending logical block ids, valid for the first
    # min(topk_blocks, ceil(seq_len / 8)) entries of a row; DeepGEMM reads only those
    blocks: torch.Tensor
    # DeepGEMM's schedule metadata (uint8) for them
    schedule: torch.Tensor
    # [rows, topk_blocks] int32: the same blocks as pool slots / 8; column j of the
    # sparse row maps to slot phys_blocks[b, j // 8] * 8 + j % 8
    phys_blocks: torch.Tensor
    # [rows] int32: length of each row of the sparse logits: the published blocks
    # laid out block by block, the newest possibly partial (`candidate_row_lens`)
    valid_lens: torch.Tensor
    # recorded on the side stream once the fields above are complete
    ready: torch.cuda.Event


class _SparsePrefillTable(_SparseTable):
    """The inputs stay attached so the late-layer tail can rebuild the table for
    its rows: the DeepGEMM schedule is per row set, it cannot be sliced."""

    compress_lens: torch.Tensor  # [rows] int32
    page_table: torch.Tensor  # [rows, index pages] int32
    request_ids: torch.Tensor  # [rows] int32 row-pair ids, see _row_pair_ids
    q_dtype: torch.dtype
    page_size: int  # index-K pool page size (slots)
    rows_per_request: List[int]  # rows of each request, in row order

    def tail(self, rows_per_request: List[int]) -> _SparsePrefillTable:
        rows = get_tail_row_indices(
            full_rows_per_request=self.rows_per_request,
            tail_rows_per_request=rows_per_request,
            device=self.blocks.device,
        )
        compress_lens = self.compress_lens[rows].contiguous()
        _, valid_lens = candidate_row_lens(compress_lens, self.blocks.shape[1])
        return _build_prefill_table(
            blocks=self.blocks[rows].contiguous(),
            compress_lens=compress_lens,
            page_table=self.page_table[rows].contiguous(),
            page_size=self.page_size,
            request_ids=self.request_ids[rows].contiguous(),
            rows_per_request=rows_per_request,
            q_dtype=self.q_dtype,
            valid_lens=valid_lens,
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

    def publish_prefill(self, inputs: PrefillInputs):
        inputs.reset_outputs()
        data = get_deep_gemm_prefill_data(inputs, self.req_to_token)
        if data is None:
            return None
        index_page_size = self.token_to_kv_pool.get_index_k_page_size(
            inputs.compress_ratio
        )
        table = publish_prefill_table(
            data=data,
            kv=self.token_to_kv_pool.get_low_ratio_index_k_fp4(
                inputs.layer_id, data.k_slots
            ),
            index_page_table=expand_index_page_table(
                inputs.kv_page_table[: data.num_rows],
                full_page_size=self.page_size,
                compress_ratio=inputs.compress_ratio,
                index_page_size=index_page_size,
            ).contiguous(),
            index_page_size=index_page_size,
            topk_blocks=self.topk_blocks,
            out_positions=inputs.out_raw_indices[: data.num_rows],
            row_shard=inputs.row_shard,
        )
        data.write_page_indices(inputs)
        return table

    def consume_prefill(
        self,
        inputs: PrefillInputs,
        published: Optional[_SparsePrefillTable],
    ) -> None:
        inputs.reset_outputs()
        data = get_deep_gemm_prefill_data(inputs, self.req_to_token)
        if data is None:
            return
        assert published is not None
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
            out_positions=inputs.out_raw_indices[: data.num_rows],
        )
        data.write_page_indices(inputs)

    def publish_decode(self, inputs: DecodeInputs):
        data = get_deep_gemm_decode_data(inputs, self.token_to_kv_pool)
        metadata = inputs.paged_metadata
        seq_lens = metadata.compressed_seq_lens.reshape(-1)
        if isinstance(metadata.deep_gemm_metadata, list):
            return self._publish_decode_chunked(inputs, data, seq_lens)
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
            inputs.out_page_indices,
            metadata.compressed_page_size,
            metadata.topk_metadata,
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
            # consume_decode reads these on the main stream
            for t in (blocks, schedule, phys_blocks, row_valid_lens):
                t.record_stream(main_stream)
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
    ) -> _SparseTable:
        """On the current stream, so each chunk's logits are released before the next."""
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
                inputs.out_page_indices,
                None,
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
        topk_transform_sparse(
            logits, published.valid_lens, published.phys_blocks, inputs.out_page_indices
        )

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
    )


def _row_pair_ids(rows_per_request: List[int], *, device: torch.device) -> torch.Tensor:
    """[rows] int32: one id per consecutive row pair of a request, never across two.
    DeepGEMM's schedule walks back over equal ids to a row's request start once per
    row, quadratic in a request's rows; per-pair ids bound the walk to one step.

    Pairs, not larger groups, because DeepGEMM's paged scheduler has BLOCK_Q == 2:
    the ids only have to delimit what it already groups, and its page table is
    indexed by Q row rather than by the id value, so the ids never address anything.
    A wider BLOCK_Q would need the pair width here to follow it."""
    counts = async_h2d(rows_per_request, dtype=torch.int64, device=device)
    rows = sum(rows_per_request)
    row = torch.arange(rows, device=device)
    request = torch.repeat_interleave(
        torch.arange(counts.numel(), device=device), counts, output_size=rows
    )
    first_row = torch.cumsum(counts, 0) - counts
    return (row - ((row - first_row[request]) & 1)).to(torch.int32)


def select_prefill_rows(
    *,
    data: DeepGEMMPrefillData,
    kv: Tuple[torch.Tensor, torch.Tensor],
    rows: slice,
    nblocks: torch.Tensor,
    out_positions: torch.Tensor,
    out_blocks: torch.Tensor,
) -> None:
    """The source's own top-k (request-local compressed positions) and its top
    blocks for ``rows``, from one pass over their dense scores; one row of each
    output per scored row, ``nblocks`` already sliced to ``rows``."""
    lens_all = data.compress_lens[rows]
    zero_offsets = torch.zeros_like(lens_all)
    # the block keys read the score rows through 32-byte vectors
    for tile, logits in score_tiles(data, kv, width_align=8, rows=rows):
        lens = lens_all[tile]
        topk_transform_ragged_v2(
            logits,
            lens,
            out_offsets=zero_offsets[tile],
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
            out_offsets=zero_offsets[: logits.shape[0]],
            out_indices=out_blocks[tile],
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
    row_shard: Optional[RowShard] = None,
) -> _SparsePrefillTable:
    """The source's own top-k into ``out_positions`` (request-local compressed
    positions) and the chunk's table, from one pass over its dense scores;
    ``index_page_table`` is at ``index_page_size`` slots. With ``row_shard`` the
    scores are this rank's share and the selections are all-gathered; the table
    is then built over every row, because its schedule cannot be sliced."""
    rows = data.num_rows
    device = data.q_fp4.device
    nblocks, valid_lens = candidate_row_lens(data.compress_lens, topk_blocks)
    if row_shard is None:
        blocks = torch.empty(rows, topk_blocks, dtype=torch.int32, device=device)
        select_prefill_rows(
            data=data,
            kv=kv,
            rows=slice(0, rows),
            nblocks=nblocks,
            out_positions=out_positions,
            out_blocks=blocks,
        )
    else:
        local = row_shard.local_rows(rows)
        per = row_shard.rows_per_rank(rows)
        local_positions = out_positions.new_full((per, out_positions.shape[1]), -1)
        local_blocks = torch.empty(per, topk_blocks, dtype=torch.int32, device=device)
        select_prefill_rows(
            data=data,
            kv=kv,
            rows=local,
            nblocks=nblocks[local],
            out_positions=local_positions,
            out_blocks=local_blocks,
        )
        out_positions.copy_(row_shard.gather(local_positions, rows))
        blocks = row_shard.gather(local_blocks, rows).contiguous()
    return _build_prefill_table(
        blocks=blocks,
        compress_lens=data.compress_lens,
        page_table=index_page_table,
        page_size=index_page_size,
        request_ids=_row_pair_ids(data.rows_per_request, device=device),
        rows_per_request=data.rows_per_request,
        q_dtype=data.q_fp4.dtype,
        valid_lens=valid_lens,
    )


def select_prefill_table(
    *,
    table: _SparsePrefillTable,
    data: DeepGEMMPrefillData,
    k_cache: torch.Tensor,
    out_positions: torch.Tensor,
) -> None:
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
