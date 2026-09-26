from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

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
    flat_index_logits_tiles,
    sparse_logits,
)
from sglang.kernels.ops.attention.dsv4.topk import (
    topk_transform_bf16_small,
    topk_transform_paged_v2,
    topk_transform_ragged_v2,
    topk_transform_sparse,
)
from sglang.srt.layers.attention.dsv4.candidate_indexer import (
    CandidateIndexer,
    CandidateMetadata,
    IndexerInputs,
    PrefillIndexerInputs,
    expand_index_page_table,
)
from sglang.srt.layers.attention.dsv4.dense_prefill_indexer import (
    _SCORE_BUDGET_BYTES,
    DenseCandidateIndexer,
)
from sglang.srt.layers.attention.dsv4.indexer import (
    topk_transform_paged_from_metadata,
)


@dataclass
class SparseBlockTable(CandidateMetadata):
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


@dataclass
class PrefillSparseBlockTable(SparseBlockTable):
    """Published by a prefill chunk's candidate source: one row per query token.
    The inputs stay attached so the late-layer tail can rebuild the table for
    its rows (the DeepGEMM schedule is per row set, it cannot be sliced)."""

    compress_lens: torch.Tensor  # [rows] int32
    page_table: torch.Tensor  # [rows, index pages] int32
    request_ids: torch.Tensor  # [rows] int32
    q_dtype: torch.dtype
    page_size: int  # index-K pool page size (slots)
    rows_per_request: List[int]  # rows of each request, in row order


# TODO(dark): support fusion of publish + topk of publish layer
class DeepGemmCandidateIndexer(CandidateIndexer):
    def __init__(
        self,
        topk_blocks: int,
        block_size: int,
        prefill_dense: Optional[DenseCandidateIndexer] = None,
    ):
        assert block_size == CANDIDATE_BLOCK_SIZE, block_size
        self.topk_blocks = topk_blocks
        self.block_size = block_size
        self.alt_stream = torch.cuda.Stream()
        self._row_ids: Optional[torch.Tensor] = None
        self._retired_row_ids: list = []  # captured graphs keep reading the buffers they saw
        # set for the forwards whose prefill rows the page table does not describe
        self._prefill_dense = prefill_dense

    def _request_ids(
        self, request_ids: Optional[torch.Tensor], rows: int, device: torch.device
    ) -> torch.Tensor:
        """int32 ``[rows]``; None means every row is its own request, served from a
        cached ``arange`` (grown in steps of 8192) so that case costs no launch."""
        if request_ids is not None:
            # the scheduler keeps request indices as int64; one small cast per publish
            return request_ids[:rows].to(torch.int32).contiguous()
        buf = self._row_ids
        if buf is None or buf.numel() < rows:
            assert not torch.cuda.is_current_stream_capturing(), (
                f"row-id buffer grows to {rows} rows inside a CUDA graph capture; "
                "warm up with the largest row count first"
            )
            if buf is not None:
                self._retired_row_ids.append(buf)
            size = max(8192, -(-rows // 8192) * 8192)
            buf = self._row_ids = torch.arange(size, dtype=torch.int32, device=device)
        return buf[:rows]

    def publish_decode(
        self,
        inputs: IndexerInputs,
        page_indices: torch.Tensor,
        raw_indices: Optional[torch.Tensor] = None,
    ) -> SparseBlockTable:
        """The publishing layer's own plain top-k into ``page_indices``, plus the
        block table for the consumers; the backend stores it on the forward
        metadata."""
        metadata = inputs.metadata
        seq_lens = metadata.compressed_seq_lens.reshape(-1)
        if isinstance(metadata.deep_gemm_metadata, list):
            return self._publish_decode_chunked(
                inputs, page_indices, raw_indices, seq_lens
            )
        logits = deep_gemm_fp4_paged_mqa_logits(
            (inputs.q_fp4, inputs.q_sf),
            inputs.k_cache,
            inputs.weights,
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
            page_indices,
            metadata.compressed_page_size,
            metadata.topk_metadata,
            out_raw_indices=raw_indices,
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
                inputs.q_fp4.dtype,
                self._request_ids(inputs.request_ids, inputs.num_rows, blocks.device),
            )
            # select_decode reads these on the main stream
            for t in (blocks, schedule, phys_blocks, row_valid_lens):
                t.record_stream(main_stream)
            ready = torch.cuda.Event()
            ready.record(self.alt_stream)
            return SparseBlockTable(
                blocks=blocks,
                schedule=schedule,
                phys_blocks=phys_blocks,
                valid_lens=row_valid_lens,
                ready=ready,
            )

    def _publish_decode_chunked(
        self,
        inputs: IndexerInputs,
        page_indices: torch.Tensor,
        raw_indices: Optional[torch.Tensor],
        seq_lens: torch.Tensor,
    ) -> SparseBlockTable:
        """Publish an eager forward whose dense logits are bounded by row chunks.

        CUDA-graph metadata always carries one tensor schedule and keeps using the
        asynchronous fast path above.  The exceptional eager path stays on the
        current stream so each chunk's full logits can be released before the next.
        """
        metadata = inputs.metadata
        block_chunks = []
        phys_block_chunks = []
        valid_len_chunks = []
        topk_plans = metadata.topk_metadata_chunks
        assert not metadata.use_topk_v2 or topk_plans is not None

        for chunk_idx, (rows, plan) in enumerate(metadata.row_chunks()):
            logits = deep_gemm_fp4_paged_mqa_logits(
                (inputs.q_fp4[rows], inputs.q_sf[rows]),
                inputs.k_cache,
                inputs.weights[rows],
                metadata.compressed_seq_lens[rows],
                metadata.page_table[rows],
                plan,
                metadata.max_compressed_seq_len,
            )
            topk_transform_paged_from_metadata(
                logits,
                metadata,
                page_indices,
                raw_indices,
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
        phys_blocks = torch.cat(phys_block_chunks)
        row_valid_lens = torch.cat(valid_len_chunks)
        schedule = build_sparse_indexer_schedule(
            blocks,
            seq_lens,
            metadata.page_table,
            metadata.compressed_page_size,
            inputs.q_fp4.dtype,
            self._request_ids(inputs.request_ids, inputs.num_rows, blocks.device),
        )
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream())
        return SparseBlockTable(
            blocks=blocks,
            schedule=schedule,
            phys_blocks=phys_blocks,
            valid_lens=row_valid_lens,
            ready=ready,
        )

    def _scores(self, table: SparseBlockTable, inputs: IndexerInputs) -> torch.Tensor:
        return sparse_logits(
            inputs.q_fp4,
            inputs.q_sf,
            inputs.k_cache,
            inputs.weights.to(torch.bfloat16),
            table.schedule,
            table.blocks.shape[1],
        )

    def select_decode(
        self,
        candidate_metadata: SparseBlockTable,
        inputs: IndexerInputs,
        page_indices: torch.Tensor,
        raw_indices: Optional[torch.Tensor] = None,
    ) -> None:
        assert raw_indices is None
        table = candidate_metadata
        torch.cuda.current_stream().wait_event(table.ready)
        logits = self._scores(table, inputs)
        # decode carries no raw_indices; the kernel writes slots only
        topk_transform_sparse(logits, table.valid_lens, table.phys_blocks, page_indices)

    def publish_prefill(
        self, inputs: PrefillIndexerInputs, out_positions: torch.Tensor
    ) -> CandidateMetadata:
        """The source layer's own top-k and the block table of the chunk from
        one pass over its dense scores, tile by tile: per tile the plain top-k
        into ``out_positions`` and one read for the block keys."""
        if self._prefill_dense is not None:
            return self._prefill_dense.publish_prefill(inputs, out_positions)
        rows = inputs.num_rows
        device = inputs.q_fp4.device
        nblocks, valid_lens = candidate_row_lens(inputs.compress_lens, self.topk_blocks)
        blocks = torch.empty(rows, self.topk_blocks, dtype=torch.int32, device=device)
        # the block keys read the score rows through 32-byte vectors
        for tile, logits in flat_index_logits_tiles(
            q=(inputs.q_fp4, inputs.q_sf),
            kv=inputs.kv,
            weights=inputs.weights,
            starts=inputs.request_starts,
            lengths=inputs.compress_lens,
            context_lengths=inputs.lens_per_request,
            budget_bytes=_SCORE_BUDGET_BYTES,
            width_align=8,
        ):
            lens = inputs.compress_lens[tile]
            topk_transform_ragged_v2(
                logits,
                lens,
                out_offsets=inputs.request_starts[tile],
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
                out_offsets=torch.zeros(
                    logits.shape[0], dtype=torch.int32, device=device
                ),
                out_indices=blocks[tile],
            )
        request_ids = torch.repeat_interleave(
            torch.arange(
                len(inputs.rows_per_request), dtype=torch.int32, device=device
            ),
            torch.tensor(inputs.rows_per_request, dtype=torch.int64, device=device),
            output_size=rows,
        )
        # index-K pool pages of each row's request
        page_table = expand_index_page_table(
            inputs.kv_page_table,
            full_page_size=inputs.kv_page_size,
            compress_ratio=inputs.compress_ratio,
            index_page_size=inputs.page_size,
        ).contiguous()
        return self._prefill_table(
            blocks,
            inputs.compress_lens,
            page_table,
            inputs.page_size,
            request_ids,
            inputs.rows_per_request,
            inputs.q_fp4.dtype,
            valid_lens,
        )

    def _prefill_table(
        self,
        blocks,
        compress_lens,
        page_table,
        page_size,
        request_ids,
        rows_per_request,
        q_dtype,
        valid_lens,
    ) -> PrefillSparseBlockTable:
        # in place: ascending, INT32_MAX padded, plus the blocks as pool slots / 8
        phys_blocks = sort_candidate_blocks(
            blocks, compress_lens, page_table, page_size
        )
        schedule = build_sparse_indexer_schedule(
            blocks, compress_lens, page_table, page_size, q_dtype, request_ids
        )
        ready = torch.cuda.Event()
        ready.record()
        return PrefillSparseBlockTable(
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

    def prefill_tail(
        self, published: CandidateMetadata, tail_lens: List[int]
    ) -> CandidateMetadata:
        if self._prefill_dense is not None:
            return self._prefill_dense.prefill_tail(published, tail_lens)
        table = published
        assert isinstance(table, PrefillSparseBlockTable), "prefill block table missing"
        rows, start = [], 0
        for n, t in zip(table.rows_per_request, tail_lens):
            rows.append(torch.arange(start + n - t, start + n))
            start += n
        rows = torch.cat(rows).to(table.blocks.device)
        compress_lens = table.compress_lens[rows].contiguous()
        _, valid_lens = candidate_row_lens(compress_lens, self.topk_blocks)
        return self._prefill_table(
            table.blocks[rows].contiguous(),
            compress_lens,
            table.page_table[rows].contiguous(),
            table.page_size,
            table.request_ids[rows].contiguous(),
            tail_lens,
            table.q_dtype,
            valid_lens,
        )

    def select_prefill(
        self,
        published: CandidateMetadata,
        inputs: PrefillIndexerInputs,
        out_positions: torch.Tensor,
    ) -> None:
        if self._prefill_dense is not None:
            return self._prefill_dense.select_prefill(published, inputs, out_positions)
        table = published
        assert isinstance(table, PrefillSparseBlockTable), "prefill block table missing"
        rows, heads = inputs.q_sf.shape
        logits = sparse_logits(
            inputs.q_fp4.view(rows, 1, heads, 64),
            inputs.q_sf.view(rows, 1, heads),
            inputs.k_cache,
            inputs.weights.to(torch.bfloat16),
            table.schedule,
            table.blocks.shape[1],
        )
        # request-relative compressed positions, -1 padded
        topk_transform_bf16_small(
            logits, table.valid_lens, table.blocks, out_positions, CANDIDATE_BLOCK_SIZE
        )
        out_positions.add_(
            torch.where(out_positions >= 0, inputs.request_starts[:, None], 0)
        )
