from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Optional

import torch

from sglang.kernels.ops.attention.dsv4.candidate_blocks import candidate_row_lens
from sglang.kernels.ops.attention.dsv4.topk import (
    amax8_varlen,
    plan_topk_v2,
    sort_candidate_blocks,
    topk_transform_bf16_small,
    topk_transform_paged_v2,
)
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsv4.candidate_indexer import (
    CandidateMetadata,
    IndexerInputs,
)
from sglang.srt.layers.attention.dsv4.indexer import (
    fp4_paged_mqa_logits,
)

CANDIDATE_BLOCK_SIZE = 8  # positions per block; DeepGEMM accepts 8 or 16


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
    # [rows] int32: length of each row of the sparse logits (see `valid_lens`)
    valid_lens: torch.Tensor


def valid_lens(seq_lens: torch.Tensor, topk_blocks: int) -> torch.Tensor:
    """Length of each row of the sparse logits: the published blocks laid out
    block by block, the newest (highest) block possibly partial. Torch reference
    of ``candidate_row_lens``; the decode path takes the kernel's value."""
    block = CANDIDATE_BLOCK_SIZE
    num = ((seq_lens + block - 1) // block).clamp_max(topk_blocks)
    return block * (num - 1) + (seq_lens - 1) % block + 1


def amax_topk_blocks(
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    nblocks: torch.Tensor,
    topk_blocks: int,
    max_seq_len: Optional[int] = None,
) -> torch.Tensor:
    """Per row the ``topk_blocks`` blocks of 8 positions with the largest block
    maximum among its first ``seq_lens[b]`` positions, the newest block always
    included: block ids in no particular order, ``-1`` past the row's count
    (``sort_candidate_blocks`` turns them into the published table). ``nblocks``
    is ``ceil(seq_lens / 8)`` as int32, from ``candidate_row_lens``."""
    rows = logits.shape[0]
    block = CANDIDATE_BLOCK_SIZE
    if max_seq_len is None:
        max_seq_len = logits.shape[1]
    # NOTE: plan cannot be the previous kernel of topk_transform_paged_v2
    plan = plan_topk_v2(nblocks)
    # block maxima, the newest block +inf; the top-k reads each row up to nblocks
    # only, so nothing past a row's keys is initialised (v2 needs stride % 4 == 0)
    keys = logits.new_empty(rows, -(-max_seq_len // (4 * block)) * 4)
    amax8_varlen(logits, seq_lens, out=keys)
    blocks = torch.empty(rows, topk_blocks, dtype=torch.int32, device=logits.device)
    topk_transform_paged_v2(keys, nblocks, None, blocks, 1, plan)
    return blocks


_ROW_IDS: dict = {}


def _row_ids(rows: int, device: torch.device) -> torch.Tensor:
    """``arange(rows)`` int32 from a cached buffer (grown in steps of 8192), so the
    every-row-its-own-request case costs no launch."""
    buf = _ROW_IDS.get(device)
    if buf is None or buf.numel() < rows:
        size = max(8192, -(-rows // 8192) * 8192)
        buf = _ROW_IDS[device] = torch.arange(size, dtype=torch.int32, device=device)
    return buf[:rows]


def build_sparse_indexer_schedule(
    blocks: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    page_size: int,
    q_dtype: torch.dtype,
    request_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """DeepGEMM's schedule for the published blocks: ``seq_lens`` ``[rows]``
    int32, ``page_table`` ``[rows, pages]`` int32 at the index pool's page size.
    ``request_ids`` ``[rows]``, one id per row with a request's rows consecutive
    (verify: its draft tokens), lets DeepGEMM pair two rows of a request on one
    KV pass; each row keeps its own block list and output layout. Paired rows
    must share their page-table row. None: every row is its own request."""
    import deep_gemm

    rows = blocks.shape[0]
    if request_ids is None:
        request_ids = _row_ids(rows, blocks.device)  # cached, no launch
    else:
        # the scheduler keeps request indices as int64; one small cast per publish
        request_ids = request_ids[:rows].to(torch.int32).contiguous()
    return deep_gemm.get_paged_sparse_mqa_logits_metadata(
        seq_lens.contiguous(),
        page_table,
        request_ids,
        page_size,
        blocks,
        q_dtype,
        CANDIDATE_BLOCK_SIZE,
    )


def sparse_logits(
    q_fp4: torch.Tensor,
    q_sf: torch.Tensor,
    k_cache: torch.Tensor,
    weights: torch.Tensor,
    table: SparseBlockTable,
) -> torch.Tensor:
    """bf16 logits ``[rows, topk_blocks * 8]`` of the published blocks: ``q_fp4``
    ``[rows, 1, heads, 64]`` int8 with ``q_sf`` ``[rows, 1, heads]`` int32 (packed
    ue8m0), ``k_cache`` ``[pages, page_size, 1, 68]`` uint8 whose page stride is
    a multiple of 512 bytes, ``weights`` ``[rows, heads]`` bf16."""
    import deep_gemm

    return deep_gemm.fp8_fp4_paged_sparse_mqa_logits(
        (q_fp4, q_sf),
        k_cache,
        weights,
        table.schedule,
        table.blocks.shape[1],
        CANDIDATE_BLOCK_SIZE,
    )


def topk_transform_sparse(
    logits: torch.Tensor,
    valid_lens: torch.Tensor,
    table: SparseBlockTable,
    page_indices: torch.Tensor,
) -> None:
    """Top-``k`` (``k = page_indices.shape[1]``) of every row of the sparse
    ``logits`` (bf16 ``[rows, topk_blocks * 8]``) within its first ``valid_lens[b]``
    columns, written as pool slots, ``-1`` where a row has fewer than ``k`` valid
    columns, in no particular order. Column ``j`` is slot
    ``phys_blocks[b, j // 8] * 8 + j % 8``: the bf16 top-k kernel's page-table
    transform at page size 8 over the physical block table."""
    topk_transform_bf16_small(
        logits, valid_lens, table.phys_blocks, page_indices, CANDIDATE_BLOCK_SIZE
    )


def warmup(rows: int, topk_blocks: int, page_size: int, q_dtype: torch.dtype, device):
    """Build one schedule so DeepGEMM allocates its per-stream metadata workspace
    outside any CUDA-graph capture."""
    seq_lens = torch.full(
        (rows,), CANDIDATE_BLOCK_SIZE, dtype=torch.int32, device=device
    )
    blocks = torch.zeros(rows, topk_blocks, dtype=torch.int32, device=device)
    page_table = torch.zeros(rows, 1, dtype=torch.int32, device=device)
    build_sparse_indexer_schedule(blocks, seq_lens, page_table, page_size, q_dtype)


class DeepGemmCandidateIndexer:
    def __init__(self, topk_blocks: int, block_size: int):
        assert block_size == CANDIDATE_BLOCK_SIZE, block_size
        self.topk_blocks = topk_blocks
        self.block_size = block_size
        self.alt_stream = None
        self.need_wait = False
        # TODO: maybe lower this priority
        if envs.SGLANG_DSV41_DEEP_GEMM_CANDIDATE_OVERLAP.get():
            self.alt_stream = torch.cuda.Stream()

    def publish_decode(
        self,
        inputs: IndexerInputs,
        page_indices: torch.Tensor,
        raw_indices: Optional[torch.Tensor] = None,
    ) -> SparseBlockTable:
        """Layer 20 end to end: dense logits, its own plain top-k into
        ``page_indices`` (and ``raw_indices`` when given), the block table from
        the same logits and DeepGEMM's schedule for it; the backend stores the
        table on the forward metadata."""
        metadata = inputs.metadata
        seq_lens = metadata.c4_seq_lens.reshape(-1)
        logits = fp4_paged_mqa_logits(
            (inputs.q_fp4, inputs.q_sf),
            inputs.k_cache,
            inputs.weights,
            metadata.c4_seq_lens,
            metadata.page_table,
            metadata.deep_gemm_metadata,
            metadata.max_c4_seq_len,
        )
        if self.alt_stream is not None:
            self.alt_stream.wait_stream(torch.cuda.current_stream())
            # the level-one chain reads `logits` on the side stream while the main
            # stream moves on: keep the allocator from handing its memory to a
            # later main-stream allocation before those reads ran
            logits.record_stream(self.alt_stream)
            self.need_wait = True
            stream_ctx = torch.cuda.stream(self.alt_stream)
        else:
            stream_ctx = contextlib.nullcontext()
        # TODO(candidate): one kernel for both selections below (dense logits read once)
        topk_transform_paged_v2(
            logits,
            seq_lens,
            metadata.page_table,
            page_indices,
            metadata.c4_page_size,
            metadata.topk_metadata,
            out_raw_indices=raw_indices,
        )
        # per row: block count for the block top-k, sparse-row length for the consumers
        with stream_ctx:
            nblocks, row_valid_lens = candidate_row_lens(seq_lens, self.topk_blocks)
            blocks = amax_topk_blocks(logits, seq_lens, nblocks, self.topk_blocks)
            # in place: ascending, INT32_MAX padded, plus the blocks as pool slots / 8
            phys_blocks = sort_candidate_blocks(
                blocks,
                seq_lens,
                metadata.page_table,
                metadata.c4_page_size,
            )
            schedule = build_sparse_indexer_schedule(
                blocks,
                seq_lens,
                metadata.page_table,
                metadata.c4_page_size,
                inputs.q_fp4.dtype,
                inputs.request_ids,
            )
            return SparseBlockTable(
                blocks=blocks,
                schedule=schedule,
                phys_blocks=phys_blocks,
                valid_lens=row_valid_lens,
            )

    def scores(self, table: SparseBlockTable, inputs: IndexerInputs) -> torch.Tensor:
        """A consumer: bf16 logits ``[rows, topk_blocks * 8]`` of the published
        blocks only."""
        return sparse_logits(
            inputs.q_fp4,
            inputs.q_sf,
            inputs.k_cache,
            inputs.weights.to(torch.bfloat16),
            table,
        )

    def select_decode(
        self,
        candidate_metadata: SparseBlockTable,
        inputs: IndexerInputs,
        page_indices: torch.Tensor,
        raw_indices: Optional[torch.Tensor] = None,
    ) -> None:
        """A consumer: top-``k`` (``k = page_indices.shape[1]``) inside the
        published blocks, written ascending as slots through the page table with
        ``-1`` past the valid count (and as positions into ``raw_indices`` when
        given)."""
        assert raw_indices is None
        table = candidate_metadata
        if self.alt_stream is not None and self.need_wait:
            self.need_wait = False
            torch.cuda.current_stream().wait_stream(self.alt_stream)
        logits = self.scores(table, inputs)
        # decode carries no raw_indices; the kernel writes slots only
        topk_transform_sparse(logits, table.valid_lens, table, page_indices)
