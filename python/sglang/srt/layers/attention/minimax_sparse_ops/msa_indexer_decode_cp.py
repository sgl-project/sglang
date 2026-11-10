"""Context-parallel MSA-v1 Q8KV4 decode indexer over the attention-TP ranks.

The index-K cache is replicated, so the stock decode reads all of it on every
rank. Here each rank scores all four index heads over 1/CP of the pages with the
stock scorer, scatters the fp32 page scores to the head owner's full-width rows,
and the owner runs the stock forced-tail top-k. Inputs to that top-k are
bit-identical to the stock path, so the selected pages are too.
"""

from __future__ import annotations

import logging

import torch

from sglang.kernels.ops.attention.msa_v1.indexer_decode_cp import (
    NUM_INDEX_HEADS,
    PAGE_GRANULE,
    compact_page_table,
    exchange_index_q,
    gather_scores,
    scatter_scores,
)
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_parallel

# Private MSA entry points: the score-only GEMM and the stock top-k.
try:
    from inference.msa_v1.indexer._common.topk_select import _topk_select
    from inference.msa_v1.indexer.decode import BatchDecodeIndexerPlan
    from inference.msa_v1.indexer.decode.q8kv4.interface import (
        _BatchDecodeProxyScoreWrapper,
    )

    _MSA_IMPORT_ERROR: Exception | None = None
except Exception as error:
    _MSA_IMPORT_ERROR = error

logger = logging.getLogger(__name__)

_HEAD_DIM = 128
_TOPK = 16
_SUPPORTED_CP_SIZES = (2,)


def _compact_width(max_pages: int, *, cp_size: int) -> int:
    # Rank 0's page count among [0, max_pages): the most any rank owns.
    chunk = cp_size * PAGE_GRANULE
    return (max_pages // chunk) * PAGE_GRANULE + min(max_pages % chunk, PAGE_GRANULE)


def _symmetric_empty(numel: int, *, dtype: torch.dtype, device, group_name: str):
    import torch.distributed._symmetric_memory as symm_mem

    # Outside inference mode: an inference tensor rejects the in-graph writes.
    with torch.inference_mode(False):
        tensor = symm_mem.empty(numel, dtype=dtype, device=device)
    handle = symm_mem.rendezvous(tensor, group_name)
    ptrs = torch.tensor(handle.buffer_ptrs, dtype=torch.int64, device=device)
    return tensor, ptrs


class MSAV1IndexerDecodeCP:
    """Process state shared by every sparse layer of one attention backend."""

    def __init__(
        self,
        *,
        group,
        max_rows: int,
        max_pages: int,
        num_local_heads: int,
        device: torch.device,
    ) -> None:
        self.cp_size = group.world_size
        self.cp_rank = group.rank_in_group
        self.num_local_heads = num_local_heads
        self.max_rows = max_rows
        self.max_pages = max_pages
        self.width = _compact_width(max_pages, cp_size=self.cp_size)
        name = group.device_group.group_name
        self.tile = torch.empty(
            max_rows * NUM_INDEX_HEADS * _HEAD_DIM, dtype=torch.uint8, device=device
        )
        self.stage_slot_bytes = self.tile.numel()
        # Peers write into these; keep the tensors referenced for the process.
        self.stage, self.stage_ptrs = _symmetric_empty(
            2 * self.stage_slot_bytes,
            dtype=torch.uint8,
            device=device,
            group_name=name,
        )
        self.stage.fill_(0xFF)
        self.recv, self.recv_ptrs = _symmetric_empty(
            (self.cp_size - 1) * num_local_heads * max_rows * self.width,
            dtype=torch.int32,
            device=device,
            group_name=name,
        )
        self.recv.fill_(-1)
        self.recv = self.recv.view(torch.float32)
        self.epoch = torch.zeros(1, dtype=torch.int32, device=device)
        self.compact_scores = torch.empty(
            NUM_INDEX_HEADS * max_rows * self.width, dtype=torch.float32, device=device
        )
        self.device = device
        torch.cuda.synchronize(device)
        group.barrier()

    def new_plan(
        self,
        *,
        batch_size: int,
        query_length: int,
        max_pages: int,
        num_valid_pages: torch.Tensor,
        full_scores: torch.Tensor,
    ) -> MSAV1IndexerDecodeCPPlan | None:
        """None for buckets past the graph-sized buffers (stock path serves them)."""
        assert max_pages == self.max_pages, (max_pages, self.max_pages)
        if batch_size * query_length > self.max_rows:
            return None
        return MSAV1IndexerDecodeCPPlan(
            cp=self,
            batch_size=batch_size,
            query_length=query_length,
            num_valid_pages=num_valid_pages,
            full_scores=full_scores,
        )


class MSAV1IndexerDecodeCPPlan:
    """Per graph bucket: the 4-head scorer bound to a compact page table."""

    def __init__(
        self,
        *,
        cp: MSAV1IndexerDecodeCP,
        batch_size: int,
        query_length: int,
        num_valid_pages: torch.Tensor,
        full_scores: torch.Tensor,
    ) -> None:
        self.cp = cp
        self.batch_size = batch_size
        self.query_length = query_length
        rows = batch_size * query_length
        width, device = cp.width, cp.device
        # The stock plan's per-row (own page + 1); every head shares it.
        self.num_valid_pages = num_valid_pages[0]
        # [rows * HL] in (row, head) order, the order of the full rows below.
        self.row_head_lengths = torch.empty(
            rows * cp.num_local_heads, dtype=torch.int32, device=device
        )
        self.full_scores = full_scores.view(-1)[
            : rows * cp.num_local_heads * cp.max_pages
        ].view(rows, cp.num_local_heads, cp.max_pages)
        self.compact_table = torch.zeros(
            batch_size, width, dtype=torch.int32, device=device
        )
        self.compact_seq_lens = torch.full(
            (batch_size,), query_length, dtype=torch.int32, device=device
        )
        self.scores = cp.compact_scores[: NUM_INDEX_HEADS * rows * width].view(
            NUM_INDEX_HEADS, rows, width
        )
        self.shared_plan = BatchDecodeIndexerPlan(
            self.compact_seq_lens,
            max_pages=width,
            num_index_heads=NUM_INDEX_HEADS,
            query_length=query_length,
        )
        self.scorer = _BatchDecodeProxyScoreWrapper(
            use_cuda_graph=True,
            page_table_buffer=self.compact_table,
            seq_lens_buffer=self.compact_seq_lens,
        )
        self.shared_plan.update()
        self.scorer.plan(
            self.compact_table,
            self.compact_seq_lens,
            num_index_heads=NUM_INDEX_HEADS,
            query_length=query_length,
            shared_plan=self.shared_plan,
        )

    def refresh(self, *, page_table: torch.Tensor, final_seq_lens: torch.Tensor):
        """Out of graph, after the stock plan updated its lengths and page table."""
        compact_page_table(
            page_table=page_table,
            final_seq_lens=final_seq_lens,
            compact_table=self.compact_table,
            compact_seq_lens=self.compact_seq_lens,
            query_length=self.query_length,
            cp_size=self.cp.cp_size,
            cp_rank=self.cp.cp_rank,
        )
        self.shared_plan.update()
        self.row_head_lengths.view(-1, self.cp.num_local_heads).copy_(
            self.num_valid_pages.unsqueeze(1)
        )

    def run(
        self,
        *,
        q: torch.Tensor,
        packed_k_cache: torch.Tensor,
        k_scale: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        """Write the local heads' top-k into out [rows, HL, 16] (contiguous)."""
        cp = self.cp
        rows = q.shape[0]
        exchange_index_q(
            q=q,
            stage_ptrs=cp.stage_ptrs,
            stage_slot_bytes=cp.stage_slot_bytes,
            tile=cp.tile,
            epoch=cp.epoch,
            cp_size=cp.cp_size,
            cp_rank=cp.cp_rank,
        )
        q_tile = (
            cp.tile[: rows * NUM_INDEX_HEADS * _HEAD_DIM]
            .view(torch.float8_e4m3fn)
            .view(self.batch_size, self.query_length, NUM_INDEX_HEADS, _HEAD_DIM)
        )
        self.scorer.run(
            q_tile, packed_k_cache, k_scale.view(torch.float8_e4m3fn), out=self.scores
        )
        scatter_scores(
            scores=self.scores,
            num_valid_pages=self.num_valid_pages,
            recv_ptrs=cp.recv_ptrs,
            full_scores=self.full_scores,
            epoch=cp.epoch,
            cp_size=cp.cp_size,
            cp_rank=cp.cp_rank,
        )
        gather_scores(
            recv=cp.recv,
            full_scores=self.full_scores,
            num_valid_pages=self.num_valid_pages,
            width=cp.width,
            cp_size=cp.cp_size,
            cp_rank=cp.cp_rank,
        )
        _topk_select(
            self.full_scores.view(-1, cp.max_pages),
            self.row_head_lengths,
            out=out.view(-1, _TOPK),
            compact_grid=True,
        )


def make_msa_v1_indexer_decode_cp(
    *,
    num_local_index_heads: int,
    max_rows: int,
    max_pages: int,
    qmul4_available: bool,
    device: torch.device,
) -> MSAV1IndexerDecodeCP | None:
    if not envs.SGLANG_OPT_MSA_V1_INDEXER_DECODE_CP.get():
        return None
    parallel = get_parallel()
    group = parallel.attn_tp_group
    why = None
    if group.world_size not in _SUPPORTED_CP_SIZES:
        why = f"attention TP size {group.world_size} (needs {_SUPPORTED_CP_SIZES})"
    elif num_local_index_heads * group.world_size != NUM_INDEX_HEADS:
        why = f"{num_local_index_heads} local index heads"
    elif not qmul4_available:
        why = "the Q8KV4 decode indexer is unavailable"
    elif _MSA_IMPORT_ERROR is not None:
        why = f"MSA score-only entry points are missing ({_MSA_IMPORT_ERROR})"
    elif max_rows <= 0:
        why = "no decode CUDA graph batch size to size the buffers"
    if why is not None:
        logger.warning("MSA-v1 indexer decode CP disabled: %s", why)
        return None
    cp = MSAV1IndexerDecodeCP(
        group=group,
        max_rows=max_rows,
        max_pages=max_pages,
        num_local_heads=num_local_index_heads,
        device=device,
    )
    logger.info(
        "MSA-v1 indexer decode CP: %d ranks score 1/%d of the index pages for all "
        "%d heads (rows<=%d, pages<=%d).",
        cp.cp_size,
        cp.cp_size,
        NUM_INDEX_HEADS,
        max_rows,
        max_pages,
    )
    return cp
