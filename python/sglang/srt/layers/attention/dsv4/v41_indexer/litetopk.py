"""Opt-in LiteTopK top-512 of the ratio-1/2 index layers' decode and target-verify
rows (``SGLANG_OPT_LITETOPK_DECODE``), with the default decode top-k's output
contract (physical slots, unordered, -1 padded), in place of the default top-k
(``topk_transform_paged_v2``). DeepGEMM's SM100 paged MQA logits count the
histogram while they write the BF16 logits, which LiteTopK reads as they are (the
default widens them to FP32 first): both select from the same values, and differ
only in which scores tied at the threshold they keep."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch

from sglang.kernels.ops.attention.litetopk_decode import BF16_TOP512, LiteTopKPlan
from sglang.srt.environ import envs
from sglang.srt.layers.attention.litetopk_decode import (
    LiteTopKDecode,
    get_litetopk_decode,
)

if TYPE_CHECKING:
    from sglang.srt.layers.attention.dsv4.metadata import PagedIndexerMetadata

    from .scoring import DeepGEMMDecodeData

_HEADS = 32
_HEAD_BYTES = 64  # packed e2m1, head dim 128


def get_litetopk_dsv41() -> Optional[Dsv41LiteTopK]:
    buffers = get_litetopk_decode(BF16_TOP512)
    return None if buffers is None else Dsv41LiteTopK(buffers)


class Dsv41LiteTopK:
    def __init__(self, buffers: LiteTopKDecode):
        self._buffers = buffers

    def prepare_metadata(
        self, metadata: PagedIndexerMetadata, tokens_per_request: int
    ) -> None:
        # Recorded in CUDA graphs before any index layer, so replays schedule the live
        # lengths (a stale schedule can deadlock DeepGEMM).
        metadata.litetopk_schedule = None
        metadata.litetopk_request_ids = None
        rows = metadata.compressed_seq_lens.numel()
        if (
            rows % tokens_per_request != 0
            or metadata.compressed_page_size != BF16_TOP512.page_size
            or metadata.row_chunk > 0
            or isinstance(metadata.deep_gemm_metadata, list)
        ):
            return
        # Rows are request-major: equal consecutive IDs make DeepGEMM read a
        # request's KV once for all of its verify tokens.
        request_ids = torch.arange(
            rows, dtype=torch.int32, device=metadata.compressed_seq_lens.device
        )
        if tokens_per_request > 1:
            request_ids = request_ids // tokens_per_request
        metadata.litetopk_request_ids = request_ids
        if tokens_per_request == 1 and _row_schedule(metadata) is not None:
            metadata.litetopk_schedule = metadata.deep_gemm_metadata
            return
        import deep_gemm

        metadata.litetopk_schedule = deep_gemm.get_paged_mqa_logits_metadata(
            _context_lens(metadata),
            BF16_TOP512.page_size,
            deep_gemm.get_num_sms(),
            indices=request_ids,
        )

    def scores(
        self,
        *,
        data: DeepGEMMDecodeData,
        metadata: PagedIndexerMetadata,
        out: torch.Tensor,
    ) -> Optional[Tuple[torch.Tensor, LiteTopKPlan]]:
        """The layer's BF16 logits with their histogram counted into the returned
        plan; None keeps the default top-k for this call."""
        rows = out.shape[0]
        if (
            rows == 0
            or metadata.litetopk_schedule is None
            or tuple(out.shape) != (rows, BF16_TOP512.topk)
            or not out.is_contiguous()
            # One query row per score row, 32 heads x 128.
            or tuple(data.q_fp4.shape) != (rows, 1, _HEADS, _HEAD_BYTES)
            or tuple(data.q_sf.shape) != (rows, 1, _HEADS)
            or tuple(data.weights.shape) != (rows, _HEADS)
        ):
            return None
        plan = self._buffers.plan(rows)
        if plan is None:
            return None
        import deep_gemm

        logits = deep_gemm.fp8_fp4_paged_mqa_logits(
            (data.q_fp4, data.q_sf),
            data.k_cache,
            # The fused Q packer rounds these weights to BF16 before widening
            # them to FP32, so this conversion is exact.
            data.weights.to(torch.bfloat16),
            _context_lens(metadata),
            metadata.page_table,
            metadata.litetopk_schedule,
            metadata.max_compressed_seq_len,
            indices=metadata.litetopk_request_ids,
            histogram=plan.histogram,
        )
        return logits, plan

    @staticmethod
    def select(
        *,
        logits: torch.Tensor,
        plan: LiteTopKPlan,
        metadata: PagedIndexerMetadata,
        out: torch.Tensor,
    ) -> None:
        plan.select(
            logits,
            metadata.compressed_seq_lens.reshape(-1).to(torch.int32),
            metadata.page_table,
            out=out,
        )


def _row_schedule(metadata: PagedIndexerMetadata) -> Optional[torch.Tensor]:
    # The forward's own schedule when it is DeepGEMM's, one request per row (the
    # TileLang indexer has the JIT builder make a schedule of another format).
    schedule = metadata.deep_gemm_metadata
    if isinstance(schedule, torch.Tensor) and (
        metadata.force_deep_gemm_metadata
        or not envs.SGLANG_OPT_USE_TILELANG_INDEXER.get()
    ):
        return schedule
    return None


def _context_lens(metadata: PagedIndexerMetadata) -> torch.Tensor:
    # int32 [rows, 1]: DeepGEMM takes the lengths of one-token rows as a column.
    lens = metadata.compressed_seq_lens.to(torch.int32)
    return lens.unsqueeze(-1) if lens.dim() == 1 else lens
