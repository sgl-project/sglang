"""Opt-in LiteTopK top-512 of the ratio-1/2 index layers' decode and target-verify
rows (``SGLANG_OPT_LITETOPK_DECODE``), with ``topk_transform_paged_v2``'s output
contract. DeepGEMM's BF16 MXFP4 logits count the histogram; exactness is relative
to them, so BF16 rounding can select differently from the default FP32 logits."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch

from sglang.kernels.ops.attention.litetopk_decode import BF16_TOP512, LiteTopKPlan
from sglang.srt.layers.attention.litetopk_decode import (
    LiteTopKDecode,
    get_litetopk_decode,
)

if TYPE_CHECKING:
    from sglang.srt.layers.attention.dsv4.metadata import PagedIndexerMetadata

    from .scoring import DeepGEMMDecodeData

_HEADS = 32
_HEAD_BYTES = 64  # packed e2m1, head dim 128
# DeepGEMM's BF16 producer: Q4 tiles serve 1..4 tokens per request, Q6 tiles 5..6.
_MAX_TOKENS_PER_REQUEST = 6
# Its schedule runs on one CTA with shared memory proportional to the rows.
_MAX_ROWS = 16384


def get_litetopk_dsv41() -> Optional[Dsv41LiteTopK]:
    buffers = get_litetopk_decode(BF16_TOP512)
    return None if buffers is None else Dsv41LiteTopK(buffers)


class Dsv41LiteTopK:
    def __init__(self, buffers: LiteTopKDecode):
        self._buffers = buffers

    def prepare_metadata(
        self,
        metadata: PagedIndexerMetadata,
        request_indices: torch.Tensor,
        tokens_per_request: int,
    ) -> None:
        # Recorded in CUDA graphs before any index layer, so replays schedule the live
        # lengths and request IDs (a stale schedule can deadlock DeepGEMM).
        metadata.litetopk_schedule = None
        metadata.litetopk_request_ids = None
        metadata.litetopk_tokens_per_request = tokens_per_request
        if (
            not 1 <= tokens_per_request <= _MAX_TOKENS_PER_REQUEST
            or metadata.compressed_page_size != BF16_TOP512.page_size
            or metadata.row_chunk > 0
            or isinstance(metadata.deep_gemm_metadata, list)
            or request_indices.numel() != metadata.compressed_seq_lens.numel()
            or request_indices.numel() > _MAX_ROWS
        ):
            return
        import deep_gemm

        request_ids = request_indices.to(torch.int32).contiguous()
        metadata.litetopk_request_ids = request_ids
        metadata.litetopk_schedule = deep_gemm.get_paged_mqa_logits_bf16_metadata(
            _context_lens(metadata),
            BF16_TOP512.page_size,
            deep_gemm.get_num_sms(),
            indices=request_ids,
            tokens_per_request=tokens_per_request,
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
            # The BF16 producer takes one query row per score row, 32 heads x 128.
            or tuple(data.q_fp4.shape) != (rows, 1, _HEADS, _HEAD_BYTES)
            or tuple(data.q_sf.shape) != (rows, 1, _HEADS)
            or tuple(data.weights.shape) != (rows, _HEADS)
        ):
            return None
        plan = self._buffers.plan(rows)
        if plan is None:
            return None
        import deep_gemm

        logits = deep_gemm.fp4_paged_mqa_logits_bf16(
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
            tokens_per_request=metadata.litetopk_tokens_per_request,
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


def _context_lens(metadata: PagedIndexerMetadata) -> torch.Tensor:
    # int32 [rows, 1]: DeepGEMM takes the lengths of one-token rows as a column.
    lens = metadata.compressed_seq_lens.to(torch.int32)
    return lens.unsqueeze(-1) if lens.dim() == 1 else lens
