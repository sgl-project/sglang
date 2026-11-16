"""The top-k selections of the V4.1 index layers under ``--dsa-topk-backend``.

flashinfer runs every selection through FlashInfer's fused page-table transform (``top_k``
for torch-scored rows) under SGLANG_DSA_TOPK_FLASHINFER_DETERMINISTIC and
SGLANG_DSA_TOPK_FLASHINFER_TIE_BREAK. FlashInfer breaks ties by score column, so every
caller lays its columns out in logical order: positions, block ids, or ascending
candidate blocks. sgl-kernel keeps DeepSelect, top-k v2 and the bf16 sparse kernel.
torch has no V4.1 top-k of its own and runs them too, as on main, with
``use_topk_v2`` off, so its fp32 paged decode takes top-k v1."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch

from sglang.kernels.ops.attention.dsv4.candidate_blocks import (
    amax_block_keys,
    amax_topk_blocks,
)
from sglang.kernels.ops.attention.dsv4.candidate_table import CANDIDATE_BLOCK_SIZE
from sglang.kernels.ops.attention.dsv4.topk import (
    topk_transform_paged,
    topk_transform_paged_v2,
    topk_transform_ragged_v2,
    topk_transform_sparse,
)
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.dsa_topk_backend import (
    DSATopKBackend,
    _flashinfer_tie_break_value,
)
from sglang.srt.layers.attention.dsv4.indexer import (
    topk_transform_flashinfer_fused,
    topk_transform_paged_bf16,
)


class TopKSelector:
    """Selections are int32, ``-1`` padded and unordered, written into contiguous
    outputs with one row per score row."""

    def __init__(self, backend: DSATopKBackend) -> None:
        if backend.is_sgl_kernel() or backend.is_torch():
            # V4.1 has no torch top-k: torch runs sgl-kernel's selections, as on main,
            # where its PagedIndexerMetadata.use_topk_v2 stays off.
            self.backend = DSATopKBackend.SGL_KERNEL
        elif backend.is_flashinfer():
            self.backend = DSATopKBackend.FLASHINFER
        else:
            raise ValueError(f"Unsupported {backend = } for the V4.1 index-layer top-k")
        # FlashInfer's CUB top-k scratch is one buffer per device, so two of its
        # selections must not run on different streams at once.
        self.can_overlap_streams = self.backend.is_sgl_kernel()

    def dense(
        self, scores: torch.Tensor, k: int, *, sorted: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(values, int64 indices)`` over the last dim of torch-scored rows."""
        if self.backend.is_sgl_kernel():
            return scores.topk(k, dim=-1, sorted=sorted)
        import flashinfer

        # FlashInfer's graph-safe CUB path cannot sort, and it is the only graph-safe
        # path on GPUs without FilteredTopK's 128 KiB of shared memory (SM120).
        values, indices = flashinfer.top_k(
            scores.contiguous(), k, sorted=False, **_flashinfer_modes()
        )
        if sorted:
            values, order = values.sort(dim=-1, descending=True)
            indices = indices.gather(-1, order)
        return values, indices

    def ragged(
        self,
        scores: torch.Tensor,
        lens: torch.Tensor,
        *,
        out: torch.Tensor,
        offsets: torch.Tensor,
    ) -> None:
        """Each row's top-k among its first ``lens`` columns, plus ``offsets``."""
        if self.backend.is_sgl_kernel():
            _sgl_kernel_ragged(scores, lens, out=out, offsets=offsets)
            return
        self._positions(scores, lens, out)
        out.copy_(torch.where(out >= 0, out + offsets[:, None], -1))

    def raw(
        self,
        scores: torch.Tensor,
        lens: torch.Tensor,
        *,
        out: torch.Tensor,
        plan: Optional[torch.Tensor],
    ) -> None:
        """Each row's top-k positions among its first ``lens`` columns; ``plan``, the
        top-k v2 plan of ``lens``, is read by sgl-kernel only."""
        if self.backend.is_sgl_kernel():
            assert plan is not None, "sgl-kernel raw top-k needs the top-k v2 plan"
            topk_transform_paged_v2(scores, lens, None, out, 1, plan)
            return
        self._positions(scores, lens, out)

    def amax_blocks(
        self,
        logits: torch.Tensor,
        seq_lens: torch.Tensor,
        nblocks: torch.Tensor,
        topk_blocks: int,
    ) -> torch.Tensor:
        """``[rows, topk_blocks]`` ids of the blocks of 8 with the largest maxima among
        each row's first ``seq_lens`` positions, the newest block always kept."""
        if self.backend.is_sgl_kernel():
            return amax_topk_blocks(logits, seq_lens, nblocks, topk_blocks)
        keys = amax_block_keys(logits, seq_lens)
        blocks = torch.empty(
            (logits.shape[0], topk_blocks), dtype=torch.int32, device=logits.device
        )
        self.raw(keys, nblocks, out=blocks, plan=None)
        return blocks

    def paged_from_metadata(
        self,
        logits: torch.Tensor,
        metadata,
        out: torch.Tensor,
        *,
        rows: Optional[slice] = None,
        plan: Optional[torch.Tensor] = None,
    ) -> None:
        """Pool slots into ``out``; ``metadata`` is a ``PagedIndexerMetadata`` and
        ``plan`` a row chunk's top-k v2 plan."""
        seq_lens, page_table = metadata.compressed_seq_lens, metadata.page_table
        if rows is not None:
            seq_lens, page_table, out = seq_lens[rows], page_table[rows], out[rows]
        page_size = metadata.compressed_page_size
        if self.backend.is_sgl_kernel():
            if logits.dtype == torch.bfloat16:
                topk_transform_paged_bf16(logits, seq_lens, page_table, out, page_size)
            elif metadata.use_topk_v2:
                topk_transform_paged_v2(
                    logits,
                    seq_lens,
                    page_table,
                    out,
                    page_size,
                    metadata.topk_metadata if plan is None else plan,
                )
            else:
                topk_transform_paged(logits, seq_lens, page_table, out, page_size)
            return
        self._paged(logits, seq_lens, page_table, out, page_size)

    def sparse(
        self,
        logits: torch.Tensor,
        valid_lens: torch.Tensor,
        outputs: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        """Top-k of each row of the bf16 sparse ``logits`` within its first
        ``valid_lens`` columns, into every ``(blocks, out)`` pair: column ``j`` as
        ``blocks[b, j // 8] * 8 + j % 8``. Under FlashInfer one selection feeds them all."""
        if self.backend.is_sgl_kernel():
            for blocks, out in outputs:
                topk_transform_sparse(logits, valid_lens, blocks, out)
            return
        (blocks, out), *rest = outputs
        columns = torch.empty_like(out) if rest else None
        self._paged(
            logits,
            valid_lens,
            blocks,
            out,
            CANDIDATE_BLOCK_SIZE,
            out_raw_indices=columns,
        )
        for blocks, out in rest:
            out.copy_(_through_table(columns, blocks, CANDIDATE_BLOCK_SIZE))

    def order_candidates(self, blocks: torch.Tensor) -> torch.Tensor:
        """Published candidate block ids in the order a consumer lays out its columns:
        ascending for FlashInfer, whose index tie-break then follows positions."""
        if self.backend.is_sgl_kernel():
            return blocks
        padding = torch.iinfo(blocks.dtype).max
        ordered = torch.where(blocks < 0, padding, blocks).sort(dim=-1).values
        return ordered.masked_fill(ordered == padding, -1)

    def _positions(
        self, scores: torch.Tensor, lens: torch.Tensor, out: torch.Tensor
    ) -> None:
        # One zero page spanning the row makes the paged transform emit positions; unlike
        # FlashInfer's ragged transform it reads DeepGEMM's padded score rows in place.
        rows, width = scores.shape
        page_size = 1 << max(width - 1, 0).bit_length()
        zero_page = torch.zeros((1, 1), dtype=torch.int32, device=scores.device)
        self._paged(scores, lens, zero_page.expand(rows, 1), out, page_size)

    def _paged(
        self,
        scores: torch.Tensor,
        lens: torch.Tensor,
        page_table: torch.Tensor,
        out: torch.Tensor,
        page_size: int,
        *,
        out_raw_indices: Optional[torch.Tensor] = None,
    ) -> None:
        topk_transform_flashinfer_fused(
            scores,
            lens.reshape(-1).to(torch.int32),
            page_table,
            out,
            page_size,
            out_raw_indices=out_raw_indices,
        )


def _flashinfer_modes() -> Dict[str, object]:
    return {
        "deterministic": envs.SGLANG_DSA_TOPK_FLASHINFER_DETERMINISTIC.get(),
        "tie_break": _flashinfer_tie_break_value(),
        "dsa_graph_safe": True,
    }


def _sgl_kernel_ragged(
    scores: torch.Tensor,
    lens: torch.Tensor,
    *,
    out: torch.Tensor,
    offsets: torch.Tensor,
) -> None:
    """DeepSelect for bf16 scores, top-k v2 for fp32."""
    if scores.dtype == torch.bfloat16:
        from sglang.kernels.ops.attention.deep_select import topk

        topk(
            scores,
            out.shape[1],
            end=lens,
            output_idx=out,
            output_idx_offset=offsets,
            idx_oob_fill_value=-1,
        )
        return
    topk_transform_ragged_v2(scores, lens, out_offsets=offsets, out_indices=out)


def _through_table(
    columns: torch.Tensor, table: torch.Tensor, page_size: int
) -> torch.Tensor:
    """Column ``j`` as ``table[b, j // page_size] * page_size + j % page_size``; ``-1``
    stays ``-1``."""
    pages = table.gather(1, (columns.clamp_min(0) // page_size).long())
    slots = pages * page_size + columns % page_size
    return torch.where(columns >= 0, slots, -1).to(torch.int32)
