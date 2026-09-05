from __future__ import annotations

from enum import Enum, IntEnum, auto
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_exec, get_spec
from sglang.srt.utils import is_hip

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

_FLASHINFER_TIE_BREAK_VALUES = {
    "small": 1,
    "large": 2,
}


class TopkTransformMethod(IntEnum):
    # Transform topk indices to indices to the page table (page_size = 1)
    PAGED = auto()
    # Transform topk indices to indices to ragged kv (non-paged)
    RAGGED = auto()


class DSATopKBackend(Enum):
    SGL_KERNEL = "sgl-kernel"
    TORCH = "torch"
    FLASHINFER = "flashinfer"
    AITER = "aiter"

    @classmethod
    def resolve(cls, model_runner: ModelRunner) -> DSATopKBackend:
        """Resolve the DSA top-k backend for one model runner.

        ``--dsa-topk-backend`` selects the target backend, while
        ``--speculative-dsa-topk-backend`` independently selects the draft.
        """
        value = (
            get_spec().speculative_dsa_topk_backend
            if model_runner.is_draft_worker
            else get_exec().kernel.dsa_topk_backend
        )
        backend = cls(value)
        if backend.is_aiter() and not is_hip():
            raise ValueError("dsa_topk_backend='aiter' requires ROCm.")
        return backend

    def is_sgl_kernel(self) -> bool:
        return self == DSATopKBackend.SGL_KERNEL

    def is_torch(self) -> bool:
        return self == DSATopKBackend.TORCH

    def is_flashinfer(self) -> bool:
        return self == DSATopKBackend.FLASHINFER

    def is_aiter(self) -> bool:
        return self == DSATopKBackend.AITER

    def should_use_topk_v2(self) -> bool:
        return self.is_sgl_kernel() and envs.SGLANG_OPT_USE_TOPK_V2.get()

    def topk_func(
        self,
        score: torch.Tensor,
        lengths: torch.Tensor,
        topk: int,
        row_starts: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.is_sgl_kernel():
            from sgl_kernel import fast_topk_v2

            return fast_topk_v2(score, lengths, topk, row_starts=row_starts)
        if self.is_torch():
            return _topk_unfused(
                score,
                lengths,
                topk,
                row_starts=row_starts,
                topk_op=torch.topk,
                topk_op_kwargs={"dim": -1},
            )
        if self.is_flashinfer():
            import flashinfer

            return _topk_unfused(
                score,
                lengths,
                topk,
                row_starts=row_starts,
                topk_op=flashinfer.top_k,
                topk_op_kwargs={
                    "sorted": False,
                    "deterministic": envs.SGLANG_DSA_TOPK_FLASHINFER_DETERMINISTIC.get(),
                    "tie_break": _flashinfer_tie_break_value(),
                    "dsa_graph_safe": True,
                },
            )
        if self.is_aiter():
            return _aiter_topk(score, lengths, topk, row_starts=row_starts)
        raise RuntimeError(f"Unsupported {self = }.")

    def topk_transform(
        self,
        logits: torch.Tensor,
        lengths: torch.Tensor,
        topk: int,
        topk_transform_method: TopkTransformMethod,
        attn_metadata,
        cu_seqlens_q_topk: Optional[torch.Tensor] = None,
        topk_indices_offset: Optional[torch.Tensor] = None,
        row_starts: Optional[torch.Tensor] = None,
        batch_idx_list: Optional[List[int]] = None,
        force_unfused_topk: bool = False,
    ) -> torch.Tensor:
        if not envs.SGLANG_DSA_FUSE_TOPK.get() or force_unfused_topk:
            return self.topk_func(logits, lengths, topk, row_starts=row_starts)

        # Decode-shaped PAGED top-k for the SGL backend (plain decode AND spec
        # verify / draft-extend, whose expanded rows match the same shape) routes
        # to the DeepSeek-V4 top-k v2 JIT kernel. It fuses top-k selection and the
        # page-table transform in one launch and consumes the indexer's own
        # page_size>=1 table directly, so no page_size=1 table is materialized.
        # Shared by DeepSeek-V3.2 and GLM DSA.
        # This is a deterministic dispatch on the work shape, not a best-effort
        # attempt: the fused-decode CUDA graph drops the page_size=1 table for
        # exactly this case (see dsa_drop_wide_page_table), so once the shape
        # matches we commit to v2 and never silently fall back to the legacy
        # page_size=1 path from here.
        if (
            self.should_use_topk_v2()
            and topk_transform_method == TopkTransformMethod.PAGED
            and row_starts is None
            and batch_idx_list is None
            and 0 < topk <= 2048
            and lengths.shape[0]
            == logits.shape[0]
            == attn_metadata.real_page_table.shape[0]
        ):
            return _topk_transform_v2_paged(logits, lengths, topk, attn_metadata)

        # Extend-shaped RAGGED top-k for the SGL backend routes to the same v2
        # kernel through its ragged entry point: no page table (the columns are
        # already flattened-KV positions), no plan (prefill has enough rows that
        # the cluster path never applies), just a per-row window and an additive
        # output transform. `batch_idx_list` is not None only on the prefill-CP
        # path, whose `topk_indices_offset` is built from cu_seqlens_q rather
        # than the KV bases -- leave that one on the legacy kernel.
        if (
            self.should_use_topk_v2()
            and topk_transform_method == TopkTransformMethod.RAGGED
            and topk_indices_offset is not None
            and batch_idx_list is None
            and 0 < topk <= 2048
            and lengths.shape[0] == logits.shape[0] == topk_indices_offset.shape[0]
        ):
            return _topk_transform_v2_ragged(
                logits, lengths, topk, topk_indices_offset, row_starts
            )

        if self.is_aiter():
            return _aiter_topk_transform(
                logits=logits,
                lengths=lengths,
                topk=topk,
                topk_transform_method=topk_transform_method,
                attn_metadata=attn_metadata,
                cu_seqlens_q_topk=cu_seqlens_q_topk,
                topk_indices_offset=topk_indices_offset,
                row_starts=row_starts,
                batch_idx_list=batch_idx_list,
            )

        # The legacy transforms below read attn_metadata.page_table_1 (page_size=1),
        # which is always present here: the fold only drops it for the decode case
        # dispatched to v2 above.
        assert attn_metadata.page_table_1 is not None

        if self.is_sgl_kernel():
            from sgl_kernel import (
                fast_topk_transform_fused,
                fast_topk_transform_ragged_fused,
            )

            if topk_transform_method == TopkTransformMethod.PAGED:
                page_table_size_1 = (
                    attn_metadata.page_table_1[batch_idx_list]
                    if batch_idx_list is not None
                    else attn_metadata.page_table_1
                )
                return fast_topk_transform_fused(
                    score=logits,
                    lengths=lengths,
                    page_table_size_1=page_table_size_1,
                    cu_seqlens_q=cu_seqlens_q_topk,
                    topk=topk,
                    row_starts=row_starts,
                )
            if topk_transform_method == TopkTransformMethod.RAGGED:
                if topk_indices_offset is None:
                    raise RuntimeError(
                        "RAGGED topk_transform requires topk_indices_offset; "
                        "expected extend-without-speculative metadata."
                    )
                return fast_topk_transform_ragged_fused(
                    score=logits,
                    lengths=lengths,
                    topk_indices_offset=topk_indices_offset,
                    topk=topk,
                    row_starts=row_starts,
                )
            raise RuntimeError(f"Unsupported {topk_transform_method = }.")

        if self.is_flashinfer():
            import flashinfer

            if topk_transform_method == TopkTransformMethod.PAGED:
                row_to_batch, page_table_row_starts = _build_flashinfer_paged_args(
                    attn_metadata=attn_metadata,
                    row_starts=row_starts,
                    cu_seqlens_q_topk=cu_seqlens_q_topk,
                    batch_idx_list=batch_idx_list,
                    device=logits.device,
                    num_rows=logits.shape[0],
                )
                return flashinfer.top_k_page_table_transform(
                    logits.contiguous(),
                    attn_metadata.page_table_1.contiguous(),
                    lengths.contiguous(),
                    topk,
                    row_to_batch=row_to_batch,
                    deterministic=envs.SGLANG_DSA_TOPK_FLASHINFER_DETERMINISTIC.get(),
                    tie_break=_flashinfer_tie_break_value(),
                    dsa_graph_safe=True,
                    row_starts=row_starts,
                    page_table_row_starts=page_table_row_starts,
                )
            if topk_transform_method == TopkTransformMethod.RAGGED:
                if topk_indices_offset is None:
                    raise RuntimeError(
                        "RAGGED topk_transform requires topk_indices_offset; "
                        "expected extend-without-speculative metadata."
                    )
                return flashinfer.top_k_ragged_transform(
                    logits.contiguous(),
                    topk_indices_offset.contiguous(),
                    lengths.contiguous(),
                    topk,
                    deterministic=envs.SGLANG_DSA_TOPK_FLASHINFER_DETERMINISTIC.get(),
                    tie_break=_flashinfer_tie_break_value(),
                    dsa_graph_safe=True,
                    row_starts=row_starts,
                )
            raise RuntimeError(f"Unsupported {topk_transform_method = }.")

        raise RuntimeError(f"Unsupported {self = } for SGLANG_DSA_FUSE_TOPK.")


def _aiter_row_bounds(
    lengths: torch.Tensor,
    row_starts: Optional[torch.Tensor],
    device: torch.device,
) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    lengths = lengths.to(dtype=torch.int32, device=device).contiguous()
    if row_starts is None:
        starts = torch.zeros_like(lengths)
        return None, starts, lengths
    starts = row_starts.to(dtype=torch.int32, device=device).contiguous()
    return starts, starts, starts + lengths


def _aiter_topk(
    score: torch.Tensor,
    lengths: torch.Tensor,
    topk: int,
    row_starts: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if not is_hip():
        raise ValueError("The AITER DSA top-k backend requires ROCm.")
    if score.dtype != torch.float32 or score.dim() != 2:
        raise ValueError("AITER DSA top-k expects a 2D float32 score tensor.")
    if score.stride(1) != 1:
        score = score.contiguous()

    num_rows, width = score.shape
    if out is None:
        out = score.new_full((num_rows, topk), -1, dtype=torch.int32)
    if num_rows == 0 or topk == 0 or width == 0:
        out.fill_(-1)
        return out

    lengths = lengths.to(dtype=torch.int32, device=score.device).contiguous()
    if width <= topk:
        positions = torch.arange(topk, dtype=torch.int32, device=score.device)
        out.copy_(
            torch.where(
                positions.unsqueeze(0) < lengths.unsqueeze(1),
                positions.unsqueeze(0),
                -1,
            )
        )
        return out

    import aiter

    op_name = "top_k_per_row_decode" if row_starts is None else "top_k_per_row_prefill"
    topk_op = getattr(aiter, op_name, None)
    if not callable(topk_op):
        raise RuntimeError(f"The AITER DSA top-k backend requires {op_name}.")
    if row_starts is None:
        topk_op(
            score,
            1,
            lengths,
            out,
            num_rows,
            score.stride(0),
            score.stride(1),
            k=topk,
        )
        return out

    _, starts, ends = _aiter_row_bounds(lengths, row_starts, score.device)
    topk_op(
        score,
        starts,
        ends,
        out,
        None,
        num_rows,
        score.stride(0),
        score.stride(1),
        k=topk,
        stable=True,
    )
    valid = out >= 0
    out.sub_(starts.unsqueeze(1))
    out.masked_fill_(~valid, -1)
    return out


def _map_topk_to_pages(
    raw_indices: torch.Tensor,
    page_table: torch.Tensor,
    page_size: int,
    row_to_batch: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if page_size <= 0 or page_size & (page_size - 1):
        raise ValueError("page_size must be a positive power of two.")
    if row_to_batch is None and page_table.shape[0] != raw_indices.shape[0]:
        raise ValueError("AITER top-k rows must match page-table rows.")
    if page_table.shape[1] == 0:
        mapped = torch.full_like(raw_indices, -1)
        if out is not None:
            out.copy_(mapped)
            return out
        return mapped

    page_bits = page_size.bit_length() - 1
    page_mask = page_size - 1
    valid = raw_indices >= 0
    safe_indices = raw_indices.clamp_min(0)
    logical_pages = (safe_indices >> page_bits).to(torch.long)
    if row_to_batch is None:
        physical_pages = torch.gather(page_table, 1, logical_pages)
    else:
        physical_pages = page_table[
            row_to_batch.to(torch.long).unsqueeze(1), logical_pages
        ]
    mapped = (physical_pages << page_bits) | (safe_indices & page_mask)
    mapped.masked_fill_(~valid, -1)
    if out is not None:
        out.copy_(mapped)
        return out
    return mapped


def _build_aiter_row_to_batch(
    batch_idx_list: Optional[List[int]],
    cu_seqlens_q_topk: Optional[torch.Tensor],
    device: torch.device,
    num_rows: int,
    require_mapping: bool = False,
) -> Optional[torch.Tensor]:
    row_to_batch = (
        torch.as_tensor(batch_idx_list, dtype=torch.int32, device=device)
        if batch_idx_list is not None
        else None
    )
    if row_to_batch is not None and cu_seqlens_q_topk is not None:
        if row_to_batch.shape[0] != num_rows:
            q_lens = (cu_seqlens_q_topk[1:] - cu_seqlens_q_topk[:-1]).to(
                dtype=torch.int32, device=device
            )
            row_to_batch = torch.repeat_interleave(
                row_to_batch, q_lens, output_size=num_rows
            )
        return row_to_batch

    if row_to_batch is None and cu_seqlens_q_topk is not None:
        num_batches = cu_seqlens_q_topk.shape[0] - 1
        if require_mapping or num_rows != num_batches:
            q_lens = (cu_seqlens_q_topk[1:] - cu_seqlens_q_topk[:-1]).to(
                dtype=torch.int32, device=device
            )
            row_to_batch = torch.repeat_interleave(
                torch.arange(q_lens.shape[0], dtype=torch.int32, device=device),
                q_lens,
                output_size=num_rows,
            )
    return row_to_batch


def aiter_topk_transform_paged(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    page_table: torch.Tensor,
    page_size: int,
    topk: int,
    row_starts: Optional[torch.Tensor] = None,
    row_to_batch: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    out_raw_indices: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if out is None:
        out = logits.new_empty((logits.shape[0], topk), dtype=torch.int32)
    if logits.shape[0] == 0 or topk == 0 or logits.shape[1] == 0:
        out.fill_(-1)
        if out_raw_indices is not None:
            out_raw_indices.fill_(-1)
        return out

    import aiter

    fused_op = getattr(aiter, "dsa_topk_transform", None)
    if callable(fused_op) and topk == 2048 and out_raw_indices is None:
        starts_arg, _, ends = _aiter_row_bounds(lengths, row_starts, logits.device)
        fused_op(
            logits,
            starts_arg,
            ends,
            page_table,
            out,
            page_size,
            topk,
            ptRowMap=row_to_batch,
        )
        return out

    raw_indices = (
        out_raw_indices
        if out_raw_indices is not None
        else logits.new_empty((logits.shape[0], topk), dtype=torch.int32)
    )
    _aiter_topk(logits, lengths, topk, row_starts=row_starts, out=raw_indices)
    return _map_topk_to_pages(
        raw_indices,
        page_table,
        page_size,
        row_to_batch=row_to_batch,
        out=out,
    )


def _aiter_topk_transform(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    topk: int,
    topk_transform_method: TopkTransformMethod,
    attn_metadata,
    cu_seqlens_q_topk: Optional[torch.Tensor],
    topk_indices_offset: Optional[torch.Tensor],
    row_starts: Optional[torch.Tensor],
    batch_idx_list: Optional[List[int]],
) -> torch.Tensor:
    if topk_transform_method == TopkTransformMethod.PAGED:
        if attn_metadata.page_table_1 is None:
            raise RuntimeError("PAGED AITER top-k requires a page table.")
        row_to_batch = _build_aiter_row_to_batch(
            batch_idx_list,
            cu_seqlens_q_topk,
            logits.device,
            logits.shape[0],
            require_mapping=row_starts is not None,
        )
        return aiter_topk_transform_paged(
            logits,
            lengths,
            attn_metadata.page_table_1,
            1,
            topk,
            row_starts=row_starts,
            row_to_batch=row_to_batch,
        )

    if topk_transform_method == TopkTransformMethod.RAGGED:
        if topk_indices_offset is None:
            raise RuntimeError("RAGGED AITER top-k requires topk_indices_offset.")
        raw_indices = _aiter_topk(logits, lengths, topk, row_starts=row_starts)
        offsets = topk_indices_offset.to(dtype=torch.int32, device=logits.device)
        if offsets.dim() == 1:
            offsets = offsets.unsqueeze(1)
        return torch.where(raw_indices >= 0, raw_indices + offsets, -1)

    raise RuntimeError(f"Unsupported {topk_transform_method = }.")


def _topk_unfused(
    score: torch.Tensor,
    lengths: torch.Tensor,
    topk: int,
    row_starts: Optional[torch.Tensor] = None,
    topk_op: Callable[..., Tuple[torch.Tensor, torch.Tensor]] = torch.topk,
    topk_op_kwargs: Optional[Dict[str, object]] = None,
) -> torch.Tensor:
    batch_size, max_score_len = score.shape
    topk_indices = score.new_full((batch_size, topk), -1, dtype=torch.int32)
    if batch_size == 0 or topk == 0 or max_score_len == 0:
        return topk_indices

    if row_starts is None:
        row_starts = torch.zeros_like(lengths, dtype=torch.int32, device=score.device)
    else:
        row_starts = row_starts.to(dtype=torch.int32, device=score.device)
    lengths = lengths.to(dtype=torch.int32, device=score.device)

    col_indices = torch.arange(max_score_len, dtype=torch.int32, device=score.device)
    col_indices = col_indices.unsqueeze(0)
    row_starts_unsqueezed = row_starts.unsqueeze(1)
    row_ends_unsqueezed = (row_starts + lengths).unsqueeze(1)
    valid_mask = (col_indices >= row_starts_unsqueezed) & (
        col_indices < row_ends_unsqueezed
    )

    masked_logits = score.masked_fill(~valid_mask, float("-inf"))
    valid_topk = min(topk, max_score_len)
    topk_kwargs = topk_op_kwargs or {}
    topk_scores, topk_col_indices = topk_op(masked_logits, valid_topk, **topk_kwargs)
    topk_local_indices = topk_col_indices.to(torch.int32) - row_starts_unsqueezed
    topk_local_indices = topk_local_indices.masked_fill(
        topk_scores == float("-inf"), -1
    )
    topk_indices[:, :valid_topk] = topk_local_indices

    return topk_indices


def _topk_transform_v2_paged(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    topk: int,
    attn_metadata,
) -> torch.Tensor:
    """Fused top-k + page-table transform via the DeepSeek-V4 v2 JIT kernel.

    Returns the transformed page indices ``(num_rows, topk)`` int32 (physical
    page_size=1 KV slots, ``-1`` padded) -- identical in meaning to
    ``fast_topk_transform_fused`` / ``flashinfer.top_k_page_table_transform``.
    The kernel selects, per row, the top-k of ``logits[row, :lengths[row]]`` and
    maps each selected position ``p`` through the page table as
    ``real_page_table[row, p // page_size] * page_size + (p % page_size)``. Feeding
    it the indexer's compact ``real_page_table`` (page_size = pool page size,
    typically 64) yields the same physical slots as gathering the page_size=1
    table, without materializing that wide table.

    This is a committed contract, not a best-effort path: ``topk_transform`` routes
    here only for the decode-shaped PAGED case, and the fused-decode CUDA graph
    drops the page_size=1 table for exactly this case (see
    ``dsa_drop_wide_page_table``). The preconditions below are therefore
    invariants the caller must uphold -- they assert (raise) on violation rather
    than fall back to the slow legacy path (which may not even have a page_size=1
    table to fall back to) or silently paper over bad input (padding, recomputing
    the plan) at the cost of the performance this path exists to deliver.

    ``lengths`` entries must be NON-NEGATIVE: the kernel reads them as
    ``uint32_t``, so a negative row length (DP-padded / idle-companion rows)
    reinterprets as ~4e9 tokens and illegal-addresses. Metadata producers clamp
    padded rows to 0 (see ``fused_dsa_draft_extend_metadata`` /
    ``seqlens_expand_kernel``); 0 takes the trivial all-(-1) output path.
    """
    from sglang.kernels.ops.attention.dsv4.topk import topk_transform_paged_v2

    num_rows = logits.shape[0]

    # The indexer (DeepGEMM) emits fp32 scores with unit row stride and a 16B-aligned
    # row stride (a multiple of 4), which is exactly the kernel's ABI (it checks
    # score_stride % 4 == 0 with strides {S, 1}). This holds even though the scores
    # may be a padded view (stride(0) > width, so not `is_contiguous()`); assert the
    # real requirement rather than force a contiguous copy of the wide score buffer.
    assert (
        logits.dtype == torch.float32
        and logits.stride(1) == 1
        and logits.stride(0) % 4 == 0
    ), (
        f"v2 top-k expects fp32 scores with unit row stride and 16B-aligned score_stride, got {logits.dtype=} {logits.stride()=}"
    )
    assert 0 < topk <= 2048, f"v2 top-k supports 0 < topk <= 2048, got {topk=}"

    page_table = attn_metadata.real_page_table

    # The plan is preprocessed once per forward (DSAMetadata.topk_v2_plan,
    # refreshed in-place under CUDA graph) and reused across layers. A missing or
    # mismatched plan means the caller skipped that preprocessing -- fail loudly
    # rather than silently recompute it per layer.
    plan = attn_metadata.topk_v2_plan
    assert plan is not None and plan.shape[0] == num_rows + 1, (
        "topk_v2_plan must be preprocessed per forward (see DSAMetadata.topk_v2_plan)"
    )

    page_size = attn_metadata.page_size
    out = logits.new_empty((num_rows, topk), dtype=torch.int32)
    topk_transform_paged_v2(logits, lengths, page_table, out, page_size, plan)
    return out


def _topk_transform_v2_ragged(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    topk: int,
    topk_indices_offset: torch.Tensor,
    row_starts: Optional[torch.Tensor],
) -> torch.Tensor:
    """Fused ragged top-k via the DeepSeek-V4 v2 JIT kernel.

    ``logits`` is written in place: the kernel reads from a 16-byte-aligned base
    and masks the <= 3 columns that pulls in ahead of the window. Those columns
    belong to a preceding request of the same row, and the score buffer is dead
    after the top-k (see ``DSAIndexer._get_topk_ragged``).

    Preconditions match the paged helper: fp32 scores with unit row stride and a
    16B-aligned row stride (DeepGEMM's contiguous-KV output satisfies this by
    construction), int32 non-negative lengths, and ``0 < topk <= 2048``.
    """
    from sglang.kernels.ops.attention.dsv4.topk import topk_transform_ragged_v2

    out = logits.new_empty((logits.shape[0], topk), dtype=torch.int32)
    topk_transform_ragged_v2(
        logits,
        lengths,
        out_offsets=topk_indices_offset,
        out_indices=out,
        row_starts=row_starts,
    )
    return out


def _build_flashinfer_paged_args(
    attn_metadata,
    row_starts: Optional[torch.Tensor],
    cu_seqlens_q_topk: Optional[torch.Tensor],
    batch_idx_list: Optional[List[int]],
    device: torch.device,
    num_rows: int,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    row_to_batch = (
        torch.as_tensor(batch_idx_list, dtype=torch.int32, device=device)
        if batch_idx_list is not None
        else None
    )

    # Both dynamic mappings contain one entry per logit row. Supplying the known
    # size avoids synchronizing CUDA to infer the sum of the repeat counts.
    if (
        row_to_batch is not None
        and cu_seqlens_q_topk is not None
        and row_to_batch.shape[0] != num_rows
    ):
        q_lens = (cu_seqlens_q_topk[1:] - cu_seqlens_q_topk[:-1]).to(
            dtype=torch.int32, device=device
        )
        row_to_batch = torch.repeat_interleave(
            row_to_batch, q_lens, output_size=num_rows
        )

    if row_to_batch is None and cu_seqlens_q_topk is not None:
        # Decode-like case (one query row per batch) does not need an explicit mapping.
        # Avoid dynamic tensor construction in this branch to keep CUDA graph capture safe.
        num_batches = cu_seqlens_q_topk.shape[0] - 1
        if not (row_starts is None and num_rows == num_batches):
            q_lens = (cu_seqlens_q_topk[1:] - cu_seqlens_q_topk[:-1]).to(
                dtype=torch.int32, device=device
            )
            row_to_batch = torch.repeat_interleave(
                torch.arange(q_lens.shape[0], dtype=torch.int32, device=device),
                q_lens,
                output_size=num_rows,
            )

    if row_starts is not None and row_to_batch is None:
        raise RuntimeError(
            "PAGED topk_transform with row_starts requires cu_seqlens_q metadata."
        )

    page_table_row_starts = row_starts
    if page_table_row_starts is not None and row_to_batch is not None:
        page_table_row_starts = (
            page_table_row_starts - attn_metadata.cu_seqlens_k[:-1][row_to_batch]
        )

    return row_to_batch, page_table_row_starts


def _flashinfer_tie_break_value() -> int:
    mode = envs.SGLANG_DSA_TOPK_FLASHINFER_TIE_BREAK.get()
    if mode is None:
        return 0
    mode = mode.lower()
    if mode not in _FLASHINFER_TIE_BREAK_VALUES:
        raise RuntimeError(
            "SGLANG_DSA_TOPK_FLASHINFER_TIE_BREAK must be one of "
            f"{tuple(_FLASHINFER_TIE_BREAK_VALUES)} or unset, got {mode!r}."
        )
    return _FLASHINFER_TIE_BREAK_VALUES[mode]
