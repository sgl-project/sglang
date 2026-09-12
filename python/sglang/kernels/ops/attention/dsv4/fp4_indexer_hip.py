"""AITER adapters for the DeepSeek-V4 FP4 indexer on HIP."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Optional, Tuple, Union

import torch

from sglang.srt.layers.attention.dsv4.fp4_logits_workspace import (
    fp4_logits_width_from_page_table,
    guarded_page_table_width,
)

if TYPE_CHECKING:
    from sglang.kernels.ops.attention.dsv4.compress import (
        CompressorDecodePlan,
        CompressorPrefillPlan,
    )
    from sglang.kernels.ops.attention.dsv4.fp4_indexer_schedule_hip import (
        PrefillScheduleBuffers,
    )


_HEADS = 64
_HEAD_DIM = 128
_ROPE_DIM = 64
_GROUP_SIZE = 32
_KV_BLOCK_SIZE = 64
_Q_SCALE_SHAPE = (1, 4, 16, 4)
# gfx950 has 256 CUs; target four persistent CTAs per CU.
_DECODE_BASE_CTA_TARGET = 1024
# Preserve per-query parallelism when the batch itself exceeds one CTA per CU.
_DECODE_CTAS_PER_QUERY = 4
_PREFILL_BASE_CTA_TARGET = 1024
# AITER varctx cta_info row: [batch_packed, chunk_start, chunk_count, ctx_len].
_DECODE_CTA_INFO_WIDTH = 4


class FP4DecodeWorkspace(NamedTuple):
    guarded_page_table: torch.Tensor
    c4_seq_lens: torch.Tensor
    cta_info: torch.Tensor
    cta_count: int
    max_seq_len: int
    # Held only so AITER's schedule scratch never returns to the graph memory
    # pool: the captured builder writes it again on every replay.
    schedule_scratch: torch.Tensor


class FP4PrefillWorkspace(NamedTuple):
    guarded_page_table: torch.Tensor
    row_to_batch: torch.Tensor
    local_starts: torch.Tensor
    cta_info: torch.Tensor
    cta_count: int
    max_seq_len: int
    # Prefix sums and scalars the fused prep kernel writes and AITER's
    # cta_info kernel reads. Pinned with the workspace so a refresh allocates
    # nothing and the buffers never return to the graph memory pool.
    schedule_buffers: Optional[PrefillScheduleBuffers] = None


class FP4PrefillChunkPlan(NamedTuple):
    start: int
    stop: int
    workspace: FP4PrefillWorkspace
    topk_metadata: Optional[torch.Tensor]


class FP4KWriteMetadata(NamedTuple):
    positions: torch.Tensor
    slots: torch.Tensor


def aiter_q_indexer_fp4(
    q: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE and Hadamard rotation, then quantize indexer Q to FP4."""
    import aiter

    num_tokens = q.shape[0]
    # AITER asserts int64 positions; the caller normally widens once per forward.
    if (
        positions.dtype is not torch.int64
        or positions.device != q.device
        or not positions.is_contiguous()
    ):
        positions = positions.to(device=q.device, dtype=torch.int64).contiguous()
    q_fp4 = torch.empty(
        (num_tokens, _HEADS, _HEAD_DIM // 2),
        dtype=aiter.dtypes.fp4x2,
        device=q.device,
    )
    q_scale = torch.empty(
        (num_tokens, *_Q_SCALE_SHAPE), dtype=torch.uint8, device=q.device
    )
    aiter.rope_rotate_activation(
        q_fp4,
        q,
        cos,
        sin,
        positions,
        rope_dim=_ROPE_DIM,
        out_scale=q_scale,
        group_size=_GROUP_SIZE,
        shuffle_scale=True,
        do_rotate_act=True,
    )
    return q_fp4, q_scale


def _as_int32_1d(t: torch.Tensor) -> torch.Tensor:
    """Normalize a length vector without dispatching when it already matches.

    Called once per C4 layer, so the no-op fast path matters: the metadata
    builder already hands us a 1-D contiguous int32 tensor.
    """
    if t.dim() == 1 and t.dtype is torch.int32 and t.is_contiguous():
        return t
    return t.reshape(-1).to(torch.int32).contiguous()


def _decode_cta_count(num_queries: int, max_seq_len: int) -> int:
    """Choose a bounded persistent grid without exceeding available KV chunks."""
    chunks_per_seq = max(1, (max_seq_len + 255) // 256)
    available_ctas = num_queries * chunks_per_seq
    target_ctas = max(_DECODE_BASE_CTA_TARGET, num_queries * _DECODE_CTAS_PER_QUERY)
    return min(available_ctas, target_ctas)


def _guarded_pages(logical_width: int) -> int:
    """Page columns after padding for 256-token scheduling."""
    return guarded_page_table_width(logical_width)


def _guard_page_table(page_table: torch.Tensor, out: Optional[torch.Tensor] = None):
    """Pad page tables for 256-token scheduling and one-chunk lookahead."""
    from sglang.kernels.ops.attention.dsv4.fp4_indexer_schedule_hip import (
        pad_page_table,
    )

    return pad_page_table(page_table, out=out)


def fp4_logits_max_seq_len(page_table: torch.Tensor) -> int:
    """Return the padded output width required by the FP4 score kernel."""
    return fp4_logits_width_from_page_table(page_table.shape[1])


def prepare_fp4_decode_workspace(
    page_table: torch.Tensor,
    c4_seq_lens: torch.Tensor,
) -> FP4DecodeWorkspace:
    """Build the decode page-table, schedule, and logits buffers.

    Safe to run under CUDA-graph capture: every tensor the captured schedule
    kernel touches is reachable from the returned workspace, so none of it can
    be handed out again by a later capture sharing the graph memory pool.
    """
    from aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4 import (
        compute_varctx_schedule,
    )

    guarded, max_seq_len = _guard_page_table(page_table)
    c4_seq_lens = _as_int32_1d(c4_seq_lens)
    num_queries = guarded.shape[0]
    cta_count = _decode_cta_count(num_queries, max_seq_len)
    cta_info = torch.empty(
        (cta_count, _DECODE_CTA_INFO_WIDTH),
        dtype=torch.int32,
        device=guarded.device,
    )
    schedule_scratch, _, _ = compute_varctx_schedule(
        c4_seq_lens,
        block_k=256,
        parallel_unit_num=cta_count,
        max_seq_len=max_seq_len,
        next_n=1,
        cta_info_out=cta_info,
    )
    return FP4DecodeWorkspace(
        guarded, c4_seq_lens, cta_info, cta_count, max_seq_len, schedule_scratch
    )


def prepare_fp4_prefill_workspace(
    page_table: torch.Tensor,
    c4_seq_lens: torch.Tensor,
    workspace: Optional[FP4PrefillWorkspace] = None,
) -> FP4PrefillWorkspace:
    """Build or refresh the prefill page-table, schedule, and logits buffers.

    Must run OUTSIDE CUDA-graph capture. Rows past the fused builder's limit
    fall back to AITER's prefill scheduler, which frees the scratch its own
    schedule kernel reads, so a captured build would replay against recycled
    graph-pool memory. Callers instead refresh this workspace per step and let
    the graph read only the pinned ``cta_info`` and page-table buffers.
    """
    from aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4_prefill import (
        CTA_INFO_WIDTH,
    )

    from sglang.kernels.ops.attention.dsv4.fp4_indexer_schedule_hip import (
        PrefillScheduleBuffers,
        build_prefill_schedule,
        padded_page_table_shape,
    )

    c4_seq_lens = _as_int32_1d(c4_seq_lens)
    rows, _, padded_width = padded_page_table_shape(page_table)
    expected_cta_count = max(_PREFILL_BASE_CTA_TARGET, rows)
    if workspace is not None and (
        workspace.guarded_page_table.shape != (rows, padded_width + 4)
        or workspace.guarded_page_table.device != page_table.device
        or workspace.row_to_batch.shape != (rows,)
        or workspace.local_starts.shape != (rows,)
        or workspace.cta_info.shape[0] != expected_cta_count
        or workspace.max_seq_len != padded_width * _KV_BLOCK_SIZE
    ):
        workspace = None
    if workspace is None:
        cta_count = expected_cta_count
        device = page_table.device
        buffers = PrefillScheduleBuffers(rows, device)
        workspace = FP4PrefillWorkspace(
            # The prep kernel writes every element it hands back, so neither the
            # padded table nor the row metadata needs a zero-fill dispatch here.
            guarded_page_table=torch.empty(
                (rows, padded_width + 4), dtype=torch.int32, device=device
            ),
            row_to_batch=buffers.row_to_batch,
            local_starts=buffers.local_starts,
            cta_info=torch.empty(
                (cta_count, CTA_INFO_WIDTH), dtype=torch.int32, device=device
            ),
            cta_count=cta_count,
            max_seq_len=padded_width * _KV_BLOCK_SIZE,
            schedule_buffers=buffers,
        )

    assert c4_seq_lens.shape[0] == workspace.row_to_batch.shape[0], (
        f"c4_seq_lens rows {c4_seq_lens.shape[0]} do not match the workspace's "
        f"{workspace.row_to_batch.shape[0]}; the schedule kernel indexes both by row"
    )
    build_prefill_schedule(
        page_table=page_table,
        local_ends=c4_seq_lens,
        cta_info_out=workspace.cta_info,
        parallel_unit_num=workspace.cta_count,
        max_seq_len=workspace.max_seq_len,
        block_k=256,
        guarded_out=workspace.guarded_page_table,
        buffers=workspace.schedule_buffers,
    )
    return workspace


def aiter_fp4_paged_mqa_logits(
    *,
    q_fp4: torch.Tensor,
    q_scale: torch.Tensor,
    k_payload: torch.Tensor,
    k_scale: torch.Tensor,
    weights: torch.Tensor,
    page_table: torch.Tensor,
    c4_seq_lens: torch.Tensor,
    weight_scale: float,
    is_decode: bool,
    decode_workspace: Optional[FP4DecodeWorkspace] = None,
    prefill_workspace: Optional[FP4PrefillWorkspace] = None,
    logits_out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute FP4 Q/K indexer logits with the decode or prefill FlyDSL kernel."""
    from aiter.ops.flydsl import (
        flydsl_pa_mqa_logits_fp4,
        flydsl_pa_mqa_logits_fp4_prefill,
    )

    num_tokens = q_fp4.shape[0]
    c4_seq_lens = _as_int32_1d(c4_seq_lens)
    workspace = decode_workspace if is_decode else prefill_workspace
    # A workspace is bound to one row count. DP padding or truncated activations
    # can leave it stale, in which case fall back to building the schedule here.
    expected_max_seq_len = fp4_logits_max_seq_len(page_table)
    if workspace is not None and (
        workspace.guarded_page_table.shape[0] != num_tokens
        or workspace.guarded_page_table.device != page_table.device
        or workspace.max_seq_len != expected_max_seq_len
    ):
        workspace = None
    # Built on the fallback path below; kept in scope so the schedule scratch
    # outlives the logits kernel that reads it.
    fallback_schedule = None
    if workspace is not None:
        page_table = workspace.guarded_page_table
        max_seq_len = workspace.max_seq_len
    elif is_decode:
        page_table, max_seq_len = _guard_page_table(page_table)
    else:
        # No usable workspace (DP padding or truncated activations): build the
        # schedule here rather than letting AITER rebuild it from ~29 torch ops
        # once per C4 layer. This pads the page table in the same dispatch.
        from aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4_prefill import (
            CTA_INFO_WIDTH,
        )

        from sglang.kernels.ops.attention.dsv4.fp4_indexer_schedule_hip import (
            build_prefill_schedule,
            padded_page_table_shape,
        )

        _, _, padded_width = padded_page_table_shape(page_table)
        max_seq_len = padded_width * _KV_BLOCK_SIZE
        cta_count = max(_PREFILL_BASE_CTA_TARGET, num_tokens)
        cta_info = torch.empty(
            (cta_count, CTA_INFO_WIDTH),
            dtype=torch.int32,
            device=page_table.device,
        )
        page_table, buffers = build_prefill_schedule(
            page_table=page_table,
            local_ends=c4_seq_lens,
            cta_info_out=cta_info,
            parallel_unit_num=cta_count,
            max_seq_len=max_seq_len,
            block_k=256,
        )
        fallback_schedule = (cta_info, cta_count, buffers)
    q_payload = q_fp4.view(torch.uint8)
    k_payload = k_payload.view(torch.uint8)
    if logits_out is None:
        logits = torch.empty(
            (num_tokens, max_seq_len), dtype=torch.float32, device=q_fp4.device
        )
    else:
        expected_shape = (num_tokens, max_seq_len)
        if (
            logits_out.shape != expected_shape
            or logits_out.dtype is not torch.float32
            or logits_out.device != q_fp4.device
            or not logits_out.is_contiguous()
        ):
            raise ValueError(
                "Managed FP4 logits output must be contiguous FP32 on the Q "
                f"device with shape {expected_shape}; got shape "
                f"{tuple(logits_out.shape)}, dtype={logits_out.dtype}, "
                f"device={logits_out.device}, "
                f"contiguous={logits_out.is_contiguous()}"
            )
        logits = logits_out
    common = {
        "weight_scale": weight_scale,
        "block_k": 256,
        "kv_block_size": _KV_BLOCK_SIZE,
        "num_warps": 4,
        "out": logits,
    }

    if is_decode:
        pinned = (
            {}
            if workspace is None
            else {
                "cta_info": workspace.cta_info,
                "total_ctas": workspace.cta_count,
            }
        )
        logits = flydsl_pa_mqa_logits_fp4(
            q_payload.reshape(num_tokens, 1, _HEADS, _HEAD_DIM // 2),
            q_scale.reshape(num_tokens, 1, *_Q_SCALE_SHAPE),
            k_payload,
            k_scale,
            page_table,
            weights,
            c4_seq_lens,
            max_seq_len,
            next_n=1,
            parallel_unit_num=None,
            **pinned,
            **common,
        )
    else:
        if workspace is None:
            cta_info, cta_count, buffers = fallback_schedule
            # As with the workspace path, a pinned cta_info lets the kernel skip
            # its -inf pre-fill: every row the length-aware top-k reads is
            # covered by a CTA.
            pinned = {"cta_info": cta_info, "n_ctas": cta_count}
            row_to_batch = buffers.row_to_batch
            local_starts = buffers.local_starts
        else:
            pinned = {
                "cta_info": workspace.cta_info,
                "n_ctas": workspace.cta_count,
            }
            row_to_batch = workspace.row_to_batch
            local_starts = workspace.local_starts
        logits = flydsl_pa_mqa_logits_fp4_prefill(
            q_payload,
            q_scale,
            k_payload,
            k_scale,
            page_table,
            weights,
            row_to_batch,
            local_starts,
            c4_seq_lens,
            max_seq_len,
            parallel_unit_num=max(_PREFILL_BASE_CTA_TARGET, num_tokens),
            **pinned,
            **common,
        )

    return logits


def prepare_fp4_k_write_metadata(
    plan: Union[CompressorDecodePlan, CompressorPrefillPlan],
    out_loc: torch.Tensor,
    rope_table_len: int,
) -> FP4KWriteMetadata:
    """
    Build RoPE positions and cache slots from a compressor plan.
    """
    plan_words = plan[1].view(torch.int32)
    seq_lens = plan_words[:, 0].to(torch.int64)
    positions = seq_lens - plan.compress_ratio
    valid = (positions >= 0) & (positions < rope_table_len)
    positions = torch.where(valid, positions, torch.zeros_like(positions))
    valid &= seq_lens % plan.compress_ratio == 0

    out_loc = out_loc.to(dtype=torch.int64)
    if plan.is_decode:
        slots = out_loc
    elif out_loc.shape[0] == 0:
        slots = torch.full_like(seq_lens, -1)
        valid.zero_()
    else:
        ragged_ids = plan_words[:, 1].bitwise_and(0xFFFF).to(torch.int64)
        valid &= ragged_ids < out_loc.shape[0]
        slots = out_loc[ragged_ids.clamp(max=out_loc.shape[0] - 1)]
    slots = torch.where(valid, slots, torch.full_like(slots, -1))
    return FP4KWriteMetadata(positions.contiguous(), slots.contiguous())


def aiter_k_indexer_fp4_cache_write(
    *,
    k: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_epsilon: float,
    cos: torch.Tensor,
    sin: torch.Tensor,
    plan: Union[CompressorDecodePlan, CompressorPrefillPlan],
    out_loc: torch.Tensor,
    k_payload: torch.Tensor,
    k_scale: torch.Tensor,
    write_metadata: Optional[FP4KWriteMetadata] = None,
) -> None:
    """
    Map compressed K rows to cache slots and run the fused AITER FP4 writer.
    """
    num_rows = k.shape[0]
    if num_rows == 0:
        return

    assert write_metadata is not None, "FP4 K-write metadata is missing."

    positions, slots = write_metadata
    # The compressor normally hands over its BF16 mirror; convert only when some
    # caller still passes the FP32 parameter.
    if norm_weight.dtype is not torch.bfloat16 or norm_weight.device != k.device:
        norm_weight = norm_weight.to(device=k.device, dtype=torch.bfloat16).contiguous()

    import aiter

    aiter.rmsnorm_rope_rotate_activation_fp4quant_kvcache(
        k_payload,
        k_scale,
        k.view(num_rows, 1, _HEAD_DIM),
        norm_weight,
        cos,
        sin,
        positions,
        slots,
        norm_epsilon,
        rope_dim=_ROPE_DIM,
        kv_block_size=_KV_BLOCK_SIZE,
        group_size=_GROUP_SIZE,
        shuffle_scale=True,
        do_rotate_act=True,
    )
