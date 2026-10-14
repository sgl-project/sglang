"""AITER adapters for the DeepSeek-V4 FP4 indexer on HIP."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, NamedTuple, Optional, Tuple, Union

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from sglang.kernels.ops.attention.dsv4.fp4_indexer import (
    _ceil_ue8m0_exp,
    quantize_fp4_indexer_row,
    quantize_fp4_indexer_tensor,
)
from sglang.kernels.ops.attention.dsv4.fp4_rope_fake_quant import (
    FP4_AMAX_FLOOR,
    rope_tail_fake_quant_fp4_row,
)
from sglang.kernels.ops.gemm.router_gemv_hip import rocm_router_gemv_split_k

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
# FlyDSL compiles one kernel per page-table width: bucket it so a new context length pays no JIT
LOW_RATIO_PAGE_TABLE_BUCKET = 64
# AITER varctx cta_info row: [batch_packed, chunk_start, chunk_count, ctx_len].
_DECODE_CTA_INFO_WIDTH = 4

# Budget for the pooled prefill logits block, in MiB. Rows are split to fit it
# (see `logits_rows_per_chunk`), so this caps the indexer's transient footprint
# independently of context length and chunked-prefill size; smaller budgets only
# buy more row chunks. 2 GiB covers 4096 rows over ~512K tokens of context.
_LOGITS_BUDGET_ELEMS = (
    int(os.environ.get("SGLANG_DSV4_FP4_LOGITS_BUDGET_MB", "2048")) * 2**20 // 4
)
_LOGITS_POOL: dict = {}


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


def _guarded_pages(logical_width: int, page_table_bucket: int = 4) -> int:
    """Page columns after padding for 256-token scheduling."""
    return max(
        4,
        (logical_width + page_table_bucket - 1)
        // page_table_bucket
        * page_table_bucket,
    )


def _guard_page_table(
    page_table: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    page_table_bucket: int = 4,
):
    """Pad page tables for 256-token scheduling and one-chunk lookahead."""
    from sglang.kernels.ops.attention.dsv4.fp4_indexer_schedule_hip import (
        pad_page_table,
    )

    return pad_page_table(page_table, out=out, page_table_bucket=page_table_bucket)


def logits_rows_per_chunk(page_table: torch.Tensor, page_table_bucket: int = 4) -> int:
    """Rows whose logits fit the pooled block, for callers that loop by row."""
    width = _guarded_pages(page_table.shape[1], page_table_bucket) * _KV_BLOCK_SIZE
    return max(1, _LOGITS_BUDGET_ELEMS // width)


def _alloc_logits(
    num_tokens: int, max_seq_len: int, device: torch.device, is_decode: bool
) -> torch.Tensor:
    """Hand out the [num_tokens, max_seq_len] fp32 scratch the logits kernel fills.

    Prefill rectangles are served from one fixed-size pooled block. A fresh
    `torch.empty` per call would instead feed the caching allocator a
    monotonically growing size sequence -- the width tracks context length, and
    an agentic session's context only ever grows -- so every request is slightly
    larger than any cached block, none can be reused, and each strands a whole
    segment. `reserved` then climbs while `allocated` stays flat, and that
    stranded memory is invisible to allocators that bypass torch: Triton kernel
    scratch fails with HSA_STATUS_ERROR_OUT_OF_RESOURCES instead of surfacing as
    a clean torch OOM. Serving every rectangle out of one block keeps the
    request size constant, so the block is always reused and nothing strands.

    Decode keeps the plain allocation: it is captured against the graph memory
    pool (bounded, separate from the fragmenting general pool), and creating the
    pooled block mid-capture would hand out graph-pool memory to later replays.
    """
    n = num_tokens * max_seq_len
    if (
        is_decode
        or n > _LOGITS_BUDGET_ELEMS
        or torch.cuda.is_current_stream_capturing()
    ):
        return torch.empty(
            (num_tokens, max_seq_len), dtype=torch.float32, device=device
        )
    buf = _LOGITS_POOL.get(device)
    if buf is None:
        buf = torch.empty(_LOGITS_BUDGET_ELEMS, dtype=torch.float32, device=device)
        _LOGITS_POOL[device] = buf
    return buf[:n].view(num_tokens, max_seq_len)


def prepare_fp4_decode_workspace(
    page_table: torch.Tensor,
    c4_seq_lens: torch.Tensor,
    page_table_bucket: int = 4,
) -> FP4DecodeWorkspace:
    """Build the decode page-table, schedule, and logits buffers.

    Safe to run under CUDA-graph capture: every tensor the captured schedule
    kernel touches is reachable from the returned workspace, so none of it can
    be handed out again by a later capture sharing the graph memory pool.
    """
    from sglang.kernels.ops.attention.dsv4.fp4_indexer_schedule_hip import (
        build_decode_schedule,
    )

    guarded, max_seq_len = _guard_page_table(
        page_table, page_table_bucket=page_table_bucket
    )
    c4_seq_lens = _as_int32_1d(c4_seq_lens)
    num_queries = guarded.shape[0]
    cta_count = _decode_cta_count(num_queries, max_seq_len)
    cta_info = torch.empty(
        (cta_count, _DECODE_CTA_INFO_WIDTH),
        dtype=torch.int32,
        device=guarded.device,
    )
    schedule_scratch = build_decode_schedule(
        c4_seq_lens, cta_info_out=cta_info, max_seq_len=max_seq_len
    )
    return FP4DecodeWorkspace(
        guarded, c4_seq_lens, cta_info, cta_count, max_seq_len, schedule_scratch
    )


def prepare_fp4_prefill_workspace(
    page_table: torch.Tensor,
    c4_seq_lens: torch.Tensor,
    workspace: Optional[FP4PrefillWorkspace] = None,
    page_table_bucket: int = 4,
) -> FP4PrefillWorkspace:
    """Build or refresh the prefill page-table, schedule, and logits buffers.

    Must run OUTSIDE CUDA-graph capture. Rows past the fused builder's limit
    fall back to AITER's prefill scheduler, which frees the scratch its own
    schedule kernel reads, so a captured build would replay against recycled
    graph-pool memory. Callers instead refresh this workspace per step and let
    the graph read only the pinned ``cta_info`` / ``logits`` / page-table
    buffers.
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
    if workspace is None:
        rows, _, padded_width = padded_page_table_shape(page_table, page_table_bucket)
        cta_count = max(_PREFILL_BASE_CTA_TARGET, rows)
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
        page_table_bucket=page_table_bucket,
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
    page_table_bucket: int = 4,
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
    if workspace is not None and workspace.guarded_page_table.shape[0] != num_tokens:
        workspace = None
    # Built on the fallback path below; kept in scope so the schedule scratch
    # outlives the logits kernel that reads it.
    fallback_schedule = None
    if workspace is not None:
        page_table = workspace.guarded_page_table
        max_seq_len = workspace.max_seq_len
    elif is_decode:
        # No usable workspace: build the schedule here too, rather than letting
        # AITER build it with its varctx kernel, which cannot reserve its scratch
        # at large batches.
        from sglang.kernels.ops.attention.dsv4.fp4_indexer_schedule_hip import (
            build_decode_schedule,
        )

        page_table, max_seq_len = _guard_page_table(
            page_table, page_table_bucket=page_table_bucket
        )
        cta_count = _decode_cta_count(num_tokens, max_seq_len)
        cta_info = torch.empty(
            (cta_count, _DECODE_CTA_INFO_WIDTH),
            dtype=torch.int32,
            device=page_table.device,
        )
        scratch = build_decode_schedule(
            c4_seq_lens, cta_info_out=cta_info, max_seq_len=max_seq_len
        )
        fallback_schedule = (cta_info, cta_count, scratch)
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

        _, _, padded_width = padded_page_table_shape(page_table, page_table_bucket)
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
            page_table_bucket=page_table_bucket,
        )
        fallback_schedule = (cta_info, cta_count, buffers)
    q_payload = q_fp4.view(torch.uint8)
    k_payload = k_payload.view(torch.uint8)
    # Scored write-once and dead when the caller's top-k returns, so the pooled
    # block can be handed straight to the next call: a pinned cta_info makes the
    # kernel skip its -inf pre-fill and the length-aware top-k reads only
    # [0, c4_seq_len), so neither ever observes the previous chunk's leftovers.
    logits = _alloc_logits(num_tokens, max_seq_len, q_fp4.device, is_decode)
    common = {
        "weight_scale": weight_scale,
        "block_k": 256,
        "kv_block_size": _KV_BLOCK_SIZE,
        "num_warps": 4,
        "out": logits,
    }

    if is_decode:
        if workspace is None:
            cta_info, cta_count, _ = fallback_schedule
            pinned = {"cta_info": cta_info, "total_ctas": cta_count}
        else:
            pinned = {
                "cta_info": workspace.cta_info,
                "total_ctas": workspace.cta_count,
            }
        logits = flydsl_pa_mqa_logits_fp4(
            q_payload.reshape(num_tokens, 1, q_fp4.shape[-2], _HEAD_DIM // 2),
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


# FlyDSL split payload / scale layout; bytes equal AITER's fp4 writer up to the sign of zero
@triton.jit
def _store_fp4_index_k_split_kernel(
    k_fp4,
    k_sf,
    payload,
    scale,
    loc,
    page_size: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token_id = tl.program_id(0)
    cache_loc = tl.load(loc + token_id)
    page = cache_loc // page_size
    page_offset = cache_loc - page * page_size

    # payload [page, 1, 4 chunks, page_size, 16 bytes]: chunk c holds dims [32c, 32c + 32).
    offsets = tl.arange(0, BLOCK)
    chunk = offsets // 16
    byte = offsets - chunk * 16
    k = tl.load(k_fp4 + token_id * BLOCK + offsets)
    tl.store(
        payload
        + page * (4 * page_size * 16)
        + chunk * (page_size * 16)
        + page_offset * 16
        + byte,
        k,
    )
    # scale [page, 1, 4, page_size]: the slot axis is a 16 x 4 tile transposed (the FlyDSL K ABI)
    shuffled = (page_offset % 16) * 4 + page_offset // 16
    sf = tl.load(k_sf + token_id)
    sf_offsets = tl.arange(0, 4)
    sf_bytes = ((sf >> (sf_offsets * 8)) & 0xFF).to(tl.uint8)
    tl.store(
        scale + page * (4 * page_size) + sf_offsets * page_size + shuffled, sf_bytes
    )


def store_fp4_index_k_cache_split(
    input: torch.Tensor,
    payload: torch.Tensor,
    scale: torch.Tensor,
    loc: torch.Tensor,
    *,
    page_size: int,
    rne: bool = False,
) -> None:
    """Quantize input [n, 128] to fp4 (per-32 ue8m0) and scatter row i to slot
    loc[i] of the split FlyDSL K layout (payload [pages, 1, 4, page_size, 16],
    scale [pages, 1, 4, page_size] with the slot axis 16 x 4 transposed)."""
    assert input.shape[-1] == _HEAD_DIM
    assert page_size == _KV_BLOCK_SIZE, "the 16 x 4 slot transpose spans a 64-slot page"
    assert payload.shape[1:] == (1, 4, page_size, 16), payload.shape
    assert scale.shape[1:] == (1, 4, page_size), scale.shape
    k_fp4, k_sf = quantize_fp4_indexer_tensor(input.contiguous(), rne=rne)
    n_tokens = k_fp4.shape[0]
    if n_tokens == 0:
        return
    _store_fp4_index_k_split_kernel[(n_tokens,)](
        k_fp4.view(torch.uint8),
        k_sf,
        payload.view(torch.uint8),
        scale,
        loc,
        page_size,
        BLOCK=64,
    )


def read_fp4_index_k_split(
    payload: torch.Tensor, scale: torch.Tensor, slots: torch.Tensor, *, page_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Inverse of store_fp4_index_k_cache_split: (payload int8 [n, 64], scales
    int32 [n] with chunk c's e8m0 byte at bits 8c..8c+7), the layout of
    quantize_fp4_indexer_tensor."""
    assert page_size == _KV_BLOCK_SIZE, "the 16 x 4 slot transpose spans a 64-slot page"
    slots = slots.to(torch.int64)
    page, off = slots // page_size, slots % page_size
    rows = payload.view(torch.uint8)[page, 0, :, off, :]  # [n, 4, 16]
    rows = rows.reshape(-1, 64).view(torch.int8)
    shuffled = (off % 16) * 4 + off // 16
    sf = scale[page, 0, :, shuffled].to(torch.int32)  # [n, 4]
    packed = sf[:, 0] | (sf[:, 1] << 8) | (sf[:, 2] << 16) | (sf[:, 3] << 24)
    return rows, packed


@triton.jit
def _quantize_fp4_query_flydsl_kernel(
    x,
    x_fp4,
    q_scale,
    HEADS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_N: tl.constexpr,
):
    """One program per (token, head slot of 64): quantize_fp4_indexer_row (RNE codes) with
    the e8m0 byte of chunk c stored straight into the FlyDSL scale layout
    [t, 0, c, h % 16, h // 16]; head slots past HEADS write zero bytes."""
    token_id = tl.program_id(0)
    h = tl.program_id(1)
    scale_base = q_scale + token_id * (4 * 16 * 4) + (h % 16) * 4 + h // 16
    chunks = tl.arange(0, 4)
    if h >= HEADS:
        tl.store(scale_base + chunks * (16 * 4), tl.zeros([4], dtype=tl.uint8))
        return
    row = token_id * HEADS + h
    values = tl.load(x + row * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.float32)
    v0, v1 = tl.split(tl.reshape(values, (BLOCK_N // 2, 2)))
    packed, packed_sf = quantize_fp4_indexer_row(
        values, v0, v1, BLOCK_N=BLOCK_N, GROUP_N=GROUP_N, RNE=True
    )
    sf_bytes = ((packed_sf >> (8 * chunks)) & 0xFF).to(tl.uint8)
    tl.store(scale_base + chunks * (16 * 4), sf_bytes)
    tl.store(x_fp4 + row * (BLOCK_N // 2) + tl.arange(0, BLOCK_N // 2), packed)


def pack_fp4_query_flydsl(q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """fp4-grid query [T, H, 128] -> (payload int8 [T, H, 64], scale uint8 [T, 1, 4, 16, 4]) in the
    FlyDSL MQA-logits layout: head h, chunk c's e8m0 byte at [t, 0, c, h % 16, h // 16] (H <= 64),
    RNE as the CUDA low-ratio path."""
    num_tokens, heads = q.shape[0], q.shape[1]
    assert heads % 16 == 0 and heads <= 64, (
        f"{heads} heads: the FlyDSL scale layout holds 16-head groups up to 64"
    )
    assert q.shape[-1] == _HEAD_DIM
    x = q.contiguous().view(-1, _HEAD_DIM)
    q_fp4 = torch.empty(
        (num_tokens, heads, _HEAD_DIM // 2), dtype=torch.int8, device=q.device
    )
    q_scale = torch.empty((num_tokens, 1, 4, 16, 4), dtype=torch.uint8, device=q.device)
    if num_tokens > 0:
        _quantize_fp4_query_flydsl_kernel[(num_tokens, 64)](
            x,
            q_fp4,
            q_scale,
            HEADS=heads,
            BLOCK_N=_HEAD_DIM,
            GROUP_N=_GROUP_SIZE,
        )
    return q_fp4, q_scale


@triton.jit
def _reduce_scale_bf16_block(
    block,
    part_ptr,
    out_ptr,
    M,
    stride_ps,
    stride_pm,
    stride_om,
    scale,
    N: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One BLOCK of _reduce_scale_bf16_kernel (shared with the fused index-Q launch)."""
    offs = block * BLOCK + tl.arange(0, BLOCK)
    m = offs // N
    n = offs % N
    mask = m < M
    acc = tl.load(part_ptr + m * stride_pm + n, mask=mask, other=0.0)
    for s in tl.static_range(1, SPLIT_K):
        acc += tl.load(
            part_ptr + s * stride_ps + m * stride_pm + n, mask=mask, other=0.0
        )
    # bf16(bf16(sum) * scale): the linear's bf16 output, then the aten multiply rounded to bf16
    w = acc.to(tl.bfloat16).to(tl.float32) * scale
    tl.store(out_ptr + m * stride_om + n, w.to(tl.bfloat16), mask=mask)


@triton.jit
def _reduce_scale_bf16_kernel(
    part_ptr,
    out_ptr,
    M,
    stride_ps,
    stride_pm,
    stride_om,
    scale,
    N: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    _reduce_scale_bf16_block(
        tl.program_id(0),
        part_ptr,
        out_ptr,
        M,
        stride_ps,
        stride_pm,
        stride_om,
        scale,
        N=N,
        SPLIT_K=SPLIT_K,
        BLOCK=BLOCK,
    )


@triton.jit
def _fp4_e2m1_code_rne_lean(x):
    """_fp4_e2m1_code_rne for every fp32 x (checked over all 2**32): within each spacing of the
    e2m1 grid, rint's ties-to-even picks the even code at the midpoints."""
    ax = tl.minimum(tl.abs(x), 6.0)
    lo = ax < 2.0
    mid = ax < 4.0
    rstep = tl.where(lo, 2.0, tl.where(mid, 1.0, 0.5))
    idx = (libdevice.rint(ax * rstep) + tl.where(lo, 0.0, tl.where(mid, 2.0, 4.0))).to(
        tl.uint8
    )
    idx = tl.where(ax == ax, idx, 0)
    sign = ((x < 0) & (idx != 0)).to(tl.uint8)
    return idx | (sign << 3)


@triton.jit
def _index_q_quant_groups(v, AMAX_FLOOR: tl.constexpr):
    """[16, G, 32] fp32 through the fake-quant of rope_tail_fake_quant_fp4_row (rounded to bf16
    as the standalone kernel stores it) and quantize_fp4_indexer_row (RNE): (packed codes
    [16, G * 16], ue8m0 exponents [16, G]). The power-of-two divisions run as multiplies by
    exact reciprocals, which round identically."""
    amax = tl.maximum(tl.max(tl.abs(v), axis=2), AMAX_FLOOR) * (1.0 / 6.0)
    bits = amax.to(tl.int32, bitcast=True)
    expo = ((bits >> 23) & 0xFF) - 127
    expo = expo + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    scale = ((expo + 127) << 23).to(tl.float32, bitcast=True)
    # 1 / scale: 2^-expo for a finite amax (expo <= 126), 0 for inf (scale = inf),
    # -inf for nan (expo = 129 wraps scale to -0.0)
    rscale = tl.where(
        expo <= 126,
        ((127 - expo) << 23).to(tl.float32, bitcast=True),
        tl.where(expo == 128, 0.0, float("-inf")),
    )
    s = tl.minimum(tl.maximum(v * rscale[:, :, None], -6.0), 6.0)
    mag = tl.abs(s)
    rstep = tl.where(mag < 2.0, 2.0, tl.where(mag < 4.0, 1.0, 0.5))
    step = ((254 << 23) - rstep.to(tl.int32, bitcast=True)).to(tl.float32, bitcast=True)
    # rint is odd-symmetric, so the sign rides along; a zero may come out as -0, which neither
    # the amax nor the codes below tell apart
    values = (libdevice.rint(s * rstep) * step * scale[:, :, None]).to(tl.bfloat16)
    values = values.to(tl.float32)

    exp = _ceil_ue8m0_exp(tl.maximum(tl.max(tl.abs(values), axis=2) / 6.0, 1.0e-4))
    # 1 / 2^(exp - 127); exp = 254 (an inf / nan group) as 0.5 * 2^-126, where any value
    # rounded differently is below 2^-126 and codes 0 either way
    top = exp == 254
    ra = tl.where(top, 0.5, 1.0)
    rb = tl.where(top, 2.0**-126, ((254 - exp) << 23).to(tl.float32, bitcast=True))
    code = _fp4_e2m1_code_rne_lean(values * ra[:, :, None] * rb[:, :, None])
    G: tl.constexpr = v.shape[1]
    c0, c1 = tl.split(tl.reshape(code, (16, G * 16, 2)))
    return (c0 & 0x0F) | ((c1 & 0x0F) << 4), exp


@triton.jit
def _index_q_rope_pack_kernel(
    q_ptr,  # [T, H * 128] bf16 (wq_b output)
    f_ptr,  # [max_pos, RD // 2, 2] fp32: the real view of the complex freqs table
    pos_ptr,  # [T] int
    q_fp4_ptr,  # [T, H, 64] int8
    q_scale_ptr,  # [T, 1, 4, 16, 4] uint8
    stride_qt,
    num_pos,
    H: tl.constexpr,
    D: tl.constexpr,
    RD: tl.constexpr,
    AMAX_FLOOR: tl.constexpr,
):
    """Grid (T, H // 16): 16 heads of one token per program, through _index_q_quant_groups, in
    the FlyDSL MQA-logits layout."""
    NOPE: tl.constexpr = D - RD
    t = tl.program_id(0)
    h0 = tl.program_id(1) * 16
    hh = h0 + tl.arange(0, 16)
    pos = tl.load(pos_ptr + t).to(tl.int64)
    # the caller owns positions < num_pos; an out-of-range row reads entry 0 instead of past the table
    pos = tl.where((pos >= 0) & (pos < num_pos), pos, 0)
    q_row = q_ptr + t.to(tl.int64) * stride_qt + hh[:, None] * D
    out_row = q_fp4_ptr + (t.to(tl.int64) * H + hh[:, None]) * (D // 2)

    x = tl.load(q_row + tl.arange(0, NOPE)[None, :]).to(tl.float32)
    packed, exp_nope = _index_q_quant_groups(
        tl.reshape(x, (16, NOPE // 32, 32)), AMAX_FLOOR
    )
    tl.store(out_row + tl.arange(0, NOPE // 2)[None, :], packed)

    # RoPE on the last RD features, adjacent pairs as one complex number
    x = tl.load(q_row + NOPE + tl.arange(0, RD)[None, :]).to(tl.float32)
    re, im = tl.split(tl.reshape(x, (16, RD // 2, 2)))
    j = tl.arange(0, RD // 2)
    fr = tl.load(f_ptr + pos * RD + 2 * j)[None, :]
    fi = tl.load(f_ptr + pos * RD + 2 * j + 1)[None, :]
    re_rot = (re * fr - im * fi).to(tl.bfloat16).to(tl.float32)
    im_rot = (re * fi + im * fr).to(tl.bfloat16).to(tl.float32)
    packed, exp_rope = _index_q_quant_groups(
        tl.reshape(tl.join(re_rot, im_rot), (16, RD // 32, 32)), AMAX_FLOOR
    )
    tl.store(out_row + NOPE // 2 + tl.arange(0, RD // 2)[None, :], packed)

    # scale bytes: chunk c of head h at [t, 0, c, h % 16, h // 16]
    base = q_scale_ptr + t.to(tl.int64) * (4 * 16 * 4)
    i = tl.arange(0, 16)[:, None]
    c_nope = tl.arange(0, NOPE // 32)[None, :]
    c_rope = NOPE // 32 + tl.arange(0, RD // 32)[None, :]
    tl.store(base + (c_nope * 16 + i) * 4 + h0 // 16, exp_nope.to(tl.uint8))
    tl.store(base + (c_rope * 16 + i) * 4 + h0 // 16, exp_rope.to(tl.uint8))
    if h0 == 0:
        # groups this head count does not have stay zero
        slot = (tl.arange(0, 4)[None, :] * 16 + i) * 4
        for g in tl.static_range(H // 16, 4):
            tl.store(base + slot + g, tl.zeros((16, 4), dtype=tl.uint8))


@triton.jit
def _index_q_pack_weights_kernel(
    q_ptr,  # [T, H * 128] bf16 (wq_b output)
    f_ptr,  # [max_pos, RD // 2, 2] fp32: the real view of the complex freqs table
    pos_ptr,  # [T] int64
    q_fp4_ptr,  # [T, H, 64] int8
    q_scale_ptr,  # [T, 1, 4, 16, 4] uint8
    part_ptr,  # [SPLIT_K, T, H] fp32 head-weight partials
    weights_ptr,  # [T, H] bf16
    stride_qt,
    stride_ps,
    stride_pm,
    weight_scale,
    T,
    num_pos,
    H: tl.constexpr,
    D: tl.constexpr,
    RD: tl.constexpr,
    AMAX_FLOOR: tl.constexpr,
    SPLIT_K: tl.constexpr,
    W_BLOCK: tl.constexpr,
):
    """Grid (T * H + 1,). Program t * H + h: the query row through rope_tail_fake_quant_fp4_row
    (rounded to bf16 as the standalone kernel stores it) and quantize_fp4_indexer_row (RNE), in
    the FlyDSL MQA-logits layout; program T * H: the head weights' _reduce_scale_bf16_block."""
    pid = tl.program_id(0)
    if pid < T * H:
        t = pid // H
        h = pid % H
        pos = tl.load(pos_ptr + t)
        # the caller owns positions < num_pos; an out-of-range row reads entry 0 instead of past the table
        pos = tl.where((pos >= 0) & (pos < num_pos), pos, 0)
        fq = rope_tail_fake_quant_fp4_row(
            q_ptr + t.to(tl.int64) * stride_qt + h * D,
            f_ptr + pos * RD,
            D=D,
            RD=RD,
            BLK=32,
            AMAX_FLOOR=AMAX_FLOOR,
            INVERSE=False,
            COMPRESSED_KV=False,
        )
        # the standalone path stores the fake-quant as bf16 and the packer reloads it
        values = fq.to(tl.bfloat16).to(tl.float32)
        v0, v1 = tl.split(tl.reshape(values, (D // 2, 2)))
        packed, packed_sf = quantize_fp4_indexer_row(
            values, v0, v1, BLOCK_N=D, GROUP_N=32, RNE=True
        )
        tl.store(q_fp4_ptr + pid.to(tl.int64) * (D // 2) + tl.arange(0, D // 2), packed)
        # scale bytes: chunk c of head h at [t, 0, c, h % 16, h // 16]
        c = tl.arange(0, 4)
        sf_bytes = ((packed_sf >> (8 * c)) & 0xFF).to(tl.uint8)
        base = q_scale_ptr + t.to(tl.int64) * (4 * 16 * 4) + c * (16 * 4) + (h % 16) * 4
        tl.store(base + h // 16, sf_bytes)
        if h < 16:
            # groups this head count does not have stay zero
            for g in tl.static_range(H // 16, 4):
                tl.store(base + g, tl.zeros((4,), dtype=tl.uint8))
    else:
        _reduce_scale_bf16_block(
            0,
            part_ptr,
            weights_ptr,
            T,
            stride_ps,
            stride_pm,
            H,
            weight_scale,
            N=H,
            SPLIT_K=SPLIT_K,
            BLOCK=W_BLOCK,
        )


def index_q_rope_pack_flydsl(
    q: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
    rope_dim: int,
    *,
    num_heads: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The index-Q RoPE and fp4 pack at any row count: (q_fp4 [T, H, 64] int8, q_scale
    [T, 1, 4, 16, 4] uint8), bitwise pack_fp4_query_flydsl(_rope_fq4(q.view(T, H, 128),
    freqs_cis[positions], rope_dim)) without the bf16 fake-quant round trip through memory."""
    T = q.shape[0]
    H = num_heads
    assert q.dtype == torch.bfloat16 and q.dim() == 2 and q.shape[1] == H * 128
    assert q.stride(1) == 1
    assert H % 16 == 0 and 0 < H <= 64, H
    assert freqs_cis.dtype == torch.complex64 and freqs_cis.shape[1] == rope_dim // 2
    # the plain and rotated halves are each whole 32-groups of a power-of-two width
    assert rope_dim == 64, rope_dim
    assert positions.shape == (T,)
    # the kernel reads positions[t] at pos_ptr + t
    positions = positions.contiguous()
    f_real = torch.view_as_real(freqs_cis)
    assert f_real.is_contiguous()
    q_fp4 = torch.empty((T, H, 64), dtype=torch.int8, device=q.device)
    q_scale = torch.empty((T, 1, 4, 16, 4), dtype=torch.uint8, device=q.device)
    if T == 0:
        return q_fp4, q_scale
    _index_q_rope_pack_kernel[(T, H // 16)](
        q,
        f_real,
        positions,
        q_fp4,
        q_scale,
        q.stride(0),
        f_real.shape[0],
        H=H,
        D=128,
        RD=rope_dim,
        AMAX_FLOOR=FP4_AMAX_FLOOR,
        num_warps=2,
        # bitwise parity with the standalone RoPE kernel: a contracted FMA in the rotation flips
        # the bf16 rounding of a few values in 10^8
        enable_fp_fusion=False,
    )
    return q_fp4, q_scale


def index_q_rope_pack_weights_flydsl(
    q: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
    rope_dim: int,
    head_weight_partials: torch.Tensor,
    weight_scale: float,
    *,
    num_heads: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The HIP index-Q inputs in one launch: (q_fp4 [T, H, 64] int8, q_scale [T, 1, 4, 16, 4]
    uint8, weights [T, H] bf16), bitwise pack_fp4_query_flydsl(_rope_fq4(q.view(T, H, 128),
    freqs_cis[positions], rope_dim)) and indexer_head_weights's reduce of the
    [split_k, T, H] fp32 head_weight_partials."""
    T = q.shape[0]
    H = num_heads
    assert q.dtype == torch.bfloat16 and q.dim() == 2 and q.shape[1] == H * 128
    assert q.stride(1) == 1
    assert H % 16 == 0 and 0 < H <= 64, H
    assert freqs_cis.dtype == torch.complex64 and freqs_cis.shape[1] == rope_dim // 2
    assert positions.shape == (T,)
    split_k, pt, ph = head_weight_partials.shape
    assert (pt, ph) == (T, H) and head_weight_partials.dtype == torch.float32
    f_real = torch.view_as_real(freqs_cis)
    assert f_real.is_contiguous()
    q_fp4 = torch.empty((T, H, 64), dtype=torch.int8, device=q.device)
    q_scale = torch.empty((T, 1, 4, 16, 4), dtype=torch.uint8, device=q.device)
    weights = torch.empty((T, H), dtype=torch.bfloat16, device=q.device)
    if T == 0:
        return q_fp4, q_scale, weights
    w_block = triton.next_power_of_2(T * H)
    assert w_block <= 4096, (
        f"T * H = {T * H} exceeds the 4096-wide head-weight reduce block"
    )
    _index_q_pack_weights_kernel[(T * H + 1,)](
        q,
        f_real,
        positions,
        q_fp4,
        q_scale,
        head_weight_partials,
        weights,
        q.stride(0),
        head_weight_partials.stride(0),
        head_weight_partials.stride(1),
        float(weight_scale),
        T,
        f_real.shape[0],
        H=H,
        D=128,
        RD=rope_dim,
        AMAX_FLOOR=FP4_AMAX_FLOOR,
        SPLIT_K=split_k,
        W_BLOCK=w_block,
        num_warps=4,
    )
    return q_fp4, q_scale, weights


def indexer_head_weights(
    x: torch.Tensor, weight: torch.Tensor, scale: float
) -> torch.Tensor:
    """bf16(bf16(x @ weight.T) * scale) as a contiguous bf16 [M, N], the layout the
    FlyDSL logits kernels take. x bf16 [M, K] and weight bf16 [N, K], with
    M <= rocm_gemv_split_k_max_tokens(n=N, k=K, weight_dtype=torch.bfloat16)."""
    partials = rocm_router_gemv_split_k(x, weight)
    split_k, M, N = partials.shape
    out = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
    block = triton.next_power_of_2(M * N)
    _reduce_scale_bf16_kernel[(triton.cdiv(M * N, block),)](
        partials,
        out,
        M,
        partials.stride(0),
        partials.stride(1),
        out.stride(0),
        float(scale),
        N=N,
        SPLIT_K=split_k,
        BLOCK=block,
        num_warps=4,
    )
    return out
