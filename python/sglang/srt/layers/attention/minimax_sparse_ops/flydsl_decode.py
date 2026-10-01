"""MiniMax-M3 paged decode using AITER #4332/#5546.

Main KV is written directly in SHUFFLE 5D; index KV remains NHD. Sparse
selections become page-table rows, without copying K/V. Only dense graph
decode uses a work plan: the fixed sparse top-k window uses static splits.
"""

import logging
from functools import cache

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _prefill_rows(cu_q, prefix, req, row_req, row_len, BLOCK: tl.constexpr):
    batch = tl.program_id(0)
    start = tl.load(cu_q + batch)
    end = tl.load(cu_q + batch + 1)
    offset = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = start + offset < end
    tl.store(row_req + start + offset, tl.load(req + batch), mask)
    tl.store(row_len + start + offset, tl.load(prefix + batch) + offset + 1, mask)


@triton.jit
def _sparse_page_table(
    topk,
    req_to_token,
    requests,
    lengths,
    tables,
    selected_lengths,
    stride_th: tl.constexpr,
    stride_tb: tl.constexpr,
    stride_tt: tl.constexpr,
    stride_req: tl.constexpr,
    HEADS: tl.constexpr,
    SPARSE_BLOCK: tl.constexpr,
    PAGE: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    batch, head = row // HEADS, row % HEADS
    i = tl.arange(0, BLOCK)
    pages_per_block: tl.constexpr = SPARSE_BLOCK // PAGE
    selected = tl.load(
        topk
        + head * stride_th
        + batch * stride_tb
        + (i // pages_per_block) * stride_tt,
        i < WIDTH,
        other=-1,
    )
    length = tl.load(lengths + batch)
    pos = selected * SPARSE_BLOCK + (i % pages_per_block) * PAGE
    valid = (i < WIDTH) & (selected >= 0) & (pos < length)
    request = tl.load(requests + batch).to(tl.int64)
    slot = tl.load(
        req_to_token + request * stride_req + pos,
        valid,
        other=0,
    ).to(tl.int64)
    # Move a partial page to the end regardless of the indexer's score order.
    # pa_decode masks only the last page using selected_lengths.
    full = valid & (pos + PAGE <= length)
    full_rank = tl.cumsum(full.to(tl.int32)) - 1
    partial_rank = tl.sum(full.to(tl.int32), 0)
    dest = tl.where(full, full_rank, partial_rank)
    physical_page = slot // PAGE * HEADS + head
    tl.store(tables + row * WIDTH + dest, physical_page, valid)
    n_tokens = tl.sum(tl.where(valid, tl.minimum(PAGE, length - pos), 0), 0)
    tl.store(selected_lengths + row, n_tokens)


@cache
def load_flydsl():
    # Explicit opt-in must fail clearly if the image lacks the kernel/compiler.
    try:
        from aiter.ops.flydsl.pa_decode import (
            get_recommended_splits,
            pa_decode,
            plan_pa_decode,
        )
    except ImportError as exc:
        raise RuntimeError(
            "MiniMax FlyDSL decode requires AITER #4332/#5546 and FlyDSL. "
            "Use the pinned candidate stack or disable SGLANG_MINIMAX_FLYDSL_DECODE."
        ) from exc
    logger.info("MiniMax: loaded AITER FlyDSL paged attention and GPU work planner")
    return pa_decode, plan_pa_decode, get_recommended_splits


class DenseDecodePlanner:
    """Dense lengths and optional plans owned by this backend instance.

    Mint plans before capture; record refresh inside the graph so it reads the
    current padded lengths on every replay. Eager batches use static partitions
    and cannot grow the persistent plan cache.
    """

    def __init__(self, num_kv_heads, *, enable_plan=True):
        self.num_kv_heads = num_kv_heads
        self.enable_plan = enable_plan
        self.plans = {}
        self.length_buffers = {}
        self.active_plan = None
        self.active_lengths = None
        _, self.plan, _ = load_flydsl()

    def prepare(self, forward_batch, *, in_capture=False):
        self.active_plan = None
        self.active_lengths = None
        if not forward_batch.forward_mode.is_decode_or_idle():
            return
        lengths = forward_batch.seq_lens
        bs = forward_batch.batch_size
        if bs == 0:
            return
        if lengths.dtype not in (torch.int32, torch.int64) or lengths.shape != (bs,):
            raise ValueError("MiniMax FlyDSL requires padded integer seq_lens")
        key = (bs, lengths.device)
        if in_capture and key not in self.length_buffers:
            self.length_buffers[key] = torch.empty_like(lengths, dtype=torch.int32)
        self.active_lengths = self.length_buffers.get(key)
        if self.active_lengths is None:
            self.active_lengths = lengths.to(torch.int32)
            return
        if in_capture:
            self.active_lengths.copy_(lengths)
        if self.enable_plan and bs <= 4096:
            if in_capture and key not in self.plans:
                self.plans[key] = self.plan(self.active_lengths, self.num_kv_heads)
            self.active_plan = self.plans.get(key)

    def prepare_eager(self, forward_batch):
        self.active_plan = None
        self.active_lengths = (
            forward_batch.seq_lens.to(torch.int32)
            if forward_batch.forward_mode.is_decode_or_idle()
            else None
        )

    def refresh(self, forward_batch):
        if self.active_lengths is not None:
            # SGLang graph inputs use int64. Capture the conversion once per
            # forward, so every dense layer reads the live int32 lengths.
            self.active_lengths.copy_(forward_batch.seq_lens)
        if self.active_plan is not None:
            self.plan(
                self.active_lengths,
                self.num_kv_heads,
                plan=self.active_plan,
            )


def decode(
    output,
    query,
    k_cache,
    v_cache,
    lengths,
    tables,
    scale,
    k_scale=None,
    v_scale=None,
    work_plan=None,
):
    kernel, _, splits = load_flydsl()
    heads = k_cache.shape[1]
    bs = lengths.numel()
    if bs == 0:
        return
    if lengths.dtype not in (torch.int32, torch.int64):
        raise TypeError("MiniMax FlyDSL requires integer context lengths")
    lengths = lengths.to(torch.int32)
    partitions = (
        work_plan.max_partitions if work_plan is not None else splits(bs, heads)
    )
    # AITER allocates correctly shaped scratch for either static or packed
    # plans. Graph capture owns these allocations; no process-wide scratch
    # cache can alias concurrent backends/streams.
    kernel(
        output=output,
        query=query,
        key_cache=k_cache,
        value_cache=v_cache,
        context_lengths=lengths,
        block_tables=tables,
        softmax_scale=scale,
        query_length=1,
        max_context_partition_num=partitions,
        compute_type=k_cache.dtype,
        key_scale=k_scale,
        value_scale=v_scale,
        work_plan=work_plan,
    )


def sparse_decode(
    q,
    k_cache,
    v_cache,
    topk_idx,
    req_to_token,
    requests,
    lengths,
    block_size,
    sm_scale,
    k_scale,
    v_scale,
):
    batch, q_heads, dim = q.shape
    heads, page = k_cache.shape[1], k_cache.shape[3]
    if block_size % page:
        raise ValueError("MiniMax sparse block must contain whole KV pages")
    if q_heads % heads or topk_idx.shape[:2] != (heads, batch):
        raise ValueError("MiniMax FlyDSL query/top-k head layout mismatch")
    output = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    if batch == 0:
        return output
    width = topk_idx.shape[-1] * (block_size // page)
    tables = torch.empty((batch * heads, width), dtype=torch.int32, device=q.device)
    selected_lengths = torch.empty((batch * heads,), dtype=torch.int32, device=q.device)
    _sparse_page_table[(batch * heads,)](
        topk_idx,
        req_to_token,
        requests,
        lengths,
        tables,
        selected_lengths,
        *topk_idx.stride(),
        req_to_token.stride(0),
        heads,
        block_size,
        page,
        width,
        triton.next_power_of_2(width),
    )
    # Folding the physical page and head axes is a zero-copy view. Each
    # (request, KV head) has its own selection and a single KV head in AITER.
    k_view = k_cache.view(-1, 1, dim // 16, page, 16)
    v_view = v_cache.view(-1, 1, page // 16, dim, 16)
    decode(
        output.view(batch * heads, q_heads // heads, dim),
        q.reshape(batch * heads, q_heads // heads, dim),
        k_view,
        v_view,
        selected_lengths,
        tables,
        dim**-0.5 if sm_scale is None else sm_scale,
        k_scale,
        v_scale,
    )
    return output


def sparse_prefill(
    q,
    k_cache,
    v_cache,
    topk_idx,
    req_to_token,
    requests,
    cu_q,
    prefix,
    max_q,
    block_size,
    sm_scale,
    k_scale,
    v_scale,
):
    """Run sparse prefill as independent causal rows.

    Bound transient page-table/partial-output memory by processing 1024 query
    rows at a time. Only Q/output are sliced; the physical KV cache is shared.
    """
    rows = q.shape[0]
    row_requests = torch.empty(rows, dtype=requests.dtype, device=q.device)
    row_lengths = torch.empty(rows, dtype=torch.int32, device=q.device)
    output = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    if rows == 0:
        return output
    _prefill_rows[(requests.numel(), triton.cdiv(max_q, 256))](
        cu_q,
        prefix,
        requests,
        row_requests,
        row_lengths,
        256,
    )
    for start in range(0, rows, 1024):
        end = min(start + 1024, rows)
        output[start:end] = sparse_decode(
            q[start:end],
            k_cache,
            v_cache,
            topk_idx[:, start:end],
            req_to_token,
            row_requests[start:end],
            row_lengths[start:end],
            block_size,
            sm_scale,
            k_scale,
            v_scale,
        )
    return output
