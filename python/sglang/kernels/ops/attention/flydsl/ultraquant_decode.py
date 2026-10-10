# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""High-level API for the FlyDSL UltraQuant 4-bit decode specialization.

Replaces the Triton UltraQuant decode on gfx950 for head_dim 256 and GQA
6/8/16. The kernel writes per-split partials into the stock ``attn_logits``/
``attn_lse`` buffers and ``ultraquant_decode_reduce`` merges them.

The kernel rotates the query in-register, so it takes the RAW query.
"""

from __future__ import annotations

import functools
import importlib.util

import torch
import triton
import triton.language as tl

from sglang.srt.layers.quantization.ultraquant_tensor import code_bytes, n_groups
from sglang.srt.utils import is_gfx95_supported, next_power_of_2

_HEAD_SIZE = 256
_SUPPORTED_GQA = (6, 8, 16)
# The kernel addresses every buffer through a 32-bit byte offset.
_MAX_BUFFER_BYTES = 1 << 32

# Partition p walks the KV in blocks p, p + P, p + 2P, ... of block_kv tokens.
_BLOCK_KV = 256
# Chunked launches cut the batch into whole blocks; short ones keep each chunk
# close to its even share of the tokens.
_SMALL_BLOCK_KV = 128

# Past about this many workgroups per CU, extra splits no longer hide memory
# latency and only add reducer work.
_WORKGROUPS_PER_CU = 8
# Bounds the fp32 split buffers kept for the largest graph batch.
_MAX_KV_SPLITS = 256
# Each of the 64 lanes scans up to 8 sequences to find a chunk's sequence.
_MAX_CHUNKED_SEQS = 64 * 8
# Elements of [chunks, head dim] one reducer wave loads per step.
_REDUCE_TILE = 4096


def ultraquant_decode_max_kv_splits(
    base_max_kv_splits: int, max_context_len: int
) -> int:
    """Split-buffer width: one partition per block at the max context."""
    blocks = -(-max(max_context_len, 1) // _SMALL_BLOCK_KV)
    return max(base_max_kv_splits, min(_MAX_KV_SPLITS, next_power_of_2(blocks)))


@functools.cache
def ultraquant_decode_launch_config(
    batch_size: int,
    num_kv_heads: int,
    min_kv_splits: int,
    max_kv_splits: int,
    core_count: int,
    split_rows: int,
) -> tuple[int | None, int, int]:
    """``(num_splits, block_kv, work_budget)`` for one launch.

    The batch's tokens are cut into about ``work_budget - batch_size`` equal
    chunks per KV head, one workgroup each, whatever the mix of context
    lengths. ``split_rows`` is how many ``[num_q_heads, head_dim]`` partials
    the split buffers hold. Otherwise each sequence gets ``num_splits`` strided
    partitions (``work_budget`` 0), which batch-invariant inference pins.
    """
    # A pinned split count (batch-invariant inference) pins the block size too.
    if core_count <= 0 or min_kv_splits >= max_kv_splits:
        return max_kv_splits, _BLOCK_KV, 0
    target = _WORKGROUPS_PER_CU * core_count
    budget = target // num_kv_heads
    if (
        batch_size <= _MAX_CHUNKED_SEQS
        and batch_size < budget
        and batch_size + budget <= split_rows
    ):
        return None, _SMALL_BLOCK_KV, budget
    splits = next_power_of_2(-(-target // (batch_size * num_kv_heads)))
    return max(min_kv_splits, min(max_kv_splits, splits)), _BLOCK_KV, 0


@functools.cache
def _kernel():
    from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

    from .kernels import ultraquant_decode_hd256 as kmod

    return kmod.create_ultraquant_decode_hd256_kernel, _run_compiled


def is_flydsl_ultraquant_decode_supported(
    head_dim: int, query_group_size: int, dtype: torch.dtype
) -> bool:
    """Return whether this gfx950-only specialization covers the given shape."""
    return (
        head_dim == _HEAD_SIZE
        and query_group_size in _SUPPORTED_GQA
        and dtype == torch.bfloat16
        and is_gfx95_supported()
        and importlib.util.find_spec("flydsl") is not None
    )


def flydsl_ultraquant_decode_fits(
    num_seqs: int, num_q_heads: int, split_stride: int
) -> bool:
    """Whether ``attn_logits`` for this batch fits the kernel's 32-bit offsets."""
    return num_seqs * num_q_heads * split_stride * _HEAD_SIZE * 4 <= _MAX_BUFFER_BYTES


def _check_tensor(
    name: str,
    tensor: torch.Tensor,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    # Strides are baked into the kernel, so the layout must be dense.
    if (
        tensor.shape != shape
        or tensor.dtype != dtype
        or tensor.device != device
        or not tensor.is_contiguous()
    ):
        raise ValueError(
            f"`{name}` must be a contiguous {dtype} tensor of shape {list(shape)} "
            f"on {device}, got {tensor.dtype} {list(tensor.shape)} with strides "
            f"{tensor.stride()} on {tensor.device}."
        )


def _chunk_views(
    attn_logits: torch.Tensor, attn_lse: torch.Tensor, rows: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """``[rows, Hq, 1, D]`` and ``[rows, Hq, 1]`` views of the split buffers' storage."""
    num_q_heads, head_dim = attn_logits.shape[1], attn_logits.shape[-1]
    if (
        not attn_logits.is_contiguous()
        or not attn_lse.is_contiguous()
        or attn_logits.numel() < rows * num_q_heads * head_dim
        or attn_lse.numel() < rows * num_q_heads
    ):
        raise ValueError(
            f"The split buffers must be contiguous and hold {rows} chunk partials."
        )
    return (
        attn_logits.view(-1)[: rows * num_q_heads * head_dim].view(
            rows, num_q_heads, 1, head_dim
        ),
        attn_lse.view(-1)[: rows * num_q_heads].view(rows, num_q_heads, 1),
    )


def _check_split_buffer(
    name: str,
    tensor: torch.Tensor,
    num_seqs: int,
    trailing: tuple[int, ...],
    device: torch.device,
) -> None:
    """Check a split-K output whose batch dim may be padded past ``num_seqs``."""
    if tensor.shape[0] < num_seqs:
        raise ValueError(
            f"`{name}` covers {tensor.shape[0]} sequences, need {num_seqs}."
        )
    _check_tensor(name, tensor, (tensor.shape[0], *trailing), torch.float32, device)


def flydsl_ultraquant_decode(
    query: torch.Tensor,
    k_code_buffer: torch.Tensor,
    k_scale_buffer: torch.Tensor,
    v_code_buffer: torch.Tensor,
    v_scale_buffer: torch.Tensor,
    attn_logits: torch.Tensor,
    attn_lse: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    softmax_scale: float,
    num_splits: int | None,
    block_kv: int = _BLOCK_KV,
    work_budget: int = 0,
) -> None:
    """Run stage 1 of the UltraQuant decode, filling ``attn_logits``/``attn_lse``.

    ``query`` is the raw (unrotated) bf16 ``[num_seqs, num_q_heads, 256]``
    tensor; check ``is_flydsl_ultraquant_decode_supported`` and
    ``flydsl_ultraquant_decode_fits`` first. With ``work_budget`` 0 each
    sequence writes ``num_splits`` partitions. Otherwise the batch is cut into
    equal chunks whose partials fill the first ``num_seqs + work_budget`` rows
    of the buffers' storage, and ``num_splits`` must be None. Merge with
    ``ultraquant_decode_reduce`` and the same ``num_splits``, ``block_kv`` and
    ``work_budget``.
    """
    device = query.device
    if query.dim() != 3:
        raise ValueError(f"`query` must have rank 3, got rank {query.dim()}.")
    num_seqs, num_q_heads, head_dim = query.shape
    num_kv_heads = k_code_buffer.shape[1]
    if num_q_heads % num_kv_heads:
        raise ValueError(
            f"`query` head count {num_q_heads} is not a multiple of the "
            f"{num_kv_heads} KV heads."
        )
    query_group_size = num_q_heads // num_kv_heads
    # The two outer query strides are baked in; the head dim must be dense.
    if query.stride(2) != 1:
        raise ValueError("`query` must be contiguous along the head dim.")

    if block_kv not in (_SMALL_BLOCK_KV, _BLOCK_KV):
        raise ValueError(
            f"`block_kv` must be {_SMALL_BLOCK_KV} or {_BLOCK_KV}, got {block_kv}."
        )
    if work_budget < 0:
        raise ValueError(f"`work_budget` must be >= 0, got {work_budget}.")
    rows = num_seqs
    if work_budget:
        if num_splits is not None:
            raise ValueError("Pass `num_splits` or `work_budget`, not both.")
        if not 0 < num_seqs <= min(_MAX_CHUNKED_SEQS, work_budget - 1):
            raise ValueError(
                f"Chunked decode takes 1 to min({_MAX_CHUNKED_SEQS}, work_budget - "
                f"1) sequences, got {num_seqs} with work_budget {work_budget}."
            )
        rows = num_seqs + work_budget
        attn_logits, attn_lse = _chunk_views(attn_logits, attn_lse, rows)
        num_splits = 1
    split_stride = attn_logits.shape[2]
    if not 0 < num_splits <= split_stride:
        raise ValueError(
            f"`num_splits` must be in [1, {split_stride}], got {num_splits}."
        )
    _check_split_buffer(
        "attn_logits",
        attn_logits,
        rows,
        (num_q_heads, split_stride, head_dim),
        device,
    )
    _check_split_buffer("attn_lse", attn_lse, rows, (num_q_heads, split_stride), device)
    if (
        kv_indptr.numel() < num_seqs + 1
        or kv_indptr.dtype != torch.int32
        or kv_indptr.device != device
        or kv_indptr.stride(-1) != 1
    ):
        raise ValueError(
            "`kv_indptr` must be a contiguous int32 tensor on the query device "
            f"with at least {num_seqs + 1} entries, got {list(kv_indptr.shape)} "
            f"{kv_indptr.dtype}."
        )
    # The kernel reads the low dword of each entry, so the width is load-bearing.
    if (
        kv_indices.dtype != torch.int64
        or kv_indices.device != device
        or kv_indices.stride(-1) != 1
    ):
        raise ValueError(
            "`kv_indices` must be a contiguous int64 tensor on the query device."
        )

    num_slots = k_code_buffer.shape[0]
    for name, buf, row in (
        ("k_code_buffer", k_code_buffer, code_bytes(head_dim)),
        ("v_code_buffer", v_code_buffer, code_bytes(head_dim)),
        ("k_scale_buffer", k_scale_buffer, n_groups(head_dim)),
        ("v_scale_buffer", v_scale_buffer, n_groups(head_dim)),
    ):
        _check_tensor(name, buf, (num_slots, num_kv_heads, row), torch.uint8, device)

    create_kernel, run_compiled = _kernel()
    launch = create_kernel(
        num_kv_heads=num_kv_heads,
        num_partitions=num_splits,
        block_kv=block_kv,
        softmax_scale=float(softmax_scale),
        query_group_size=query_group_size,
        stride_q_seq=query.stride(0),
        stride_q_head=query.stride(1),
        split_stride=split_stride,
        work_budget=work_budget,
        seqs_per_lane=next_power_of_2(-(-num_seqs // 64)) if work_budget else 1,
    )
    with torch.cuda.device(device):
        run_compiled(
            launch,
            attn_logits,
            attn_lse,
            query,
            k_code_buffer,
            k_scale_buffer,
            v_code_buffer,
            v_scale_buffer,
            kv_indptr,
            kv_indices,
            rows,
            torch.cuda.current_stream(device),
        )


@triton.jit
def _reduce_strided_splits_kernel(
    Mid_O,
    Mid_Lse,
    O,
    kv_indptr,
    stride_mid_ob: tl.int64,
    stride_mid_oh: tl.int64,
    stride_mid_os: tl.int64,
    stride_lse_b: tl.int64,
    stride_lse_h: tl.int64,
    stride_ob: tl.int64,
    stride_oh: tl.int64,
    NUM_SPLITS: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    cur_batch = tl.program_id(0).to(tl.int64)
    cur_head = tl.program_id(1)
    offs_d = tl.program_id(2) * BLOCK_D + tl.arange(0, BLOCK_D)

    seq_len = tl.load(kv_indptr + cur_batch + 1) - tl.load(kv_indptr + cur_batch)
    offs_s = tl.arange(0, BLOCK_SPLITS)
    # Split s holds blocks s, s + NUM_SPLITS, ..., so it has data iff block s exists.
    has_kv = (offs_s < NUM_SPLITS) & (offs_s * BLOCK_KV < seq_len)

    lse = tl.load(
        Mid_Lse + cur_batch * stride_lse_b + cur_head * stride_lse_h + offs_s,
        mask=has_kv,
        other=-float("inf"),
    )
    lse_max = tl.max(lse, axis=0)
    weight = tl.where(has_kv, tl.exp(lse - lse_max), 0.0)
    weight_sum = tl.sum(weight, axis=0)

    partial = tl.load(
        Mid_O
        + cur_batch * stride_mid_ob
        + cur_head * stride_mid_oh
        + offs_s[:, None] * stride_mid_os
        + offs_d[None, :],
        mask=has_kv[:, None],
        other=0.0,
    )
    acc = tl.sum(partial * weight[:, None], axis=0)
    acc = tl.where(weight_sum > 0.0, acc / weight_sum, 0.0)
    tl.store(O + cur_batch * stride_ob + cur_head * stride_oh + offs_d, acc)


@triton.jit
def _reduce_chunks_kernel(
    Mid_O,
    Mid_Lse,
    O,
    kv_indptr,
    stride_mid_oc: tl.int64,
    stride_mid_oh: tl.int64,
    stride_lse_c: tl.int64,
    stride_lse_h: tl.int64,
    stride_ob: tl.int64,
    stride_oh: tl.int64,
    WORK_BUDGET: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    offs_d = tl.program_id(2) * BLOCK_D + tl.arange(0, BLOCK_D)

    # The chunking the decode kernel derived from the same kv_indptr.
    kv_start = tl.load(kv_indptr)
    batch_tokens = tl.load(kv_indptr + tl.num_programs(0)) - kv_start
    budget_tokens = (WORK_BUDGET - tl.num_programs(0)) * BLOCK_KV
    chunk_tokens = tl.cdiv(batch_tokens, budget_tokens) * BLOCK_KV
    chunk_tokens = tl.maximum(chunk_tokens, BLOCK_KV)
    seq_start = tl.load(kv_indptr + cur_batch) - kv_start
    seq_len = tl.load(kv_indptr + cur_batch + 1) - kv_start - seq_start
    first = seq_start // chunk_tokens + cur_batch
    num_chunks = tl.maximum(tl.cdiv(seq_len, chunk_tokens), 1)

    run_max = tl.full([], -float("inf"), tl.float32)
    weight_sum = tl.full([], 0.0, tl.float32)
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    for c0 in range(0, num_chunks, BLOCK_SPLITS):
        offs_c = c0 + tl.arange(0, BLOCK_SPLITS)
        valid = offs_c < num_chunks
        rows = (first + offs_c).to(tl.int64)
        lse = tl.load(
            Mid_Lse + rows * stride_lse_c + cur_head * stride_lse_h,
            mask=valid,
            other=-float("inf"),
        )
        new_max = tl.maximum(run_max, tl.max(lse, axis=0))
        # An empty chunk has lse -inf; keep the exponents finite until one has data.
        ref = tl.where(new_max > -float("inf"), new_max, 0.0)
        rescale = tl.exp(run_max - ref)
        weight = tl.exp(lse - ref)
        partial = tl.load(
            Mid_O
            + rows[:, None] * stride_mid_oc
            + cur_head * stride_mid_oh
            + offs_d[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        acc = acc * rescale + tl.sum(partial * weight[:, None], axis=0)
        weight_sum = weight_sum * rescale + tl.sum(weight, axis=0)
        run_max = new_max
    acc = tl.where(weight_sum > 0.0, acc / weight_sum, 0.0)
    tl.store(O + cur_batch * stride_ob + cur_head * stride_oh + offs_d, acc)


def ultraquant_decode_reduce(
    attn_logits: torch.Tensor,
    attn_lse: torch.Tensor,
    kv_indptr: torch.Tensor,
    out: torch.Tensor,
    num_splits: int | None,
    block_kv: int = _BLOCK_KV,
    work_budget: int = 0,
) -> None:
    """Merge the partials of ``flydsl_ultraquant_decode`` into ``out``.

    ``out`` is ``[num_seqs, num_q_heads, 256]``. The head dim is spread across
    programs, so a small batch still fills the GPU instead of walking the
    splits one at a time.
    """
    num_seqs, num_q_heads, head_dim = out.shape
    if out.stride(2) != 1:
        raise ValueError("`out` must be contiguous along the head dim.")
    if work_budget:
        attn_logits, attn_lse = _chunk_views(
            attn_logits, attn_lse, num_seqs + work_budget
        )
        # One wave per program; a small batch has many chunks per sequence, so
        # it takes a narrower head-dim slice and more programs.
        block_d = 16 if num_seqs <= 2 else 32 if num_seqs <= 32 else 64
        _reduce_chunks_kernel[(num_seqs, num_q_heads, head_dim // block_d)](
            attn_logits,
            attn_lse,
            out,
            kv_indptr,
            attn_logits.stride(0),
            attn_logits.stride(1),
            attn_lse.stride(0),
            attn_lse.stride(1),
            out.stride(0),
            out.stride(1),
            WORK_BUDGET=work_budget,
            BLOCK_KV=block_kv,
            BLOCK_SPLITS=_REDUCE_TILE // block_d,
            BLOCK_D=block_d,
            num_warps=1,
        )
        return
    block_d = 64
    _reduce_strided_splits_kernel[(num_seqs, num_q_heads, head_dim // block_d)](
        attn_logits,
        attn_lse,
        out,
        kv_indptr,
        attn_logits.stride(0),
        attn_logits.stride(1),
        attn_logits.stride(2),
        attn_lse.stride(0),
        attn_lse.stride(1),
        out.stride(0),
        out.stride(1),
        NUM_SPLITS=num_splits,
        BLOCK_SPLITS=next_power_of_2(num_splits),
        BLOCK_KV=block_kv,
        BLOCK_D=block_d,
        num_warps=4,
    )
