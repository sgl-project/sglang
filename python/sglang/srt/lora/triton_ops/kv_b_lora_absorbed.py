"""Triton kernels for absorbed-MLA ``kv_b_proj`` LoRA correction.

The absorbed-MLA path bypasses ``kv_b_proj.forward()`` and folds the K/V
sides as plain BMMs ``q_nope @ w_kc`` and ``attn_output @ w_vc``.  When a
LoRA adapter is active on ``kv_b_proj`` we add the LoRA delta to
``q_nope_out`` / ``attn_bmm_output`` manually.

Using the standard LoRA factored math we *never* materialize ``B @ A``:

    q_correction = q_nope     @ B_kc @ A   * scaling          # K-side
    v_correction = attn_output @ A.T @ B_vc.T * scaling       # V-side

where ``A: (slot, rank, kv_lora_rank)`` is the LoRA-A of ``kv_b_proj``
(shared across heads) and ``B: (slot, num_heads*(qk_nope+v_head_dim), rank)``
is the LoRA-B; ``B_kc`` / ``B_vc`` are its K-half / V-half slices.

Four kernels split the math along the factorization boundary, all using
the SGMM idiom from ``sgemm_lora_a`` / ``qkv_lora_b`` and the segment-indptr
routing used by ``chunked_sgmv_*``:

  * ``step_a_q_fwd``: per-head per-slot SGMM, ``(S,H,qk_nope) -> (S,H,rank)``
  * ``step_b_q_fwd``: shared-A per-slot SGMM, scaled+accumulated,
    ``(S,H,rank) -> (S,H,kv_lora_rank)``
  * ``step_a_v_fwd``: shared-A.T per-slot SGMM, ``(S,H,kv_lora_rank) -> (S,H,rank)``
  * ``step_b_v_fwd``: per-head per-slot SGMM with V-half of B, transposed,
    scaled+accumulated, ``(S,H,rank) -> (S,H,v_head_dim)``

Grid axes for each kernel:
  axis 0 : output tile in (S, N)         -- tile_id = pid_s * num_pid_n + pid_n
  axis 1 : head_id                       -- per-head weight slice
  axis 2 : batch_id (segment / request)  -- per-slot weight routing via weight_indices

Per-segment routing: each program derives its segment length from
``seg_indptr[segment_id + 1] - seg_indptr[segment_id]``, loads
``weight_indices[segment_id]`` once, and uses that slot's slice of the LoRA
weight stack.  When ``permutation`` is present, rows are routed through it,
matching the csgmv backend's adapter-grouped chunks.  No Python loops over slots
or heads.

The math also stays in the input dtype (no fp32 round-trip) -- the
contraction dim ``rank`` is small (typically 16-64), so bf16 accumulation
over it is acceptable.  ``tl.dot`` itself uses fp32 accumulation internally.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.srt.lora.triton_ops.kernel_utils import _resolve_token_positions
from sglang.srt.lora.utils import LoRABatchInfo

# ---------------------------------------------------------------------------
# Block sizes -- chosen per-kernel from the natural shape of each step.
#
# The factored math gives the four kernels these contraction (K) and output
# (N) ranges (for Kimi-K2.5: rank=16-32, qk_nope=v_head_dim=128, kv_lora_rank=512):
#
#                              K (contraction)      N (output)
#   step_a_q                   qk_nope (~128)       rank (~16-32)
#   step_b_q                   rank (~16-32)        kv_lora_rank (~512)
#   step_a_v                   kv_lora_rank (~512)  rank (~16-32)
#   step_b_v                   rank (~16-32)        v_head_dim (~128)
#
# So the "step_a_*" kernels want a large BLOCK_K (to keep loop iters small)
# and a small BLOCK_N (matched to rank to avoid wasted tile lanes), while
# the "step_b_*" kernels are the inverse.  Kernels aren't autotuned -- the
# decode-shape workload is too small to benefit and the sweep surface is
# wide.
# ---------------------------------------------------------------------------

_BLOCK_S = 16
_STEP_A_BLOCK_K = 64  # contraction over qk_nope (~128) or kv_lora_rank (~512)
_STEP_A_BLOCK_N = 16  # output is rank
_STEP_B_BLOCK_K = 16  # contraction is rank
_STEP_B_BLOCK_N = 64  # output is kv_lora_rank (~512) or v_head_dim (~128)


def _num_segments(batch_info: LoRABatchInfo) -> int:
    return batch_info.num_segments or batch_info.bs


def _max_segment_len(batch_info: LoRABatchInfo) -> int:
    if batch_info.max_len is not None:
        return batch_info.max_len
    if batch_info.seg_lens is not None:
        return int(batch_info.seg_lens.max().item())
    raise ValueError("LoRA batch_info must provide max_len or seg_lens.")


def _segment_grid_size(batch_info: LoRABatchInfo, num_segments: int) -> int:
    return batch_info.bs if batch_info.use_cuda_graph else num_segments


# ---------------------------------------------------------------------------
# Kernel 1 -- Step A_q: per-head per-slot SGMM, reads K-half of B
#
#     q_lora_a[t, h, r] = sum_{i<qk_nope} q_nope[t, h, i] * B[slot, h*FULL_K + i, r]
#
# x      : (S, H, qk_nope)
# w (B)  : (num_lora, H*FULL_K, rank)   -- FULL_K = qk_nope + v_head_dim
# out    : (S, H, rank)                 -- fresh allocation, no accumulate
# ---------------------------------------------------------------------------


@triton.jit(do_not_specialize=["num_segments"])
def _step_a_q_kernel(
    x,
    w,
    out,
    # dims
    S,
    H_FULL_K,  # H * (qk_nope + v_head_dim), the row-stride landmark
    K,  # qk_nope (contraction)
    N,  # rank (output)
    # strides
    x_stride_s,
    x_stride_h,
    x_stride_k,
    w_stride_l,
    w_stride_n,
    w_stride_k,
    out_stride_s,
    out_stride_h,
    out_stride_n,
    # batch info
    seg_indptr,
    weight_indices,
    lora_ranks,
    sorted_token_ids,
    num_segments,
    # meta
    FULL_K: tl.constexpr,  # per-head row stride in B (qk_nope + v_head_dim)
    SORTED_BY_ADAPTER: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    batch_id = tl.program_id(axis=2)
    head_id = tl.program_id(axis=1)
    pid = tl.program_id(axis=0)

    if batch_id >= num_segments:
        return

    w_index = tl.load(weight_indices + batch_id)
    cur_rank = tl.load(lora_ranks + w_index)
    if cur_rank == 0:
        return

    seg_start = tl.load(seg_indptr + batch_id)
    seg_end = tl.load(seg_indptr + batch_id + 1)
    seg_len = seg_end - seg_start
    if seg_len == 0:
        return

    # Truncate output N to this slot's rank (allows mixed-rank batches).
    N_eff = tl.minimum(N, cur_rank)

    num_pid_n = tl.cdiv(N_eff, BLOCK_N)
    pid_s = pid // num_pid_n
    pid_n = pid % num_pid_n
    if pid_s * BLOCK_S >= seg_len:
        return

    s_offset = tl.arange(0, BLOCK_S) + pid_s * BLOCK_S
    n_offset = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    k_offset = tl.arange(0, BLOCK_K)

    s_physical = _resolve_token_positions(
        sorted_token_ids, seg_start, s_offset, seg_len, SORTED_BY_ADAPTER
    )

    # Clamp masked-lane indices into the valid range so pointer arithmetic
    # stays in-bounds even before the load mask drops the values.
    row_mask = s_offset < seg_len
    safe_row = tl.minimum(s_physical, S - 1)
    safe_n = tl.minimum(n_offset, N_eff - 1)

    head_row_base = (
        head_id * FULL_K
    )  # row offset for this head's K-half (i in [0, qk_nope))

    partial_sum = tl.zeros((BLOCK_S, BLOCK_N), dtype=tl.float32)
    for k_block in range(0, tl.cdiv(K, BLOCK_K)):
        cur_k = k_block * BLOCK_K + k_offset
        k_mask = cur_k < K
        safe_k = tl.minimum(cur_k, K - 1)

        # x[s, h, k]
        x_tile = tl.load(
            x
            + safe_row[:, None] * x_stride_s
            + head_id * x_stride_h
            + safe_k[None, :] * x_stride_k,
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0,
        )

        # B[slot, h*FULL_K + i, r]: row dim of B carries i (= GEMM K),
        # column dim carries r (= GEMM N).
        w_tile = tl.load(
            w
            + w_index * w_stride_l
            + (head_row_base + safe_k[:, None]) * w_stride_n
            + safe_n[None, :] * w_stride_k,
            mask=k_mask[:, None] & (n_offset[None, :] < N_eff),
            other=0.0,
        )

        partial_sum += tl.dot(x_tile, w_tile)

    partial_sum = partial_sum.to(x.dtype.element_ty)
    out_offs = (
        safe_row[:, None] * out_stride_s
        + head_id * out_stride_h
        + safe_n[None, :] * out_stride_n
    )
    out_mask = row_mask[:, None] & (n_offset[None, :] < N_eff)
    tl.store(out + out_offs, partial_sum, mask=out_mask)


def step_a_q_fwd(
    q_nope: torch.Tensor,
    B_buf: torch.Tensor,
    batch_info: LoRABatchInfo,
    full_K_per_head: int,
) -> torch.Tensor:
    """Step A of the q-side correction.

    Args:
        q_nope: ``(S, H, qk_nope)``, the absorbed-MLA q intermediate.
        B_buf: ``(num_lora, H*full_K_per_head, rank)`` from the LoRA pool.
        batch_info: standard ``LoRABatchInfo``.
        full_K_per_head: ``qk_nope + v_head_dim``, the row stride per head in B.

    Returns:
        ``(S, H, rank)`` -- per-token, per-head low-rank intermediate, ready for step B_q.
    """
    S, H, qk_nope_dim = q_nope.shape
    rank = B_buf.shape[-1]
    out = torch.empty((S, H, rank), device=q_nope.device, dtype=q_nope.dtype)
    num_segments = _num_segments(batch_info)
    max_segment_len = _max_segment_len(batch_info)
    segment_grid = _segment_grid_size(batch_info, num_segments)

    grid = (
        triton.cdiv(max_segment_len, _BLOCK_S) * triton.cdiv(rank, _STEP_A_BLOCK_N),
        H,
        segment_grid,
    )
    sorted_by_adapter = batch_info.permutation is not None

    _step_a_q_kernel[grid](
        q_nope,
        B_buf,
        out,
        S,
        H * full_K_per_head,
        qk_nope_dim,
        rank,
        q_nope.stride(0),
        q_nope.stride(1),
        q_nope.stride(2),
        B_buf.stride(0),
        B_buf.stride(1),
        B_buf.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        batch_info.seg_indptr,
        batch_info.weight_indices,
        batch_info.lora_ranks,
        batch_info.permutation,
        num_segments,
        FULL_K=full_K_per_head,
        SORTED_BY_ADAPTER=sorted_by_adapter,
        BLOCK_S=_BLOCK_S,
        BLOCK_N=_STEP_A_BLOCK_N,
        BLOCK_K=_STEP_A_BLOCK_K,
    )
    return out


# ---------------------------------------------------------------------------
# Kernel 2 -- Step B_q: shared-A per-slot SGMM, scaled + accumulated
#
#     base[t, h, k] += sum_r x[t, h, r] * A[slot, r, k] * scaling
#
# x      : (S, H, rank)
# w (A)  : (num_lora, rank, kv_lora_rank)
# base   : (S, H, kv_lora_rank), updated in-place (accumulated)
# ---------------------------------------------------------------------------


@triton.jit(do_not_specialize=["num_segments"])
def _step_b_q_kernel(
    x,
    w,
    base,
    # dims
    S,
    K,  # rank (contraction)
    N,  # kv_lora_rank (output)
    # strides
    x_stride_s,
    x_stride_h,
    x_stride_k,
    w_stride_l,
    w_stride_k,
    w_stride_n,
    b_stride_s,
    b_stride_h,
    b_stride_n,
    # batch info
    seg_indptr,
    weight_indices,
    lora_ranks,
    sorted_token_ids,
    scalings,
    num_segments,
    # meta
    SORTED_BY_ADAPTER: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    batch_id = tl.program_id(axis=2)
    head_id = tl.program_id(axis=1)
    pid = tl.program_id(axis=0)

    if batch_id >= num_segments:
        return

    w_index = tl.load(weight_indices + batch_id)
    cur_rank = tl.load(lora_ranks + w_index)
    if cur_rank == 0:
        return

    seg_start = tl.load(seg_indptr + batch_id)
    seg_end = tl.load(seg_indptr + batch_id + 1)
    seg_len = seg_end - seg_start
    if seg_len == 0:
        return
    scaling = tl.load(scalings + w_index)

    # Truncate contraction K to this slot's rank.
    K_eff = tl.minimum(K, cur_rank)

    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_s = pid // num_pid_n
    pid_n = pid % num_pid_n
    if pid_s * BLOCK_S >= seg_len:
        return

    s_offset = tl.arange(0, BLOCK_S) + pid_s * BLOCK_S
    n_offset = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    k_offset = tl.arange(0, BLOCK_K)

    s_physical = _resolve_token_positions(
        sorted_token_ids, seg_start, s_offset, seg_len, SORTED_BY_ADAPTER
    )

    row_mask = s_offset < seg_len
    safe_row = tl.minimum(s_physical, S - 1)
    n_mask = n_offset[None, :] < N
    safe_n = tl.minimum(n_offset, N - 1)

    partial_sum = tl.zeros((BLOCK_S, BLOCK_N), dtype=tl.float32)
    for k_block in range(0, tl.cdiv(K_eff, BLOCK_K)):
        cur_k = k_block * BLOCK_K + k_offset
        k_mask = cur_k < K_eff
        safe_k = tl.minimum(cur_k, K_eff - 1)

        # x[s, h, k]  (k iterates over rank)
        x_tile = tl.load(
            x
            + safe_row[:, None] * x_stride_s
            + head_id * x_stride_h
            + safe_k[None, :] * x_stride_k,
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0,
        )

        # A[slot, k, n]: read k along contraction, n along output.
        w_tile = tl.load(
            w
            + w_index * w_stride_l
            + safe_k[:, None] * w_stride_k
            + safe_n[None, :] * w_stride_n,
            mask=k_mask[:, None] & n_mask,
            other=0.0,
        )

        partial_sum += tl.dot(x_tile, w_tile)

    partial_sum *= scaling
    partial_sum = partial_sum.to(x.dtype.element_ty)

    # Accumulate into base[s, h, n].
    base_offs = (
        safe_row[:, None] * b_stride_s
        + head_id * b_stride_h
        + safe_n[None, :] * b_stride_n
    )
    out_mask = row_mask[:, None] & n_mask
    partial_sum += tl.load(base + base_offs, mask=out_mask, other=0.0)
    tl.store(base + base_offs, partial_sum, mask=out_mask)


def step_b_q_fwd(
    q_lora_a: torch.Tensor,
    A_buf: torch.Tensor,
    batch_info: LoRABatchInfo,
    base_output: torch.Tensor,
) -> torch.Tensor:
    """Step B of the q-side correction, accumulating into ``base_output``.

    Args:
        q_lora_a: ``(S, H, rank)`` from step A_q.
        A_buf: ``(num_lora, rank, kv_lora_rank)`` from the LoRA pool.
        batch_info: standard ``LoRABatchInfo``.
        base_output: ``(S, H, kv_lora_rank)``, modified in-place
            (the absorbed ``q_nope @ w_kc`` result).

    Returns:
        ``base_output`` (same object, mutated).
    """
    S, H, rank = q_lora_a.shape
    kv_lora_rank = A_buf.shape[-1]
    num_segments = _num_segments(batch_info)
    max_segment_len = _max_segment_len(batch_info)
    segment_grid = _segment_grid_size(batch_info, num_segments)

    grid = (
        triton.cdiv(max_segment_len, _BLOCK_S)
        * triton.cdiv(kv_lora_rank, _STEP_B_BLOCK_N),
        H,
        segment_grid,
    )
    sorted_by_adapter = batch_info.permutation is not None

    _step_b_q_kernel[grid](
        q_lora_a,
        A_buf,
        base_output,
        S,
        rank,
        kv_lora_rank,
        q_lora_a.stride(0),
        q_lora_a.stride(1),
        q_lora_a.stride(2),
        A_buf.stride(0),
        A_buf.stride(1),
        A_buf.stride(2),
        base_output.stride(0),
        base_output.stride(1),
        base_output.stride(2),
        batch_info.seg_indptr,
        batch_info.weight_indices,
        batch_info.lora_ranks,
        batch_info.permutation,
        batch_info.scalings,
        num_segments,
        SORTED_BY_ADAPTER=sorted_by_adapter,
        BLOCK_S=_BLOCK_S,
        BLOCK_N=_STEP_B_BLOCK_N,
        BLOCK_K=_STEP_B_BLOCK_K,
    )
    return base_output


# ---------------------------------------------------------------------------
# Kernel 3 -- Step A_v: shared-A.T per-slot SGMM (no scaling, fresh output)
#
#     attn_lora_a[t, h, r] = sum_k attn_output[t, h, k] * A[slot, r, k]
#
# x      : (S, H, kv_lora_rank)
# w (A)  : (num_lora, rank, kv_lora_rank) -- accessed transposed vs step B_q
# out    : (S, H, rank), fresh allocation
# ---------------------------------------------------------------------------


@triton.jit(do_not_specialize=["num_segments"])
def _step_a_v_kernel(
    x,
    w,
    out,
    # dims
    S,
    K,  # kv_lora_rank (contraction)
    N,  # rank (output)
    # strides
    x_stride_s,
    x_stride_h,
    x_stride_k,
    w_stride_l,
    w_stride_n,  # A's "rank" axis (= GEMM N)
    w_stride_k,  # A's "kv_lora_rank" axis (= GEMM K)
    out_stride_s,
    out_stride_h,
    out_stride_n,
    # batch info
    seg_indptr,
    weight_indices,
    lora_ranks,
    sorted_token_ids,
    num_segments,
    # meta
    SORTED_BY_ADAPTER: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    batch_id = tl.program_id(axis=2)
    head_id = tl.program_id(axis=1)
    pid = tl.program_id(axis=0)

    if batch_id >= num_segments:
        return

    w_index = tl.load(weight_indices + batch_id)
    cur_rank = tl.load(lora_ranks + w_index)
    if cur_rank == 0:
        return

    seg_start = tl.load(seg_indptr + batch_id)
    seg_end = tl.load(seg_indptr + batch_id + 1)
    seg_len = seg_end - seg_start
    if seg_len == 0:
        return

    # Truncate output N to this slot's rank.
    N_eff = tl.minimum(N, cur_rank)

    num_pid_n = tl.cdiv(N_eff, BLOCK_N)
    pid_s = pid // num_pid_n
    pid_n = pid % num_pid_n
    if pid_s * BLOCK_S >= seg_len:
        return

    s_offset = tl.arange(0, BLOCK_S) + pid_s * BLOCK_S
    n_offset = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    k_offset = tl.arange(0, BLOCK_K)

    s_physical = _resolve_token_positions(
        sorted_token_ids, seg_start, s_offset, seg_len, SORTED_BY_ADAPTER
    )

    row_mask = s_offset < seg_len
    safe_row = tl.minimum(s_physical, S - 1)
    safe_n = tl.minimum(n_offset, N_eff - 1)

    partial_sum = tl.zeros((BLOCK_S, BLOCK_N), dtype=tl.float32)
    for k_block in range(0, tl.cdiv(K, BLOCK_K)):
        cur_k = k_block * BLOCK_K + k_offset
        k_mask = cur_k < K
        safe_k = tl.minimum(cur_k, K - 1)

        # x[s, h, k]
        x_tile = tl.load(
            x
            + safe_row[:, None] * x_stride_s
            + head_id * x_stride_h
            + safe_k[None, :] * x_stride_k,
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0,
        )

        # A[slot, r, k] -- here we want each (k, r) so we read along k
        # (inner / contraction) and produce r as output.  Stride access:
        # the row dim is r (= GEMM N), column dim is k (= GEMM K).
        w_tile = tl.load(
            w
            + w_index * w_stride_l
            + safe_k[:, None] * w_stride_k
            + safe_n[None, :] * w_stride_n,
            mask=k_mask[:, None] & (n_offset[None, :] < N_eff),
            other=0.0,
        )

        partial_sum += tl.dot(x_tile, w_tile)

    partial_sum = partial_sum.to(x.dtype.element_ty)
    out_offs = (
        safe_row[:, None] * out_stride_s
        + head_id * out_stride_h
        + safe_n[None, :] * out_stride_n
    )
    out_mask = row_mask[:, None] & (n_offset[None, :] < N_eff)
    tl.store(out + out_offs, partial_sum, mask=out_mask)


def step_a_v_fwd(
    attn_output: torch.Tensor,
    A_buf: torch.Tensor,
    batch_info: LoRABatchInfo,
) -> torch.Tensor:
    """Step A of the v-side correction.

    Args:
        attn_output: ``(S, H, kv_lora_rank)``, the post-attention intermediate.
        A_buf: ``(num_lora, rank, kv_lora_rank)``.
        batch_info: standard ``LoRABatchInfo``.

    Returns:
        ``(S, H, rank)`` -- per-token, per-head low-rank intermediate for step B_v.
    """
    S, H, kv_lora_rank = attn_output.shape
    rank = A_buf.shape[1]
    out = torch.empty((S, H, rank), device=attn_output.device, dtype=attn_output.dtype)
    num_segments = _num_segments(batch_info)
    max_segment_len = _max_segment_len(batch_info)
    segment_grid = _segment_grid_size(batch_info, num_segments)

    grid = (
        triton.cdiv(max_segment_len, _BLOCK_S) * triton.cdiv(rank, _STEP_A_BLOCK_N),
        H,
        segment_grid,
    )
    sorted_by_adapter = batch_info.permutation is not None

    _step_a_v_kernel[grid](
        attn_output,
        A_buf,
        out,
        S,
        kv_lora_rank,
        rank,
        attn_output.stride(0),
        attn_output.stride(1),
        attn_output.stride(2),
        A_buf.stride(0),
        A_buf.stride(1),
        A_buf.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        batch_info.seg_indptr,
        batch_info.weight_indices,
        batch_info.lora_ranks,
        batch_info.permutation,
        num_segments,
        SORTED_BY_ADAPTER=sorted_by_adapter,
        BLOCK_S=_BLOCK_S,
        BLOCK_N=_STEP_A_BLOCK_N,
        BLOCK_K=_STEP_A_BLOCK_K,
    )
    return out


# ---------------------------------------------------------------------------
# Kernel 4 -- Step B_v: per-head per-slot SGMM with V-half of B (transposed),
# scaled + accumulated
#
#     base[t, h, j] += sum_r x[t, h, r] * B[slot, h*FULL_K + qk_nope + j, r] * scaling
#
# x      : (S, H, rank)
# w (B)  : (num_lora, H*FULL_K, rank), V-half slice via offset
# base   : (S, H, v_head_dim), updated in-place (accumulated)
# ---------------------------------------------------------------------------


@triton.jit(do_not_specialize=["num_segments"])
def _step_b_v_kernel(
    x,
    w,
    base,
    # dims
    S,
    K,  # rank (contraction)
    N,  # v_head_dim (output)
    # strides
    x_stride_s,
    x_stride_h,
    x_stride_k,
    w_stride_l,
    w_stride_n,  # B's row dim (h*FULL_K + j) -- this is GEMM N
    w_stride_k,  # B's rank dim -- this is GEMM K
    b_stride_s,
    b_stride_h,
    b_stride_n,
    # batch info
    seg_indptr,
    weight_indices,
    lora_ranks,
    sorted_token_ids,
    scalings,
    num_segments,
    # meta
    FULL_K: tl.constexpr,  # qk_nope + v_head_dim
    QK_NOPE_OFFSET: tl.constexpr,  # offset of V-half within each head's row block
    SORTED_BY_ADAPTER: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    batch_id = tl.program_id(axis=2)
    head_id = tl.program_id(axis=1)
    pid = tl.program_id(axis=0)

    if batch_id >= num_segments:
        return

    w_index = tl.load(weight_indices + batch_id)
    cur_rank = tl.load(lora_ranks + w_index)
    if cur_rank == 0:
        return

    seg_start = tl.load(seg_indptr + batch_id)
    seg_end = tl.load(seg_indptr + batch_id + 1)
    seg_len = seg_end - seg_start
    if seg_len == 0:
        return
    scaling = tl.load(scalings + w_index)

    K_eff = tl.minimum(K, cur_rank)

    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_s = pid // num_pid_n
    pid_n = pid % num_pid_n
    if pid_s * BLOCK_S >= seg_len:
        return

    s_offset = tl.arange(0, BLOCK_S) + pid_s * BLOCK_S
    n_offset = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    k_offset = tl.arange(0, BLOCK_K)

    s_physical = _resolve_token_positions(
        sorted_token_ids, seg_start, s_offset, seg_len, SORTED_BY_ADAPTER
    )

    row_mask = s_offset < seg_len
    safe_row = tl.minimum(s_physical, S - 1)
    n_mask = n_offset[None, :] < N
    safe_n = tl.minimum(n_offset, N - 1)

    # V-half row base for this head: h*FULL_K + qk_nope
    head_row_base = head_id * FULL_K + QK_NOPE_OFFSET

    partial_sum = tl.zeros((BLOCK_S, BLOCK_N), dtype=tl.float32)
    for k_block in range(0, tl.cdiv(K_eff, BLOCK_K)):
        cur_k = k_block * BLOCK_K + k_offset
        k_mask = cur_k < K_eff
        safe_k = tl.minimum(cur_k, K_eff - 1)

        # x[s, h, k]
        x_tile = tl.load(
            x
            + safe_row[:, None] * x_stride_s
            + head_id * x_stride_h
            + safe_k[None, :] * x_stride_k,
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0,
        )

        # B[slot, h*FULL_K + qk_nope + j, r] -- row dim is j (= GEMM N),
        # column dim is r (= GEMM K).  Transposed access vs step A_q.
        w_tile = tl.load(
            w
            + w_index * w_stride_l
            + safe_k[:, None] * w_stride_k
            + (head_row_base + safe_n[None, :]) * w_stride_n,
            mask=k_mask[:, None] & n_mask,
            other=0.0,
        )

        partial_sum += tl.dot(x_tile, w_tile)

    partial_sum *= scaling
    partial_sum = partial_sum.to(x.dtype.element_ty)

    base_offs = (
        safe_row[:, None] * b_stride_s
        + head_id * b_stride_h
        + safe_n[None, :] * b_stride_n
    )
    out_mask = row_mask[:, None] & n_mask
    partial_sum += tl.load(base + base_offs, mask=out_mask, other=0.0)
    tl.store(base + base_offs, partial_sum, mask=out_mask)


def step_b_v_fwd(
    attn_lora_a: torch.Tensor,
    B_buf: torch.Tensor,
    batch_info: LoRABatchInfo,
    base_output: torch.Tensor,
    qk_nope_head_dim: int,
    v_head_dim: int,
) -> torch.Tensor:
    """Step B of the v-side correction, accumulating into ``base_output``.

    Args:
        attn_lora_a: ``(S, H, rank)`` from step A_v.
        B_buf: ``(num_lora, H*(qk_nope+v_head_dim), rank)``.
        batch_info: standard ``LoRABatchInfo``.
        base_output: ``(S, H, v_head_dim)``, modified in-place
            (the absorbed ``attn_output @ w_vc`` result).
        qk_nope_head_dim: offset of V-half within each head's row block of B.
        v_head_dim: output feature dim per head.

    Returns:
        ``base_output`` (same object, mutated).
    """
    S, H, rank = attn_lora_a.shape
    full_K_per_head = qk_nope_head_dim + v_head_dim
    num_segments = _num_segments(batch_info)
    max_segment_len = _max_segment_len(batch_info)
    segment_grid = _segment_grid_size(batch_info, num_segments)

    grid = (
        triton.cdiv(max_segment_len, _BLOCK_S)
        * triton.cdiv(v_head_dim, _STEP_B_BLOCK_N),
        H,
        segment_grid,
    )
    sorted_by_adapter = batch_info.permutation is not None

    _step_b_v_kernel[grid](
        attn_lora_a,
        B_buf,
        base_output,
        S,
        rank,
        v_head_dim,
        attn_lora_a.stride(0),
        attn_lora_a.stride(1),
        attn_lora_a.stride(2),
        B_buf.stride(0),
        B_buf.stride(1),
        B_buf.stride(2),
        base_output.stride(0),
        base_output.stride(1),
        base_output.stride(2),
        batch_info.seg_indptr,
        batch_info.weight_indices,
        batch_info.lora_ranks,
        batch_info.permutation,
        batch_info.scalings,
        num_segments,
        FULL_K=full_K_per_head,
        QK_NOPE_OFFSET=qk_nope_head_dim,
        SORTED_BY_ADAPTER=sorted_by_adapter,
        BLOCK_S=_BLOCK_S,
        BLOCK_N=_STEP_B_BLOCK_N,
        BLOCK_K=_STEP_B_BLOCK_K,
    )
    return base_output
