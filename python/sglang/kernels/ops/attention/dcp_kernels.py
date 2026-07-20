# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Triton kernels for decode context parallel (DCP).

Consolidated from the two merged DCP implementations:
  - create_triton_kv_indices_for_dcp_triton  (PR #25090, Triton/MHA path)
  - create_dcp_kv_indices / update_kv_lens_and_indices  (PR #14194, MLA path)
  - _correct_attn_cp_out_kernel / correct_attn_out / CPTritonContext  (PR #14194)
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# KV-index build (PR #25090, Triton/MHA): per-rank local KV indices.
# ---------------------------------------------------------------------------
@triton.jit
def create_triton_kv_indices_for_dcp_triton(
    req_to_token_ptr,  # [max_batch, max_context_len]
    req_pool_indices_ptr,
    dcp_kernel_lens_ptr,
    kv_indptr,
    kv_start_idx,
    kv_indices_ptr,
    req_to_token_ptr_stride: tl.constexpr,
    dcp_size: tl.constexpr,
    dcp_rank: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(axis=0)
    req_pool_index = tl.load(req_pool_indices_ptr + pid)
    kv_indices_offset = tl.load(kv_indptr + pid)

    kv_start = 0
    if kv_start_idx:
        kv_start = tl.load(kv_start_idx + pid).to(tl.int32)

    # First absolute token position in this range owned by dcp_rank.
    # Triton follows C-style remainder for negative values, so avoid
    # computing the offset as a negative remainder when kv_start > dcp_rank.
    kv_start_mod = kv_start % dcp_size
    first = kv_start + ((dcp_rank + dcp_size - kv_start_mod) % dcp_size)
    local_len = tl.load(dcp_kernel_lens_ptr + pid).to(tl.int32)

    num_loop = tl.cdiv(local_len, BLOCK_SIZE)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_SIZE).to(tl.int64) + i * BLOCK_SIZE
        mask = offset < local_len
        abs_pos = first + offset * dcp_size
        data = tl.load(
            req_to_token_ptr + req_pool_index * req_to_token_ptr_stride + abs_pos,
            mask=mask,
        )
        tl.store(
            kv_indices_ptr + kv_indices_offset + offset, data // dcp_size, mask=mask
        )


# ---------------------------------------------------------------------------
# KV-index build (PR #14194, MLA): global prefix+extend layout for the
# all-gathered dcp_kv_buffer, plus the per-rank shard/compact kernel.
# ---------------------------------------------------------------------------
@triton.jit
def create_dcp_kv_indices(
    kv_indptr,
    extend_lens_ptr,
    extend_cu_lens_ptr,
    extend_prefix_lens_ptr,
    extend_cu_prefix_lens_ptr,
    kv_indices_ptr,
    extend_prefix_lens_sum,
    dcp_world_size: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(axis=0)
    prefix_len = tl.load(extend_prefix_lens_ptr + pid)
    prefix_start = tl.load(extend_cu_prefix_lens_ptr + pid)
    kv_ind_start = tl.load(kv_indptr + pid)
    num_loop = tl.cdiv(prefix_len, BLOCK_SIZE)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = offset < prefix_len
        data = prefix_start + offset
        tl.store(kv_indices_ptr + kv_ind_start + offset, data, mask=mask)
    extend_len = tl.load(extend_lens_ptr + pid)
    extend_start = tl.load(extend_cu_lens_ptr + pid)
    num_loop = tl.cdiv(extend_len, BLOCK_SIZE)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = offset < extend_len
        data = extend_prefix_lens_sum + extend_start + offset
        tl.store(
            kv_indices_ptr + kv_ind_start + prefix_len + offset,
            data,
            mask=mask,
        )


@triton.jit
def update_kv_lens_and_indices(
    kv_lens: torch.Tensor,
    kv_lens_cumsum: torch.Tensor,
    kv_indices: torch.Tensor,
    local_kv_lens: torch.Tensor,
    local_kv_lens_cumsum: torch.Tensor,
    local_kv_indices: torch.Tensor,
    dcp_rank: tl.constexpr,
    dcp_world_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    bs_idx = tl.program_id(0)
    block_idx = tl.program_id(1)

    local_kv_len = tl.load(local_kv_lens + bs_idx)
    local_kv_indices_start = tl.load(local_kv_lens_cumsum + bs_idx)
    kv_indices_start = tl.load(kv_lens_cumsum + bs_idx)

    block_start = block_idx * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < local_kv_len

    kv_indice_offsets = offsets * dcp_world_size + dcp_rank + kv_indices_start
    local_kv_indices_offsets = local_kv_indices_start + offsets

    kv_values = tl.load(kv_indices + kv_indice_offsets, mask=mask)
    tl.store(
        local_kv_indices + local_kv_indices_offsets,
        kv_values // dcp_world_size,
        mask=mask,
    )


# ---------------------------------------------------------------------------
# Partial-attention LSE correction (PR #14194, MLA path).
# ---------------------------------------------------------------------------
@triton.jit
def _correct_attn_cp_out_kernel(
    outputs_ptr,
    new_output_ptr,
    lses_ptr,
    vlse_ptr,
    outputs_stride_B,
    outputs_stride_H,
    outputs_stride_D,
    lses_stride_N,
    lses_stride_B,
    lses_stride_H,
    new_outputs_stride_H,
    new_outputs_stride_B,
    new_outputs_stride_D,
    lse_idx,
    HEAD_DIM: tl.constexpr,
    N_ROUNDED: tl.constexpr,
):
    """
    Apply the all-gathered lses to correct each local rank's attention
    output. we still need perform a cross-rank reduction to obtain the
    final attention output.

    Args:
        outputs_ptr (triton.PointerType):
            Pointer to input tensor of shape [ B, H, D ]
        lses_ptr (triton.PointerType):
            Pointer to input tensor of shape [ N, B, H ]
        new_output_ptr (triton.PointerType):
            Pointer to output tensor of shape [ H, B, D ]
        vlse_ptr (triton.PointerType):
            Pointer to output tensor of shape [ B, H ]
    """
    batch_idx = tl.program_id(axis=0).to(tl.int64)
    head_idx = tl.program_id(axis=1).to(tl.int64)

    # Use int32 for offsets where possible to reduce register pressure
    b_i32 = batch_idx.to(tl.int32)
    h_i32 = head_idx.to(tl.int32)

    # Vectorized load of LSE values: shape = [N]
    num_n_offsets = tl.arange(0, N_ROUNDED)
    lse_offsets = (
        num_n_offsets * lses_stride_N + b_i32 * lses_stride_B + h_i32 * lses_stride_H
    )

    # Compute final LSE using online softmax algorithm (more numerically stable)
    lse = tl.load(lses_ptr + lse_offsets)

    # Replace NaN and inf with -inf for numerical stability
    neg_inf = float("-inf")
    lse = tl.where((lse != lse) | (lse == float("inf")), neg_inf, lse)

    # Online softmax: find max, subtract, exp, sum, log
    lse_max = tl.max(lse, axis=0)
    lse_max = tl.where(lse_max == neg_inf, 0.0, lse_max)
    lse = lse - lse_max
    lse_exp = tl.exp2(lse)
    lse_acc = tl.sum(lse_exp, axis=0)
    final_lse = tl.log2(lse_acc) + lse_max

    # Compute correction factor
    lse_offset = lse_idx * lses_stride_N + b_i32 * lses_stride_B + h_i32 * lses_stride_H
    local_lse = tl.load(lses_ptr + lse_offset)
    lse_diff = local_lse - final_lse
    lse_diff = tl.where(
        (lse_diff != lse_diff) | (lse_diff == float("inf")),
        neg_inf,
        lse_diff,
    )
    factor = tl.exp2(lse_diff)

    # Store final LSE
    tl.store(vlse_ptr + b_i32 * lses_stride_B + h_i32 * lses_stride_H, final_lse)

    # Load output with vectorized access: shape = [D]
    d_offsets = tl.arange(0, HEAD_DIM)
    output_offsets = (
        batch_idx * outputs_stride_B
        + head_idx * outputs_stride_H
        + d_offsets * outputs_stride_D
    )

    new_output_offsets = (
        head_idx * new_outputs_stride_H
        + batch_idx * new_outputs_stride_B
        + d_offsets * new_outputs_stride_D
    )
    # Apply correction and store
    output = tl.load(outputs_ptr + output_offsets)
    output = output * factor
    tl.store(new_output_ptr + new_output_offsets, output)


class CPTritonContext:
    """The CPTritonContext is used to avoid recompilation of the Triton JIT."""

    def __init__(self):
        self.inner_kernel = None

    def call_kernel(self, kernel, grid, *regular_args, **const_args):
        if self.inner_kernel is None:
            self.inner_kernel = kernel[grid](*regular_args, **const_args)
        else:
            self.inner_kernel[grid](*regular_args)


def correct_attn_out(
    out: torch.Tensor,
    lses: torch.Tensor,
    cp_rank: int,
    ctx: Optional[CPTritonContext],
    new_output: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Correct the attention output using the all-gathered lses.

    Args:
        out: Tensor of shape [ B, H, D ]
        lses: Tensor of shape [ N, B, H ]
        cp_rank: Current rank in the context-parallel group
        ctx: Triton context to avoid recompilation

    Returns:
        Tuple of (out, lse) with corrected attention and final log-sum-exp.
    """
    if ctx is None:
        ctx = CPTritonContext()

    # --- Normalize to 3D views ---
    if out.ndim == 4 and out.shape[1] == 1:
        out = out.squeeze(1)
    assert out.ndim == 3, f"expected out [B,H,D] or [B,1,H,D], got {tuple(out.shape)}"

    if lses.ndim == 4 and lses.shape[-1] == 1:
        lses = lses.squeeze(-1)
    if lses.ndim == 4 and lses.shape[1] == 1:
        lses = lses.squeeze(1)
    assert lses.ndim == 3, (
        f"expected lses [N,B,H] (optionally with a 1-sized extra dim), "
        f"got {tuple(lses.shape)}"
    )

    B, H, D = out.shape
    N = lses.shape[0]

    # Strides after we normalized shapes to 3-D views.  The kernel computes
    # offsets for `vlse_ptr` using lses_stride_B/H, so the output buffer must
    # have the same B/H stride layout as a slice of `lses`.
    o_sB, o_sH, o_sD = out.stride()
    l_sN, l_sB, l_sH = lses.stride()
    no_sH, no_sB, no_sD = new_output.stride()
    # Allocate LSE with the same B/H strides as `lses` so writes land correctly
    # even when `lses` is a non-contiguous view (e.g., 4-D to 3-D squeeze).
    lse = torch.empty_strided(
        (B, H), (l_sB, l_sH), device=lses.device, dtype=lses.dtype
    )

    # Kernel launch config
    grid = (B, H, 1)

    regular_args = (
        out,
        new_output,
        lses,
        lse,
        o_sB,
        o_sH,
        o_sD,
        l_sN,
        l_sB,
        l_sH,
        no_sH,
        no_sB,
        no_sD,
        cp_rank,
    )
    const_args = {"HEAD_DIM": D, "N_ROUNDED": N}

    ctx.call_kernel(_correct_attn_cp_out_kernel, grid, *regular_args, **const_args)
    return new_output, lse


# ---------------------------------------------------------------------------
# A2A DCP reduce: LSE-weighted combine of N partial attention outputs.
# Used by the a2a / fi_a2a communication backends (see comm.py).
# ---------------------------------------------------------------------------


def _lse_pack_dim(output_dtype: torch.dtype) -> int:
    """Number of output-dtype elements needed to store one fp32 LSE value."""
    return torch.finfo(torch.float32).bits // torch.finfo(output_dtype).bits


@triton.jit
def _dcp_lse_combine_kernel(
    recv_output_ptr,
    recv_lse_ptr,
    out_ptr,
    out_lse_ptr,
    recv_output_stride_N,
    recv_output_stride_B,
    recv_output_stride_H,
    recv_output_stride_D,
    recv_lse_stride_N,
    recv_lse_stride_B,
    recv_lse_stride_H,
    out_stride_B,
    out_stride_H,
    out_stride_D,
    out_lse_stride_B,
    out_lse_stride_H,
    N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IS_BASE_E: tl.constexpr,
    RETURN_LSE: tl.constexpr,
):
    """Combine N partial attention outputs weighted by their LSE values.

    Grid: (B, H_local).
    Each program handles one (batch, head) position across all N shards.

    Two-pass approach:
    Pass 1: find max LSE and weight sum across shards
    Pass 2: accumulate weighted outputs
    """
    batch_idx = tl.program_id(0).to(tl.int64)
    head_idx = tl.program_id(1).to(tl.int64)
    d_offsets = tl.arange(0, HEAD_DIM)

    lse_base = batch_idx * recv_lse_stride_B + head_idx * recv_lse_stride_H

    # Pass 1: find max LSE across N shards
    lse_max = tl.load(recv_lse_ptr + lse_base).to(tl.float32)
    lse_max = tl.where(
        (lse_max != lse_max) | (lse_max == float("inf")), -float("inf"), lse_max
    )
    for i in tl.static_range(1, N):
        lse_i = tl.load(recv_lse_ptr + lse_base + i * recv_lse_stride_N).to(tl.float32)
        lse_i = tl.where(
            (lse_i != lse_i) | (lse_i == float("inf")), -float("inf"), lse_i
        )
        lse_max = tl.where(lse_i > lse_max, lse_i, lse_max)

    lse_max = tl.where(lse_max == -float("inf"), 0.0, lse_max)

    # Pass 2: accumulate weighted outputs
    weight_sum = tl.zeros([], dtype=tl.float32)
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    for i in tl.static_range(N):
        lse_i = tl.load(recv_lse_ptr + lse_base + i * recv_lse_stride_N).to(tl.float32)
        lse_i = tl.where(
            (lse_i != lse_i) | (lse_i == float("inf")), -float("inf"), lse_i
        )
        centered = lse_i - lse_max
        if IS_BASE_E:
            w = tl.exp(centered)
        else:
            w = tl.exp2(centered)
        weight_sum += w

        o_offsets = (
            i * recv_output_stride_N
            + batch_idx * recv_output_stride_B
            + head_idx * recv_output_stride_H
            + d_offsets * recv_output_stride_D
        )
        partial_out = tl.load(recv_output_ptr + o_offsets).to(tl.float32)
        acc += partial_out * w

    acc = acc / weight_sum

    out_offsets = (
        batch_idx * out_stride_B + head_idx * out_stride_H + d_offsets * out_stride_D
    )
    tl.store(out_ptr + out_offsets, acc.to(out_ptr.dtype.element_ty))

    if RETURN_LSE:
        if IS_BASE_E:
            global_lse = tl.log(weight_sum) + lse_max
        else:
            global_lse = tl.log2(weight_sum) + lse_max
        # Store with out_lse's OWN strides. The recv_lse strides used before
        # only worked because every caller passed a contiguous [N, B, H]
        # recv_lse whose (B, H) strides coincide with contiguous out_lse's;
        # a strided recv_lse (the zero-copy packed-tail view of
        # comm.dcp_unpack_lse_combine) made the store scatter out of bounds.
        out_lse_offset = batch_idx * out_lse_stride_B + head_idx * out_lse_stride_H
        tl.store(out_lse_ptr + out_lse_offset, global_lse)


def dcp_lse_combine_triton(
    recv_output: torch.Tensor,
    recv_lse: torch.Tensor,
    is_lse_base_on_e: bool = True,
    return_lse: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Launch the Triton LSE-combine kernel.

    Args:
        recv_output: [N, B, H_local, D] partial outputs from each DCP rank.
        recv_lse:    [N, B, H_local]    log-sum-exp from each DCP rank.
        is_lse_base_on_e: True if LSE uses base-e (FlashAttention),
                          False if base-2 (FlashInfer).
        return_lse: If True, also return the combined global LSE.

    Returns:
        (combined_output [B, H_local, D], combined_lse [B, H_local] or None)
    """
    N, B, H_local, D = recv_output.shape
    out = torch.empty(
        (B, H_local, D), device=recv_output.device, dtype=recv_output.dtype
    )
    if return_lse:
        out_lse = torch.empty(
            (B, H_local), device=recv_lse.device, dtype=recv_lse.dtype
        )
        out_lse_stride_b, out_lse_stride_h = out_lse.stride(0), out_lse.stride(1)
    else:
        out_lse = recv_lse.new_empty(0)  # 1-D placeholder, never stored to
        out_lse_stride_b = out_lse_stride_h = 0

    grid = (B, H_local)
    _dcp_lse_combine_kernel[grid](
        recv_output,
        recv_lse,
        out,
        out_lse,
        recv_output.stride(0),
        recv_output.stride(1),
        recv_output.stride(2),
        recv_output.stride(3),
        recv_lse.stride(0),
        recv_lse.stride(1),
        recv_lse.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out_lse_stride_b,
        out_lse_stride_h,
        N=N,
        HEAD_DIM=D,
        IS_BASE_E=is_lse_base_on_e,
        RETURN_LSE=return_lse,
    )
    return out, (out_lse if return_lse else None)


def _lse_weighted_combine_cpu(
    partial_outputs: torch.Tensor,
    partial_lses: torch.Tensor,
    is_lse_base_on_e: bool = True,
) -> torch.Tensor:
    """CPU reference: combine N partial attention outputs using LSE weights.

    Args:
        partial_outputs: [N, B, H_local, D]
        partial_lses:    [N, B, H_local]
        is_lse_base_on_e: base-e (True) or base-2 (False)

    Returns:
        [B, H_local, D] combined output
    """
    N, B, H_local, D = partial_outputs.shape
    partial_outputs = partial_outputs.float()
    partial_lses = partial_lses.float()

    # Sanitize
    partial_lses = torch.where(
        torch.isnan(partial_lses) | torch.isinf(partial_lses),
        torch.full_like(partial_lses, float("-inf")),
        partial_lses,
    )

    # Max LSE for numerical stability: [B, H_local]
    lse_max, _ = partial_lses.max(dim=0)
    lse_max = torch.where(lse_max == float("-inf"), torch.zeros_like(lse_max), lse_max)

    # Compute weights: [N, B, H_local]
    centered = partial_lses - lse_max.unsqueeze(0)
    if is_lse_base_on_e:
        weights = torch.exp(centered)
    else:
        weights = torch.pow(2.0, centered)

    weight_sum = weights.sum(dim=0, keepdim=True)
    weights = weights / weight_sum

    # Weighted sum: [N, B, H_local, D] * [N, B, H_local, 1] -> sum -> [B, H_local, D]
    combined = (partial_outputs * weights.unsqueeze(-1)).sum(dim=0)
    return combined


# ---------------------------------------------------------------------------
# Fused mask+pack for the DCP verify cascade: build the a2a send buffer
# (comm.dcp_a2a_lse_reduce eager layout) directly from the strided pass-1
# outputs, applying the zero-owner row mask inline. Replaces, per layer:
# 2x torch.where + 2x .contiguous() + the two pack copies (~6 kernels).
# ---------------------------------------------------------------------------


@triton.jit
def _dcp_mask_pack_kernel(
    o_ptr,
    lse_ptr,
    zero_ptr,
    send_ptr,
    send_f32_ptr,
    o_stride_b,
    o_stride_t,
    o_stride_h,
    o_stride_d,
    lse_stride_b,
    lse_stride_t,
    lse_stride_h,
    send_stride_n,
    send_stride_tok,
    send_stride_h,
    sendf_stride_n,
    sendf_stride_tok,
    sendf_stride_h,
    T,
    H_PER_RANK,
    LSE_F32_IDX,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """One program per (token, full head): write the head's D output values
    into send[rank_slot, token, local_head, :D] and its fp32 LSE into the
    packed tail (through the fp32 alias of the same buffer). Masked batch
    rows (zero_ptr[b] != 0) get out -> 0 and lse -> -inf, bit-identical to
    the torch.where pair they replace (pure select, no dtype promotion)."""
    tok = tl.program_id(0).to(tl.int64)
    hf = tl.program_id(1).to(tl.int64)
    b = tok // T
    t = tok - b * T
    n = hf // H_PER_RANK
    h = hf - n * H_PER_RANK

    masked = tl.load(zero_ptr + b) != 0

    d_off = tl.arange(0, BLOCK_D)
    d_msk = d_off < D
    o = tl.load(
        o_ptr + b * o_stride_b + t * o_stride_t + hf * o_stride_h + d_off * o_stride_d,
        mask=d_msk,
    )
    o = tl.where(masked, tl.zeros_like(o), o)
    tl.store(
        send_ptr
        + n * send_stride_n
        + tok * send_stride_tok
        + h * send_stride_h
        + d_off,
        o,
        mask=d_msk,
    )

    lse = tl.load(lse_ptr + b * lse_stride_b + t * lse_stride_t + hf * lse_stride_h).to(
        tl.float32
    )
    lse = tl.where(masked, float("-inf"), lse)
    tl.store(
        send_f32_ptr
        + n * sendf_stride_n
        + tok * sendf_stride_tok
        + h * sendf_stride_h
        + LSE_F32_IDX,
        lse,
    )


def dcp_mask_pack_triton(
    o1: torch.Tensor,
    lse1: torch.Tensor,
    zero_mask: torch.Tensor,
    cp_world: int,
    send_buf: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused zero-owner mask + a2a-send pack for the DCP verify cascade.

    Produces EXACTLY the eager send layout of ``comm.dcp_a2a_lse_reduce``:
    ``send_combined [cp_world, bs*T, H_per_rank, D + lse_pack_dim]`` in the
    output dtype, with the fp32 LSE bit-packed (reinterpreted as output-dtype
    elements) into the ``[D:]`` tail columns. The zero-owner row mask is
    applied inline (masked rows: out -> 0, lse -> -inf), collapsing the two
    ``torch.where``, the two ``.contiguous()`` and the pack copies of the old
    path into a single kernel.

    Args:
        o1:        [bs, T, H_full, D] pass-1 attention output (any stride).
        lse1:      [bs, T, H_full]    fp32 pass-1 LSE (any stride).
        zero_mask: [bs] bool — requests whose rank-local prefix is empty.
        cp_world:  DCP world size (H_full % cp_world == 0).
        send_buf:  optional preallocated [cp_world, bs*T, H_per_rank, D+lpd]
                   contiguous buffer in ``o1.dtype``.

    Returns:
        The packed send buffer, ready for the byte-level all_to_all
        (``comm.dcp_a2a_exchange_packed``).
    """
    bs, T, H_full, D = o1.shape
    assert (
        H_full % cp_world == 0
    ), f"num_heads ({H_full}) must be divisible by dcp_size ({cp_world})"
    H_per_rank = H_full // cp_world
    out_dtype = o1.dtype
    lpd = _lse_pack_dim(out_dtype)
    itemsize = out_dtype.itemsize
    # The fp32 LSE tail is written through an fp32 alias of the send buffer,
    # so every LSE slot must be 4-byte aligned.
    assert (D * itemsize) % 4 == 0 and ((D + lpd) * itemsize) % 4 == 0, (
        f"dcp_mask_pack_triton needs 4-byte-aligned LSE slots "
        f"(D={D}, itemsize={itemsize}, lse_pack_dim={lpd})"
    )
    assert lse1.dtype == torch.float32 and zero_mask.dtype == torch.bool

    B = bs * T
    if send_buf is None:
        send_buf = torch.empty(
            (cp_world, B, H_per_rank, D + lpd), dtype=out_dtype, device=o1.device
        )
    assert send_buf.is_contiguous() and send_buf.dtype == out_dtype
    send_f32 = send_buf.view(torch.float32)

    grid = (B, H_full)
    _dcp_mask_pack_kernel[grid](
        o1,
        lse1,
        zero_mask.view(torch.uint8),
        send_buf,
        send_f32,
        o1.stride(0),
        o1.stride(1),
        o1.stride(2),
        o1.stride(3),
        lse1.stride(0),
        lse1.stride(1),
        lse1.stride(2),
        send_buf.stride(0),
        send_buf.stride(1),
        send_buf.stride(2),
        send_f32.stride(0),
        send_f32.stride(1),
        send_f32.stride(2),
        T,
        H_per_rank,
        (D * itemsize) // 4,
        D=D,
        BLOCK_D=triton.next_power_of_2(D),
    )
    return send_buf


# ---------------------------------------------------------------------------
# Verify-cascade pass-2: tiny causal attention over the per-request draft
# chain (T <= 8 tokens). Replaces the full tokenspeed decode-kernel call,
# whose launch/tiling floor dwarfs the actual workload.
# ---------------------------------------------------------------------------


@triton.jit
def _dcp_pass2_causal_attn_kernel(
    q_ptr,
    k_ptr,
    kr_ptr,
    seq_ptr,
    out_ptr,
    lse_ptr,
    q_stride_b,
    q_stride_t,
    q_stride_h,
    q_stride_d,
    k_stride_n,
    k_stride_d,
    kr_stride_n,
    kr_stride_d,
    out_stride_b,
    out_stride_t,
    out_stride_h,
    out_stride_d,
    lse_stride_b,
    lse_stride_t,
    lse_stride_h,
    softmax_scale_log2,
    output_scale,
    T,
    D_LATENT: tl.constexpr,
    D_ROPE: tl.constexpr,
    T_BLOCK: tl.constexpr,
    BLOCK_DL: tl.constexpr,
    BLOCK_DR: tl.constexpr,
):
    """Tiny causal MLA attention over the per-request draft chain (verify
    pass-2). Grid: (bs*T, H) — one program per (batch, q_token, head) row.

    Score/softmax math mirrors the tokenspeed fp8 decode kernel it replaces
    (mla_decode_fp8.py epilogue): scores are the RAW fp8-value dots, the
    softmax runs in the exp2 domain with ``softmax_scale_log2 = softmax_scale
    * log2(e)``, the output is ``P @ V / row_sum * output_scale``, and the LSE
    is base-2: ``lse = log2(row_sum) + softmax_scale_log2 * row_max`` (the
    convention ``dcp_lse_combine_triton(is_lse_base_on_e=False)`` consumes).
    All compute is fp32 (the tokenspeed kernel additionally quantizes P to
    fp8 for its PV MMA; this kernel keeps P in fp32, so it is strictly more
    accurate). K doubles as V: MLA draft KV is the latent, V = K[:, :D_LATENT].

    Masking: q token ``qt`` attends KV ``j`` iff ``j <= qt`` and ``j <
    seq_lens[b]`` (runtime tensor — capture-safe; grid/shape are static). A
    row with zero valid positions emits lse = -inf (combine weights it out)
    and a non-finite out, same class of edge as the tokenspeed kernel's
    seq_len==0 NaN; verify always has seq_lens[b] == T >= 1.
    """
    n = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1).to(tl.int64)
    b = n // T
    qt = n - b * T

    seq_b = tl.load(seq_ptr + b)
    offs_t = tl.arange(0, T_BLOCK)
    valid = (offs_t <= qt) & (offs_t < seq_b)
    k_rows = b * T + offs_t

    # ---- scores s[j] = sum_d q[d] * k[j, d]  (latent + rope, raw fp32) ----
    q_base = q_ptr + b * q_stride_b + qt * q_stride_t + h * q_stride_h
    s = tl.zeros([T_BLOCK], dtype=tl.float32)
    for d0 in tl.static_range(0, D_LATENT, BLOCK_DL):
        d = d0 + tl.arange(0, BLOCK_DL)
        qv = tl.load(q_base + d * q_stride_d).to(tl.float32)
        kv = tl.load(
            k_ptr + k_rows[:, None] * k_stride_n + d[None, :] * k_stride_d,
            mask=valid[:, None],
            other=0.0,
        ).to(tl.float32)
        s += tl.sum(kv * qv[None, :], axis=1)
    dr = tl.arange(0, BLOCK_DR)
    dr_msk = dr < D_ROPE
    qr = tl.load(q_base + (D_LATENT + dr) * q_stride_d, mask=dr_msk, other=0.0).to(
        tl.float32
    )
    krv = tl.load(
        kr_ptr + k_rows[:, None] * kr_stride_n + dr[None, :] * kr_stride_d,
        mask=valid[:, None] & dr_msk[None, :],
        other=0.0,
    ).to(tl.float32)
    s += tl.sum(krv * qr[None, :], axis=1)

    # ---- softmax in the exp2 domain (tokenspeed convention) ----
    s = tl.where(valid, s, float("-inf"))
    row_max = tl.max(s, axis=0)
    row_max = tl.where(row_max == float("-inf"), 0.0, row_max)
    p = tl.exp2((s - row_max) * softmax_scale_log2)  # -inf -> 0
    row_sum = tl.sum(p, axis=0)
    lse = tl.log2(row_sum) + softmax_scale_log2 * row_max
    tl.store(lse_ptr + b * lse_stride_b + qt * lse_stride_t + h * lse_stride_h, lse)

    # ---- out = P @ V / row_sum * output_scale  (V = K latent) ----
    epi_scale = output_scale / row_sum
    out_base = out_ptr + b * out_stride_b + qt * out_stride_t + h * out_stride_h
    for d0 in tl.static_range(0, D_LATENT, BLOCK_DL):
        d = d0 + tl.arange(0, BLOCK_DL)
        v = tl.load(
            k_ptr + k_rows[:, None] * k_stride_n + d[None, :] * k_stride_d,
            mask=valid[:, None],
            other=0.0,
        ).to(tl.float32)
        acc = tl.sum(v * p[:, None], axis=0) * epi_scale
        tl.store(out_base + d * out_stride_d, acc.to(out_ptr.dtype.element_ty))


def dcp_pass2_causal_attn_triton(
    q: torch.Tensor,
    k_latent: torch.Tensor,
    k_rope: torch.Tensor,
    seq_lens: torch.Tensor,
    softmax_scale: float,
    output_scale: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Causal draft-chain attention for the DCP verify cascade pass-2.

    Replaces the tokenspeed decode-kernel call (SGLANG_DCP_TRITON_PASS2=0
    reverts): per sequence, T query tokens attend causally to that sequence's
    own T draft-token latents (chain topk=1, T <= 8), so the whole workload is
    a few thousand MACs — far below the big kernel's launch/tiling floor. Reads
    q and k/k_rope directly (any strides), so the per-layer draft page-pool
    build (torch.cat + zeros + copy) of the old path disappears too.

    Args:
        q:        [bs, T, H, D_latent + D_rope] query (fp8/bf16/fp16; may be a
                  strided head-slice view — no .contiguous() needed).
        k_latent: [bs*T, D_latent] draft-token latents (same dtype as q).
                  Row b*T + j is sequence b's j-th draft token. Doubles as V.
        k_rope:   [bs*T, D_rope] draft-token rope keys (same dtype as q).
        seq_lens: [bs] int32 valid draft-token counts (runtime tensor; == T
                  for target-verify).
        softmax_scale: raw-score scale (layer.scaling * k_scale for fp8 KV,
                  mirroring the tokenspeed backend's _run_decode_kernel).
        output_scale: linear output scale (k_scale for fp8 KV).

    Returns:
        (out [bs, T, H, D_latent] bf16 for fp8 q / q.dtype otherwise,
         lse [bs, T, H] fp32 base-2) — the exact (o2, lse2) contract of the
        tokenspeed pass-2 call this replaces.
    """
    bs, T, H, D_qk = q.shape
    N, D_latent = k_latent.shape
    D_rope = k_rope.shape[1]
    assert N == bs * T, f"k_latent rows ({N}) != bs*T ({bs * T})"
    assert D_qk == D_latent + D_rope, f"q last dim {D_qk} != {D_latent}+{D_rope}"
    assert k_rope.shape[0] == N
    assert k_latent.dtype == q.dtype and k_rope.dtype == q.dtype
    assert seq_lens.dtype == torch.int32 and seq_lens.numel() == bs

    BLOCK_DL = 128
    assert D_latent % BLOCK_DL == 0, f"D_latent ({D_latent}) % {BLOCK_DL} != 0"

    out_dtype = torch.bfloat16 if q.dtype == torch.float8_e4m3fn else q.dtype
    out = torch.empty((bs, T, H, D_latent), dtype=out_dtype, device=q.device)
    lse = torch.empty((bs, T, H), dtype=torch.float32, device=q.device)

    # Static grid (bs, T, H all shape-derived — capture-safe); runtime data
    # dependence only through seq_lens.
    grid = (bs * T, H)
    _dcp_pass2_causal_attn_kernel[grid](
        q,
        k_latent,
        k_rope,
        seq_lens,
        out,
        lse,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k_latent.stride(0),
        k_latent.stride(1),
        k_rope.stride(0),
        k_rope.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        lse.stride(0),
        lse.stride(1),
        lse.stride(2),
        softmax_scale * 1.4426950408889634,  # log2(e)
        output_scale,
        T,
        D_LATENT=D_latent,
        D_ROPE=D_rope,
        T_BLOCK=triton.next_power_of_2(T),
        BLOCK_DL=BLOCK_DL,
        BLOCK_DR=triton.next_power_of_2(D_rope),
    )
    return out, lse
