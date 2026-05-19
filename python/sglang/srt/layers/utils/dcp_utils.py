from dataclasses import dataclass
from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.distributed.parallel_state import (
    GroupCoordinator,
    get_dcp_rank,
    get_dcp_world_size,
)
from sglang.srt.model_executor.forward_batch_deepseek_mha_mixin import (
    create_chunked_prefix_cache_kv_indices,
)
from sglang.srt.server_args import get_global_server_args


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


def cp_lse_ag_out_rs(
    cp_attn_out: torch.Tensor,
    cp_attn_lse: torch.Tensor,
    cp_group: GroupCoordinator,
    ctx: Optional[CPTritonContext] = None,
):
    """
    cp_attn_out: [ B, H, D ]
    cp_attn_lse: [ B, H ]
    """
    if cp_group.world_size == 1:
        return cp_attn_out

    if ctx is None:
        ctx = CPTritonContext()

    with use_symmetric_memory(cp_group):
        # cp_attn_out is [B,H,D], we want to transpose it to [H,B,D] for the kernel, and then transpose back after correction.
        new_output = cp_attn_out.new_empty(cp_attn_out.transpose(0, 1).shape)
        cp_attn_lse = cp_attn_lse.clone()
    lses = cp_group.all_gather(cp_attn_lse, dim=0).view(
        (cp_group.world_size,) + cp_attn_lse.shape
    )
    out, _ = correct_attn_out(
        cp_attn_out, lses, cp_group.rank_in_group, ctx, new_output
    )
    out = cp_group.reduce_scatter_along_dim(out, dim=0)
    return out


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
    local_prefix_len = prefix_len // dcp_world_size
    prefix_start = tl.load(extend_cu_prefix_lens_ptr + pid) // dcp_world_size
    kv_ind_start = tl.load(kv_indptr + pid)
    num_loop = tl.cdiv(prefix_len, BLOCK_SIZE)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = offset < prefix_len
        data = (
            prefix_start
            + (offset % dcp_world_size) * (extend_prefix_lens_sum // dcp_world_size)
            + (offset // dcp_world_size % local_prefix_len)
        )
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


@dataclass
class DecodeContextParallelMetadata:
    # For decode context parallel
    dcp_kv_indptr: Optional[torch.Tensor] = None
    dcp_kv_buffer: Optional[torch.Tensor] = None
    dcp_kv_indices: Optional[torch.Tensor] = None
    dcp_local_prefix_kv_indices: Optional[torch.Tensor] = None
    dcp_extend_prefix_lens_sum: Optional[int] = None


def prepare_decode_context_parallel_metadata(
    seq_lens: torch.Tensor,
    extend_prefix_lens: torch.Tensor,
    extend_prefix_lens_cpu: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    seq_lens_sum: int,
    kv_buffer_shape: torch.Size,
    kv_cache_dtype,
    kv_cache_device,
) -> Optional[DecodeContextParallelMetadata]:
    if get_dcp_world_size() <= 1:
        return None
    # dcp_kv_buffer tokens' layout
    # [ rank0_r1.prefix_tokens, rank0_r2.prefix_tokens,
    #   ...,
    #   rank8_r2.prefix_tokens, rank8_r3.prefix_tokens,
    #   r1.extend_tokens, r2.extent_tokens, r3.extend_tokens ]
    extend_prefix_starts = torch.zeros(
        len(seq_lens),
        dtype=torch.int32,
        device=get_global_server_args().device,
    )
    extend_cu_prefix_lens = torch.zeros(
        len(seq_lens) + 1,
        dtype=torch.int32,
        device=get_global_server_args().device,
    )
    extend_cu_prefix_lens[1:] = torch.cumsum(extend_prefix_lens, dim=0)
    extend_cu_prefix_lens = extend_cu_prefix_lens[:-1]
    extend_prefix_lens_sum = sum([i for i in extend_prefix_lens_cpu])

    dcp_prefix_kv_indices = torch.empty(
        sum(extend_prefix_lens_cpu),
        dtype=torch.int32,
        device=get_global_server_args().device,
    )
    create_chunked_prefix_cache_kv_indices[(len(seq_lens),)](
        req_to_token,
        req_pool_indices,
        extend_prefix_starts,
        extend_prefix_lens,
        extend_cu_prefix_lens,
        dcp_prefix_kv_indices,
        req_to_token.shape[1],
    )
    dcp_kv_indptr = torch.zeros(
        len(seq_lens) + 1,
        dtype=torch.int32,
        device=get_global_server_args().device,
    )
    dcp_kv_indptr[1:] = seq_lens.cumsum(dim=0)
    dcp_kv_indptr = dcp_kv_indptr[: (len(seq_lens) + 1)]
    dcp_kv_indices = torch.zeros(
        seq_lens_sum,
        dtype=torch.int32,
        device=get_global_server_args().device,
    )

    extend_cu_lens = torch.zeros(
        len(seq_lens) + 1,
        dtype=torch.int32,
        device=get_global_server_args().device,
    )
    extend_cu_lens[1:] = torch.cumsum(extend_seq_lens, dim=0)
    extend_cu_lens = extend_cu_lens[:-1]

    create_dcp_kv_indices[(len(seq_lens),)](
        dcp_kv_indptr,
        extend_seq_lens,
        extend_cu_lens,
        extend_prefix_lens,
        extend_cu_prefix_lens,
        dcp_kv_indices,
        extend_prefix_lens_sum,
        get_dcp_world_size(),
    )
    dcp_local_prefix_kv_indices = (
        dcp_prefix_kv_indices[
            dcp_prefix_kv_indices % get_dcp_world_size() == get_dcp_rank()
        ]
        // get_dcp_world_size()
    )
    dcp_kv_buffer = torch.empty(
        (
            seq_lens_sum,
            *kv_buffer_shape[1:],
        ),
        dtype=kv_cache_dtype,
        device=kv_cache_device,
    )
    attn_dcp_metadata = DecodeContextParallelMetadata(
        dcp_kv_indptr=dcp_kv_indptr,
        dcp_kv_buffer=dcp_kv_buffer,
        dcp_kv_indices=dcp_kv_indices,
        dcp_local_prefix_kv_indices=dcp_local_prefix_kv_indices,
        dcp_extend_prefix_lens_sum=extend_prefix_lens_sum,
    )
    return attn_dcp_metadata
