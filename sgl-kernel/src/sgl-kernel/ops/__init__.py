from sgl_kernel.ops._kernels import all_reduce as _all_reduce
from sgl_kernel.ops._kernels import dispose as _dispose
from sgl_kernel.ops._kernels import init_custom_ar as _init_custom_ar
from sgl_kernel.ops._kernels import int8_scaled_mm as _int8_scaled_mm
from sgl_kernel.ops._kernels import moe_align_block_size as _moe_align_block_size
from sgl_kernel.ops._kernels import moe_align_block_size_stage1 as _moe_align_block_size_stage1
from sgl_kernel.ops._kernels import moe_align_block_size_stage2 as _moe_align_block_size_stage2
from sgl_kernel.ops._kernels import moe_align_block_size_stage3 as _moe_align_block_size_stage3
from sgl_kernel.ops._kernels import moe_align_block_size_stage4 as _moe_align_block_size_stage4
from sgl_kernel.ops._kernels import moe_align_block_size_stage5 as _moe_align_block_size_stage5

import torch
import triton
import triton.language as tl

def init_custom_reduce(rank_id, num_devices, buffers, barrier_in, barrier_out):
    return _init_custom_ar(rank_id, num_devices, buffers, barrier_in, barrier_out)


def custom_dispose(fa):
    _dispose(fa)


def custom_reduce(fa, inp, out):
    _all_reduce(fa, inp, out)


def moe_align_block_size(
    topk_ids,
    num_experts,
    block_size,
    sorted_token_ids,
    experts_ids,
    num_tokens_post_pad,
    token_cnts_buffer,
    cumsum_buffer,
):
    _moe_align_block_size(
        topk_ids,
        num_experts,
        block_size,
        sorted_token_ids,
        experts_ids,
        num_tokens_post_pad,
        token_cnts_buffer,
        cumsum_buffer,
    )

def ceil_div(a, b):
    return (a + b - 1) // b

@triton.jit
def triton_moe_align_block_size_stage5(
    topk_ids_ptr,
    sorted_token_ids_ptr,
    tokens_cnts_ptr,
    cumsum_ptr,
    num_experts: tl.constexpr,
    block_size: tl.constexpr,
    numel: tl.constexpr,
    tokens_per_thread: tl.constexpr,
):
    pid = tl.program_id(0)

    start_idx = pid * tokens_per_thread
    off_t = pid * num_experts

    for i in range(start_idx, tl.minimum(start_idx + tokens_per_thread, numel)):
        expert_id = tl.load(topk_ids_ptr + i)
        token_cnt = tl.load(tokens_cnts_ptr + off_t + expert_id)
        rank_post_pad = token_cnt + tl.load(cumsum_ptr + expert_id)
        tl.store(sorted_token_ids_ptr + rank_post_pad, i)
        tl.store(tokens_cnts_ptr + off_t + expert_id, token_cnt + 1)

def moe_align_block_size_triton(
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    tokens_cnts: torch.Tensor,
    cumsum: torch.Tensor,
    num_experts: int,
    block_size: int,
) -> None:
    numel = topk_ids.numel()
    grid = (num_experts,)
    tokens_per_thread = ceil_div(numel, num_experts)

    triton_moe_align_block_size_stage5[grid](
        topk_ids,
        sorted_token_ids,
        tokens_cnts,
        cumsum,
        num_experts,
        block_size,
        numel,
        tokens_per_thread,
    )

def moe_align_block_size_v2(
    topk_ids,
    num_experts,
    block_size,
    sorted_token_ids,
    experts_ids,
    num_tokens_post_pad,
    token_cnts_buffer,
    cumsum_buffer,
):
    _moe_align_block_size_stage1(topk_ids, token_cnts_buffer, num_experts)
    _moe_align_block_size_stage2(token_cnts_buffer, num_experts)
    _moe_align_block_size_stage3(topk_ids, sorted_token_ids, num_tokens_post_pad, token_cnts_buffer, cumsum_buffer, num_experts, block_size)
    _moe_align_block_size_stage4(topk_ids, sorted_token_ids, experts_ids, token_cnts_buffer, cumsum_buffer, num_experts, block_size, experts_ids.numel())
    # _moe_align_block_size_stage5(topk_ids, sorted_token_ids, token_cnts_buffer, cumsum_buffer, num_experts)
    moe_align_block_size_triton(topk_ids, sorted_token_ids, token_cnts_buffer, cumsum_buffer, num_experts, block_size)

def int8_scaled_mm(mat_a, mat_b, scales_a, scales_b, out_dtype, bias=None):
    return _int8_scaled_mm(
        mat_a,
        mat_b,
        scales_a,
        scales_b,
        out_dtype,
        bias,
    )
