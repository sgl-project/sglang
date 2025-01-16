import argparse
import itertools
import time

import torch
import torch.nn.functional as F

import triton
import triton.language as tl
# from sgl_kernel import moe_align_block_size_v2 as moe_align_block_size

from sgl_kernel import moe_align_block_size

SEED = 0
USE_TRITON_MOE_V2=True
# Torch ops 2x slower, ups!
USE_TORCH_OR_TORCH_JIT=False

# import os
# os.environ["TRITON_INTERPRET"] = "1"

def ceil_div(a, b):
    return (a + b - 1) // b

@triton.jit
def moe_align_block_size_stage1(
    topk_ids_ptr,
    tokens_cnts_ptr,
    num_experts: tl.constexpr,
    numel: tl.constexpr,
    tokens_per_thread: tl.constexpr,
):
    pid = tl.program_id(0)
    start_idx = pid * tokens_per_thread
    off_c = (pid + 1) * num_experts

    for i in range(tokens_per_thread):
        if start_idx + i < numel:
            expert_idx = tl.load(topk_ids_ptr + start_idx + i)
            token_cnt = tl.load(tokens_cnts_ptr + off_c + expert_idx)
            tl.store(tokens_cnts_ptr + off_c + expert_idx, token_cnt + 1)


# NOTE(yiakwy) : count tokens processed by each expert
@triton.jit
def moe_align_block_size_stage1_v2(
    topk_ids_ptr,
    tokens_cnts_ptr,
    expert_ids_pos_info_ptr,
    aligned_num_experts: tl.constexpr,
    num_tokens : tl.constexpr,
    K : tl.constexpr,
    tokens_per_block: tl.constexpr,
):
    pid = tl.program_id(0)
    start_idx = pid * tokens_per_block

    # NOTE(yiakwy) : load 1 or few rows of tokens (depends on K), ceil_div(num_tokens / tokens_per_block) blocks; each block cover tokens_per_block*K expert ids
    token_ids_block_ptr = tl.make_block_ptr(
        base=topk_ids_ptr,
        shape=(num_tokens, K),
        block_shape=(tokens_per_block, K,),
        strides=(K, 1,),
        offsets=(start_idx, 0,),
        order=(1, 0,)
    )

    # NOTE(yiakwy) : expert-token-onehot-id mapping (num_tokens, num_experts) into SRAM
    tokens_cnts_ptr = tl.make_block_ptr(
        base=tokens_cnts_ptr,
        shape=(num_tokens, aligned_num_experts),
        block_shape=(tokens_per_block, aligned_num_experts),
        strides=(aligned_num_experts, 1),
        offsets=(start_idx, 0),
        order=(1, 0)
    )

    expert_ids_pos_info_ptr = tl.make_block_ptr(
        base=expert_ids_pos_info_ptr,
        shape=(num_tokens, aligned_num_experts),
        block_shape=(tokens_per_block, aligned_num_experts),
        strides=(aligned_num_experts, 1),
        offsets=(start_idx, 0),
        order=(1, 0)
    )

    expert_ids = tl.load(token_ids_block_ptr)
    token_cnt = tl.load(tokens_cnts_ptr)

    row = tl.arange(0, aligned_num_experts)[None, :]

    # NOTE (yiakwy) : this counts expert occurrencies for each block but lose its position info
    expert_ids_expanded = expert_ids.reshape((tokens_per_block*K, 1))
    expert_ids_one_hot = tl.where(row == expert_ids_expanded, 1, 0)
    expert_ids_one_hot = expert_ids_one_hot.reshape((tokens_per_block, K, aligned_num_experts))

    expert_ids_one_hot_sum = tl.sum(expert_ids_one_hot, 1, keep_dims=True)
    expert_ids_one_hot_sum = expert_ids_one_hot_sum.reshape((tokens_per_block, aligned_num_experts))
    
    token_cnt = expert_ids_one_hot_sum + token_cnt 

    tl.store(tokens_cnts_ptr, token_cnt)

    # NOTE (yiakwy) : this fills out expert position info 
    row_minus = (tl.arange(0, tokens_per_block*K) % K + 1)[None, :]

    col = tl.arange(0, aligned_num_experts)[:, None]
    index = tl.zeros_like(col) + row_minus
    expert_ids = expert_ids.reshape((1, tokens_per_block*K))
    expert_ids_pos_info = tl.where(col == expert_ids, index, 0)

    expert_ids_pos_info = expert_ids_pos_info.reshape((aligned_num_experts, tokens_per_block, K))

    expert_ids_pos_info_sum = tl.sum(expert_ids_pos_info, 2)
    expert_ids_pos_info_sum = expert_ids_pos_info_sum.trans((1, 0))

    tl.store(expert_ids_pos_info_ptr, expert_ids_pos_info_sum)

@triton.jit
def moe_align_block_size_stage2(
    tokens_cnts_ptr,
    num_experts: tl.constexpr,
):
    pid = tl.program_id(0)
    last_cnt = 0
    for i in range(1, num_experts + 1):
        token_cnt = tl.load(tokens_cnts_ptr + i * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + i * num_experts + pid, last_cnt)

@triton.jit
def moe_align_block_size_stage3(
    total_tokens_post_pad_ptr,
    tokens_cnts_ptr,
    cumsum_ptr,
    num_experts: tl.constexpr,
    block_size: tl.constexpr,
):
    last_cumsum = 0
    off_cnt = num_experts * num_experts
    for i in range(1, num_experts + 1):
        token_cnt = tl.load(tokens_cnts_ptr + off_cnt + i - 1)
        last_cumsum = last_cumsum + tl.cdiv(token_cnt, block_size) * block_size
        tl.store(cumsum_ptr + i, last_cumsum)
    tl.store(total_tokens_post_pad_ptr, last_cumsum)


@triton.jit
def moe_align_block_size_stage3_v2(
    tokens_cnts_ptr,
    occurrencies_per_expert_ptr,
    aligned_num_experts: tl.constexpr,
    num_tokens : tl.constexpr,
    num_blocks : tl.constexpr,
    block_size : tl.constexpr,
    tokens_per_block : tl.constexpr,
    REPEAT_TIMES : tl.constexpr,
    stages : tl.constexpr
):
    pid = tl.program_id(0)
    start_idx = pid * tokens_per_block * REPEAT_TIMES
    end_idx = (pid+1) * tokens_per_block * REPEAT_TIMES

    end_idx = tl.minimum(end_idx, num_tokens)

    # NOTE(yiakwy) : expert-token-onehot-id mapping (num_tokens, num_experts) into SRAM
    tokens_cnts_block_ptr = tl.make_block_ptr(
        base=tokens_cnts_ptr,
        shape=(num_tokens, aligned_num_experts),
        block_shape=(tokens_per_block, aligned_num_experts),
        strides=(aligned_num_experts, 1),
        offsets=(start_idx, 0),
        order=(1, 0)
    )

    block_buf = tl.zeros([1, aligned_num_experts,], dtype=tl.int32)
    
    # loop body
    for i in tl.range(start_idx, end_idx - tokens_per_block, tokens_per_block, num_stages=stages):
        tokens_cnts = tl.load(tokens_cnts_block_ptr)

        tokens_cnts_sum = tl.sum(tokens_cnts, 0, keep_dims=True)
        block_buf = block_buf + tokens_cnts_sum

        tokens_cnts_block_ptr = tl.advance(tokens_cnts_block_ptr, (tokens_per_block, 0))

    # loop tail
    if tokens_per_block == 1 or end_idx % tokens_per_block != 0:
        off_r = tl.arange(0, tokens_per_block)[:, None] + i + tokens_per_block
        off_c = tl.arange(0, aligned_num_experts)[None, :]

        mask_r = off_r < end_idx
        mask_c = off_c >= 0

        tokens_cnts = tl.load(tokens_cnts_ptr + off_r * aligned_num_experts + off_c, mask=mask_r + mask_c)

        tokens_cnts_sum = tl.sum(tokens_cnts, 0, keep_dims=True)
        block_buf = block_buf + tokens_cnts_sum

    # NOTE(yiakwy) : no block synchronizaiton barrier available in triton (triton-3.2.0+gite1697f6b)
    tl.atomic_add(occurrencies_per_expert_ptr + tl.arange(0, aligned_num_experts)[None, :], block_buf)

        
@triton.jit
def moe_align_block_size_stage3_1_v2(
    occurrencies_per_expert_ptr,
    offsets_ptr,
    aligned_offsets_ptr,
    unaligned_cumsum_ptr,
    cumsum_ptr,
    num_tokens_post_pad_ptr,
    aligned_num_experts : tl.constexpr,
    num_blocks : tl.constexpr,
    block_size : tl.constexpr
):
    pid = tl.program_id(0)

    occurrencies_per_expert_block_ptr = tl.make_block_ptr(
        base=occurrencies_per_expert_ptr,
        shape=(aligned_num_experts,),
        block_shape=(aligned_num_experts,),
        strides=(1,),
        offsets=(0,),
        order=(0,)
    )

    offsets_block_ptr = tl.make_block_ptr(
        base=offsets_ptr,
        shape=(aligned_num_experts+1,),
        block_shape=(aligned_num_experts,),
        strides=(1,),
        offsets=(1,),
        order=(0,)
    )

    aligned_offsets_block_ptr = tl.make_block_ptr(
        base=aligned_offsets_ptr,
        shape=(aligned_num_experts+1,),
        block_shape=(aligned_num_experts,),
        strides=(1,),
        offsets=(1,),
        order=(0,)
    )

    unaligned_cumsum_block_ptr = tl.make_block_ptr(
        base=unaligned_cumsum_ptr,
        shape=(aligned_num_experts+1,),
        block_shape=(aligned_num_experts,),
        strides=(1,),
        offsets=(1,),
        order=(0,) 
    )

    cumsum_block_ptr = tl.make_block_ptr(
        base=cumsum_ptr,
        shape=(aligned_num_experts+1,),
        block_shape=(aligned_num_experts,),
        strides=(1,),
        offsets=(1,),
        order=(0,)
    )

    ocrr_per_expert = tl.load(occurrencies_per_expert_block_ptr)

    tl.store(offsets_block_ptr, ocrr_per_expert)
    
    cumsum_0 = tl.cumsum(ocrr_per_expert, axis=0)
    tl.store(unaligned_cumsum_block_ptr, cumsum_0)

    ocrr_per_expert = ((ocrr_per_expert + block_size - 1) // block_size) * block_size
    tl.store(aligned_offsets_block_ptr, ocrr_per_expert)

    cumsum = tl.cumsum(ocrr_per_expert, axis=0)
    tl.store(cumsum_block_ptr, cumsum)

    const_minus_1 = tl.arange(aligned_num_experts-1, aligned_num_experts)

    val = tl.gather(cumsum, const_minus_1, axis=0)
    tl.store(num_tokens_post_pad_ptr + tl.arange(0,1), val)


@triton.jit
def moe_align_block_size_stage4(
    topk_ids_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    tokens_cnts_ptr,
    cumsum_ptr,
    num_experts: tl.constexpr,
    block_size: tl.constexpr,
    numel: tl.constexpr,
    tokens_per_thread: tl.constexpr,
):
    pid = tl.program_id(0)
    start_idx = tl.load(cumsum_ptr + pid)
    end_idx = tl.load(cumsum_ptr + pid + 1)

    for i in range(start_idx, end_idx, block_size):
        tl.store(expert_ids_ptr + i // block_size, pid)

    start_idx = pid * tokens_per_thread
    off_t = pid * num_experts

    for i in range(start_idx, tl.minimum(start_idx + tokens_per_thread, numel)):
        expert_id = tl.load(topk_ids_ptr + i)
        token_cnt = tl.load(tokens_cnts_ptr + off_t + expert_id)
        rank_post_pad = token_cnt + tl.load(cumsum_ptr + expert_id)
        tl.store(sorted_token_ids_ptr + rank_post_pad, i)
        tl.store(tokens_cnts_ptr + off_t + expert_id, token_cnt + 1)


# NOTE (yiakwy) : the essentail secrets of the algorithm is that the tranpose of tokens_cnts (num_tokens, num_experts) helps us order the sorted ids
@triton.jit
def moe_align_block_size_stage4_v2(
    expert_ids_ptr,
    expert_ids_info_ptr,
    tokens_cnts_ptr,
    expert_ids_pos_info_ptr,
    cumsum_ptr,
    num_tokens: tl.constexpr,
    K: tl.constexpr,
    alinged_K: tl.constexpr,
    num_experts: tl.constexpr,
    aligned_num_experts: tl.constexpr,
    block_size: tl.constexpr,
    num_tokens_post_pad_const: tl.constexpr
):
    pid = tl.program_id(0)

    expert_id = pid

    tokens_cnts_ptr = tl.make_block_ptr(
        base=tokens_cnts_ptr,
        shape=(num_tokens, aligned_num_experts),
        block_shape=(num_tokens, 1),
        strides=(aligned_num_experts, 1),
        offsets=(0, expert_id),
        order=(1, 0)
    )

    expert_ids_pos_info_ptr = tl.make_block_ptr(
        base=expert_ids_pos_info_ptr,
        shape=(num_tokens, aligned_num_experts),
        block_shape=(num_tokens, 1),
        strides=(aligned_num_experts, 1),
        offsets=(0, expert_id),
        order=(1, 0)
    )

    expert_ids_info_block_ptr = tl.make_block_ptr(
        base=expert_ids_info_ptr,
        shape=(aligned_num_experts, num_tokens),
        block_shape=(1, num_tokens),
        strides=(num_tokens, 1),
        offsets=(expert_id, 0),
        order=(1, 0)
    )

    cumsum_block_ptr = tl.make_block_ptr(
        base=cumsum_ptr,
        shape=(aligned_num_experts,),
        block_shape=(2,),
        strides=(1,),
        offsets=(expert_id, ),
        order=(0,)
    )

    # NOTE (yiakwy) : tl.gather has been supported in Dec 2024 for element accessing. See [PR#5262](https://github.com/triton-lang/triton/pull/5262)
    offsets = tl.load(cumsum_block_ptr)

    const_0 = tl.arange(0, 1)
    const_1 = tl.arange(1, 2)
    var_start_idx = tl.gather(offsets, const_0, axis=0)
    var_end_idx = tl.gather(offsets, const_1, axis=0)

    var_start_idx_1 = var_start_idx // block_size
    var_end_idx_1 = var_end_idx // block_size

    # NOTE(yiakwy) : tl.arange does not support variable (tl.tensor) otherwise constant (tl.constexpr) as input, hence we generate mask
    expert_ids_indices = var_start_idx_1 + tl.arange(0, aligned_num_experts)
    mask = expert_ids_indices < var_end_idx_1

    vals = tl.zeros([aligned_num_experts], tl.int32) + expert_id
    tl.store(expert_ids_ptr + expert_ids_indices, vals, mask=mask)

    tokens_cnts = tl.load(tokens_cnts_ptr)
    expert_ids_pos_info = tl.load(expert_ids_pos_info_ptr)

    token_ids_indices = tl.arange(0, num_tokens)
    col = token_ids_indices[:, None]
    row = expert_ids_indices[None, :]

    token_group_row_ids = tl.where(tokens_cnts, col, float("-inf"))
    token_group_col_ids = tl.where(expert_ids_pos_info > 0, expert_ids_pos_info - 1, float("-inf"))

    token_group_ids = token_group_row_ids * K + token_group_col_ids
    token_group_ids = token_group_ids.reshape((1, num_tokens,))

    tl.store(expert_ids_info_block_ptr, token_group_ids)


@triton.jit
def moe_align_block_size_stage5_v2(
    sorted_token_ids_ptr,
    expert_ids_info_ptr,
    offsets_ptr,
    unaligned_cumsum_ptr,
    cumsum_ptr,
    num_tokens: tl.constexpr,
    num_tokens_post_pad_const: tl.constexpr,
    aligned_num_experts: tl.constexpr
):
    pid = tl.program_id(0)

    expert_id = pid

    unaligned_offsets_cumsum_block_ptr = tl.make_block_ptr(
        base=unaligned_cumsum_ptr,
        shape=(aligned_num_experts,),
        block_shape=(2,),
        strides=(1,),
        offsets=(expert_id, ),
        order=(0,)
    ) 

    aligned_offsets_cumsum_block_ptr = tl.make_block_ptr(
        base=cumsum_ptr,
        shape=(aligned_num_experts,),
        block_shape=(2,),
        strides=(1,),
        offsets=(expert_id, ),
        order=(0,)
    )

    offsets_block_ptr = tl.make_block_ptr(
        base=offsets_ptr,
        shape=(aligned_num_experts,),
        block_shape=(2,),
        strides=(1,),
        offsets=(expert_id, ),
        order=(0,)
    )

    aligned_offsets = tl.load(aligned_offsets_cumsum_block_ptr)
    unaligned_offsets = tl.load(unaligned_offsets_cumsum_block_ptr)

    offsets = tl.load(offsets_block_ptr)

    const_0 = tl.arange(0, 1)
    const_1 = tl.arange(1, 2)

    var_start_idx = tl.gather(aligned_offsets, const_0, axis=0)
    var_start_idx_1 = tl.gather(unaligned_offsets, const_0, axis=0)
    var_num_elements_to_load = tl.gather(offsets, const_0, axis=0)

    var_end_idx = var_start_idx + var_num_elements_to_load
    var_end_idx_1 = var_start_idx_1 + var_num_elements_to_load
    
    off_c = tl.arange(0, num_tokens_post_pad_const)

    sorted_ids_indices = var_start_idx + off_c
    mask = sorted_ids_indices < var_end_idx

    expert_ids_info_indices = var_start_idx_1 + off_c
    mask_1 = expert_ids_info_indices < var_end_idx_1

    expert_ids_info = tl.load(expert_ids_info_ptr + expert_ids_info_indices, mask=mask_1)

    tl.store(sorted_token_ids_ptr + sorted_ids_indices, expert_ids_info, mask=mask)


def moe_align_block_size_triton_v2(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
) -> None:
    # assert len(topk_ids.shape) == 2
    num_tokens, K = topk_ids.shape

    # NOTE(yiakwy) : to meet NV SRAM memory allocator rule : power of 2
    aligned_num_experts = triton.next_power_of_2(num_experts)
    aligned_num_tokens = triton.next_power_of_2(num_tokens)
    aligned_K = triton.next_power_of_2(K)

    # TODO (yiakwy) : this pre-allocation (torch.zeros is slow) overhead will be moved out of the kernel (for compatible API)
    unaligned_cumsum = torch.zeros((aligned_num_experts+1,), dtype=torch.int32, device=topk_ids.device)
    cumsum = torch.zeros((aligned_num_experts+1,), dtype=torch.int32, device=topk_ids.device)

    tokens_cnts = torch.zeros(
        (aligned_num_tokens, aligned_num_experts), dtype=torch.int32, device=topk_ids.device
    )
    # NOTE（yiakwy）: we will use info together with tokens_cnts to compute sorted ids of each expert_id number
    expert_ids_pos_info = torch.zeros(
        (aligned_num_tokens, aligned_num_experts), dtype=torch.int32, device=topk_ids.device
    )
    expert_ids_info = torch.zeros(
        (aligned_num_experts, aligned_num_tokens), dtype=torch.float, device=topk_ids.device
    )

    if not USE_TORCH_OR_TORCH_JIT:
        occurrencies_per_expert = torch.zeros((1, aligned_num_experts), dtype=torch.int32, device=topk_ids.device)
    
    offsets = torch.zeros((aligned_num_experts+1,), dtype=torch.int32, device=topk_ids.device)
    aligned_offsets = torch.zeros((aligned_num_experts+1,), dtype=torch.int32, device=topk_ids.device)

    pad_topk_ids = F.pad(input=topk_ids, pad=(0, aligned_K - K, 0,  aligned_num_tokens - num_tokens), mode='constant', value=aligned_num_experts)

    # NOTE(yiakwy) : at most 256 experts, SM/CU scheduler will distributed blocks evenly to SMs, 2 or 3 block per SM is expected if too much tokens assigned
    tokens_per_block = 1

    # NOTE(yiakwy) : the algoirthm essentially performs a radix sort, so we first prepare occurrences table, then offests with cumsum operation
    this_grid = (ceil_div(num_tokens, tokens_per_block),)
    moe_align_block_size_stage1_v2[this_grid](
        pad_topk_ids, # topk_ids input
        tokens_cnts, # expert_ids one-hot encoding output
        expert_ids_pos_info, # since pad_topk_ids is not sorted in second dimension, we record the index of each expert_id in each row
        aligned_num_experts,
        aligned_num_tokens,
        aligned_K,
        tokens_per_block,
    )

    # NOTE (yiakwy) : compute expert_ids occurencies; the reduction op to compute it is suitable for a different parition layout, so we compute outside stage 1 kernel
    if not USE_TORCH_OR_TORCH_JIT:
        if num_tokens <= 8:
            tokens_per_block = aligned_num_tokens
            repeat_times = 1
        elif num_tokens <= 128:
            tokens_per_block = 4
            repeat_times = 4
        else:
            tokens_per_block = 4
            repeat_times = 8

        stages = 2
        num_blocks = ceil_div(num_tokens, tokens_per_block * repeat_times)
        aligned_num_blocks = triton.next_power_of_2(num_blocks)

        # TODO (yiakwy) : move this pre-allocation out of triton op
        # occurrencies_per_expert_buf = torch.zeros((aligned_num_blocks, aligned_num_experts), dtype=torch.int32, device=topk_ids.device)

        this_grid = (num_blocks,)
        moe_align_block_size_stage3_v2[this_grid](
            tokens_cnts, # input
            occurrencies_per_expert, # output
            aligned_num_experts,
            aligned_num_tokens,
            aligned_num_blocks,
            block_size,
            tokens_per_block,
            repeat_times,
            stages
        )

        # single block for copies
        this_grid = (1,)
        moe_align_block_size_stage3_1_v2[this_grid](
            occurrencies_per_expert, # occurrencies input
            offsets, # output
            aligned_offsets, # output
            unaligned_cumsum, # output
            cumsum, # output
            num_tokens_post_pad, # output
            aligned_num_experts,
            num_blocks,
            block_size
        )

    else:
        occurrencies_per_expert = torch.sum(tokens_cnts, axis=0)

        # NOTE (yiakwy) : cumsum computation along experts (up 256) should use another block level parition
        aligned_offsets[1:] = ((occurrencies_per_expert + block_size - 1) // block_size) * block_size
        cumsum = torch.cumsum(aligned_offsets, axis=0)
        num_tokens_post_pad[:] = cumsum[-1]

        offsets[1:] = occurrencies_per_expert
        unaligned_cumsum = torch.cumsum(offsets, axis=0)

    # NOTE(yiakwy) : this value will be used in triton a constant, so we fetch its value
    num_tokens_post_pad_const = num_tokens_post_pad.item()
    num_tokens_post_pad_const = triton.next_power_of_2(num_tokens_post_pad_const)

    # print("occurrencies_per_expert : ", occurrencies_per_expert)

    # NOTE (yiakwy) : fill expert_ids and prepare for the results of sorted_token_ids
    this_grid = (num_experts,)
    moe_align_block_size_stage4_v2[this_grid](
        expert_ids, # required expert ids output
        expert_ids_info, # index of each expert itermediate output
        tokens_cnts, # one-hot encoding of expert_ids input (aligned_num_tokens, aligned_num_experts)
        expert_ids_pos_info, # index of each expert_id in each row input (aligned_num_tokens, aligned_num_experts)
        cumsum, # aligned offsets of index of expert_ids input
        aligned_num_tokens,
        K,
        aligned_K,
        num_experts,
        aligned_num_experts,
        block_size,
        num_tokens_post_pad_const,
    )

    # print(" expert_ids_info : ", expert_ids_info)
    expert_ids_info = expert_ids_info[expert_ids_info >= 0]
    # print(" expert_ids_info : ", expert_ids_info)

    # NOTE (yiakwy) : fill sorted_token_ids
    moe_align_block_size_stage5_v2[this_grid](
        sorted_token_ids, # required sorted token ids output 
        expert_ids_info, # index of each expert input
        occurrencies_per_expert, # expert_ids occurencies input
        unaligned_cumsum, # offsets of sorted token ids input
        cumsum, # offsets of aligned sorted token ids input
        num_tokens,
        num_tokens_post_pad_const,
        aligned_num_experts
    )

def moe_align_block_size_triton(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
) -> None:
    numel = topk_ids.numel()
    grid = (num_experts,)
    tokens_cnts = torch.zeros(
        (num_experts + 1, num_experts), dtype=torch.int32, device=topk_ids.device
    )
    cumsum = torch.zeros((num_experts + 1,), dtype=torch.int32, device=topk_ids.device)
    tokens_per_thread = ceil_div(numel, num_experts)

    moe_align_block_size_stage1[grid](
        topk_ids,
        tokens_cnts,
        num_experts,
        numel,
        tokens_per_thread,
    )

    moe_align_block_size_stage2[grid](
        tokens_cnts,
        num_experts,
    )

    moe_align_block_size_stage3[(1,)](
        num_tokens_post_pad,
        tokens_cnts,
        cumsum,
        num_experts,
        block_size,
    )

    moe_align_block_size_stage4[grid](
        topk_ids,
        sorted_token_ids,
        expert_ids,
        tokens_cnts,
        cumsum,
        num_experts,
        block_size,
        numel,
        tokens_per_thread,
    )


def calculate_diff(batch_size, seq_len):
    num_experts = 5
    block_size = 3
    # NOTE (yiakwy) : randint will create duplicated expert_id for each token, that is not how moe works
    # topk_ids = torch.randint(
    #     0, num_experts, (batch_size, seq_len), dtype=torch.int32, device="cuda"
    # )
    num_tokens = batch_size
    K = seq_len
    assert K > 0
    num_experts = num_experts if num_experts >= K else K
    topk_ids = torch.stack([torch.randperm(num_experts, dtype=torch.int32, device="cuda")[:K] for _ in range(num_tokens)])

    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    sorted_ids_cuda = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
    )
    sorted_ids_cuda.fill_(topk_ids.numel())
    max_num_m_blocks = max_num_tokens_padded // block_size

    if USE_TRITON_MOE_V2:
        max_num_m_blocks = max(num_experts, max_num_m_blocks)

    expert_ids_cuda = torch.empty(
        (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
    )
    num_tokens_post_pad_cuda = torch.empty(
        (1), dtype=torch.int32, device=topk_ids.device
    )
    token_cnts_buffer = torch.empty(
        (num_experts + 1) * num_experts, dtype=torch.int32, device=topk_ids.device
    )
    cumsum_buffer = torch.empty(
        num_experts + 1, dtype=torch.int32, device=topk_ids.device
    )

    sorted_ids_triton = torch.empty_like(sorted_ids_cuda)
    sorted_ids_triton.fill_(topk_ids.numel())
    expert_ids_triton = torch.empty_like(expert_ids_cuda)
    num_tokens_post_pad_triton = torch.empty_like(num_tokens_post_pad_cuda)

    # compare the performance of cuda and triton implementation
    moe_align_block_size(
        topk_ids,
        num_experts,
        block_size,
        sorted_ids_cuda,
        expert_ids_cuda,
        num_tokens_post_pad_cuda,
        token_cnts_buffer,
        cumsum_buffer,
    )
    if USE_TRITON_MOE_V2:
        moe_align_block_size_triton_v2(
            topk_ids,
            num_experts,
            block_size,
            sorted_ids_triton,
            expert_ids_triton,
            num_tokens_post_pad_triton,
        )
    else:
        moe_align_block_size_triton(
            topk_ids,
            num_experts,
            block_size,
            sorted_ids_triton,
            expert_ids_triton,
            num_tokens_post_pad_triton,
        )

    print("CUDA expert_ids shape:", expert_ids_cuda.shape)
    print("Triton expert_ids shape:", expert_ids_triton.shape)
    if torch.allclose(expert_ids_cuda[:seq_len], expert_ids_triton[:seq_len]) and torch.allclose(
        num_tokens_post_pad_cuda, num_tokens_post_pad_triton
    ):
        print("✅ CUDA and Triton implementations match")
        print("topk_ids:", topk_ids)
        print("sorted_ids_cuda", sorted_ids_cuda)
        print("sorted_ids_triton", sorted_ids_triton)
        print("CUDA expert_ids:", expert_ids_cuda)
        print("Triton expert_ids:", expert_ids_triton)
        print("CUDA num_tokens_post_pad:", num_tokens_post_pad_cuda)
        print("Triton num_tokens_post_pad:", num_tokens_post_pad_triton)
    else:
        print("❌ CUDA and Triton implementations do not match")
        print("topk_ids:", topk_ids)
        print("sorted_ids_cuda", sorted_ids_cuda)
        print("sorted_ids_triton", sorted_ids_triton)
        print("CUDA expert_ids:", expert_ids_cuda)
        print("Triton expert_ids:", expert_ids_triton)
        print("CUDA num_tokens_post_pad:", num_tokens_post_pad_cuda)
        print("Triton num_tokens_post_pad:", num_tokens_post_pad_triton)

        raise Exception("Wrong Value")


batch_size_range = [2**i for i in range(8, 12)] # this can as large as possible
seq_length_range = [2**i for i in range(0, 6)] # we set num_experts 256, so the seqlen (K) must be smaller than this value
configs = list(itertools.product(batch_size_range, seq_length_range))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "seq_len"],
        x_vals=[list(_) for _ in configs],
        line_arg="provider",
        line_vals=["cuda", "triton", "triton_v2"],
        line_names=["CUDA", "Triton", "Triton V2 (tl.gather)"],
        styles=[("blue", "-"), ("red", "-"), ("green", "-")],
        ylabel="us",
        plot_name="moe-align-block-size-performance",
        args={},
    )
)
def benchmark(batch_size, seq_len, provider):
    print("begin test  =====")
    print("batch : ", batch_size)
    print("seq_len : ", seq_len)
    num_experts = 32
    block_size = 3
    # NOTE (yiakwy) : randint will create duplicated expert_id for each token, that is not how moe works
    # topk_ids = torch.randint(
    #     0, num_experts, (batch_size, seq_len), dtype=torch.int32, device="cuda"
    # )
    num_tokens = batch_size
    K = seq_len
    num_experts = num_experts if num_experts >= K else K
    topk_ids = torch.stack([torch.randperm(num_experts, dtype=torch.int32, device="cuda")[:K] for _ in range(num_tokens)])

    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    sorted_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
    )
    sorted_ids.fill_(topk_ids.numel())
    max_num_m_blocks = max_num_tokens_padded // block_size
    expert_ids = torch.empty(
        (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
    )
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)
    token_cnts_buffer = torch.empty(
        (num_experts + 1) * num_experts, dtype=torch.int32, device=topk_ids.device
    )
    cumsum_buffer = torch.empty(
        num_experts + 1, dtype=torch.int32, device=topk_ids.device
    )

    quantiles = [0.5, 0.2, 0.8]
    if provider == "cuda":
        print("start test in cuda ...")
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: moe_align_block_size(
                topk_ids,
                num_experts,
                block_size,
                sorted_ids.clone(),
                expert_ids.clone(),
                num_tokens_post_pad.clone(),
                token_cnts_buffer,
                cumsum_buffer,
            ),
            quantiles=quantiles,
        )
        print("test in cuda finished.")
    elif provider == "triton_v2":
        print("start test in triton v2...")
        triton_op = moe_align_block_size_triton_v2

        try:
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: triton_op(
                    topk_ids,
                    num_experts,
                    block_size,
                    sorted_ids.clone(),
                    expert_ids.clone(),
                    num_tokens_post_pad.clone(),
                ),
                quantiles=quantiles,
            )
            print("triton v2 test finished.")
        except Exception as e:
            print("bs : ", batch_size)
            print("seq_len : ", seq_len)
            calculate_diff(batch_size=batch_size, seq_len=seq_len)
            raise e
        print("triton test finished.")
    else:
        print("start test in triton...")
        triton_op = moe_align_block_size_triton

        try:
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: triton_op(
                    topk_ids,
                    num_experts,
                    block_size,
                    sorted_ids.clone(),
                    expert_ids.clone(),
                    num_tokens_post_pad.clone(),
                ),
                quantiles=quantiles,
            )
            print("triton test finished.")
        except Exception as e:
            print("bs : ", batch_size)
            print("seq_len : ", seq_len)
            # calculate_diff(batch_size=batch_size, seq_len=seq_len)
            raise e
            
    print("===== test ends")
    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


if __name__ == "__main__":
    torch.manual_seed(SEED)

    import numpy as np
    np.random.seed(SEED)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_path",
        type=str,
        default="./configs/benchmark_ops/moe_align_blocks/",
        help="Path to save moe align benchmark results",
    )
    args = parser.parse_args()

    # for i in range(0, 4):
    #     for j in range(0, 8):
    #         try:
    #             calculate_diff(batch_size=2**i, seq_len=2**j)
    #         except:
    #             print("batch : ", 2**i)
    #             print("seq : ", 2**j)

    #             exit(-1)
    # calculate_diff(batch_size=10, seq_len=3)

    benchmark.run(print_data=True, save_path=args.save_path)
