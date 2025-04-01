from __future__ import annotations

import functools
import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, List

import torch
import triton
import triton.language as tl

from sglang.srt.distributed import (
    GroupCoordinator,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_reduce,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

_ATTN_TP_GROUP = None
_ATTN_TP_RANK = None
_ATTN_TP_SIZE = None
_DP_RANK = None
_DP_SIZE = None


def compute_dp_attention_world_info(enable_dp_attention, tp_rank, tp_size, dp_size):
    if not enable_dp_attention:
        return tp_rank, tp_size, 0

    attn_tp_size = tp_size // dp_size
    dp_rank = tp_rank // attn_tp_size
    attn_tp_rank = tp_rank % attn_tp_size
    return attn_tp_rank, attn_tp_size, dp_rank


def initialize_dp_attention(
    enable_dp_attention: bool,
    tp_rank: int,
    tp_size: int,
    dp_size: int,
):
    global _ATTN_TP_GROUP, _ATTN_TP_RANK, _ATTN_TP_SIZE, _DP_RANK, _DP_SIZE

    from sglang.srt.layers.sampler import SYNC_TOKEN_IDS_ACROSS_TP

    _ATTN_TP_RANK, _ATTN_TP_SIZE, _DP_RANK = compute_dp_attention_world_info(
        enable_dp_attention, tp_rank, tp_size, dp_size
    )

    if enable_dp_attention:
        _DP_SIZE = dp_size
    else:
        _DP_SIZE = 1

    tp_group = get_tp_group()
    _ATTN_TP_GROUP = GroupCoordinator(
        [
            list(range(head, head + _ATTN_TP_SIZE))
            for head in range(0, tp_size, _ATTN_TP_SIZE)
        ],
        tp_group.local_rank,
        torch.distributed.get_backend(tp_group.device_group),
        SYNC_TOKEN_IDS_ACROSS_TP,
        False,
        False,
        False,
        False,
        group_name="attention_tp",
    )


def get_attention_tp_group():
    assert _ATTN_TP_GROUP is not None, "dp attention not initialized!"
    return _ATTN_TP_GROUP


def get_attention_tp_rank():
    assert _ATTN_TP_RANK is not None, "dp attention not initialized!"
    return _ATTN_TP_RANK


def get_attention_tp_size():
    assert _ATTN_TP_SIZE is not None, "dp attention not initialized!"
    return _ATTN_TP_SIZE


def get_attention_dp_rank():
    assert _DP_RANK is not None, "dp attention not initialized!"
    return _DP_RANK


def get_attention_dp_size():
    assert _DP_SIZE is not None, "dp attention not initialized!"
    return _DP_SIZE


@contextmanager
def disable_dp_size():
    """Patch the tp group temporarily until this function ends.

    This method is for draft workers of speculative decoding to run draft model
    with different tp degree from that of target model workers.

    Args:
        tp_group (GroupCoordinator): the tp group coordinator
    """
    global _DP_SIZE
    assert _DP_SIZE is not None, "dp attention not initialized!"

    old_dp_size = _DP_SIZE
    _DP_SIZE = 1
    try:
        yield
    finally:
        _DP_SIZE = old_dp_size


def get_dp_local_info(forward_batch: ForwardBatch):
    dp_rank = get_attention_dp_rank()

    if forward_batch.dp_local_start_pos is None:
        cumtokens = torch.cumsum(forward_batch.global_num_tokens_gpu, dim=0)
        if dp_rank == 0:
            local_start_pos = torch.zeros_like(cumtokens[0])
        else:
            local_start_pos = cumtokens[dp_rank - 1]
        local_num_tokens = forward_batch.global_num_tokens_gpu[dp_rank]

        forward_batch.dp_local_start_pos = local_start_pos
        forward_batch.dp_local_num_tokens = local_num_tokens

    return forward_batch.dp_local_start_pos, forward_batch.dp_local_num_tokens


@triton.jit
def memcpy_triton_kernel(
    dst_ptr,
    src_ptr,
    offset_ptr,
    sz_ptr,
    offset_src,
    chunk_size,  # multiplied for offset and sz
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0).to(tl.int64)
    offset = tl.load(offset_ptr).to(tl.int64) * chunk_size
    sz = tl.load(sz_ptr).to(tl.int64) * chunk_size

    start_index = pid * BLOCK_SIZE
    offs = tl.arange(0, BLOCK_SIZE)
    mask = start_index + offs < sz

    if offset_src:
        data = tl.load(src_ptr + offset + start_index + offs, mask=mask)
        tl.store(dst_ptr + start_index + offs, data, mask=mask)
    else:
        data = tl.load(src_ptr + start_index + offs, mask=mask)
        tl.store(dst_ptr + offset + start_index + offs, data, mask=mask)


def prod(x):
    return functools.reduce(lambda a, b: a * b, x, 1)


def memcpy_triton(dst, src, dim, offset, sz, offset_src):
    max_size = min(src.numel(), dst.numel())
    assert dim == 0, "dim != 0 unsupported"
    assert src.shape[1:] == dst.shape[1:], "src and dst must have same shape"
    chunk_size = prod(src.shape[1:])
    BLOCK_SIZE = 8192
    grid = (triton.cdiv(max_size, BLOCK_SIZE),)

    memcpy_triton_kernel[grid](dst, src, offset, sz, offset_src, chunk_size, BLOCK_SIZE)


def _dp_gather(
    global_tokens: torch.Tensor,
    local_tokens: torch.Tensor,
    forward_batch: ForwardBatch,
    is_partial: bool,
):
    local_start_pos, local_num_tokens = get_dp_local_info(forward_batch)

    global_tokens.fill_(0)
    assert local_tokens.is_contiguous()
    assert global_tokens.is_contiguous()

    if local_tokens.shape[0] > 0 and (is_partial or get_attention_tp_rank() == 0):
        """# torch compile report error
        assert (
            global_tokens.untyped_storage().data_ptr()
            != local_tokens.untyped_storage().data_ptr()
        ), "aliasing between global_tokens and local_tokens not allowed"
        """
        memcpy_triton(
            global_tokens, local_tokens, 0, local_start_pos, local_num_tokens, False
        )

    # Input IDs are in int 32. We should use inplace_all_reduce for local case becaues of custom all reduce.
    NUM_GPUS_PER_NODE = 8
    if (
        not local_tokens.dtype.is_floating_point
        and get_tensor_model_parallel_world_size() <= NUM_GPUS_PER_NODE
    ):
        torch.ops.sglang.inplace_all_reduce(
            global_tokens, group_name=get_tp_group().unique_name
        )

    else:
        global_tokens[:] = tensor_model_parallel_all_reduce(global_tokens)


def dp_gather_partial(
    global_tokens: torch.Tensor,
    local_tokens: torch.Tensor,
    forward_batch: ForwardBatch,
):
    _dp_gather(global_tokens, local_tokens, forward_batch, is_partial=True)


def dp_gather_replicate(
    global_tokens: torch.Tensor,
    local_tokens: torch.Tensor,
    forward_batch: ForwardBatch,
):
    _dp_gather(global_tokens, local_tokens, forward_batch, is_partial=False)


def dp_scatter(
    local_tokens: torch.Tensor,  # output
    global_tokens: torch.Tensor,  # input
    forward_batch: ForwardBatch,
):
    # local_num_tokens is not necessarily the same as local_tokens.shape[0],
    # since local_tokens may be padded for cuda graph
    local_start_pos, local_num_tokens = get_dp_local_info(forward_batch)

    local_tokens.fill_(0)
    assert local_tokens.is_contiguous()
    assert global_tokens.is_contiguous()
    if local_tokens.shape[0] > 0:
        """# torch compile report error
        assert (
            local_tokens.untyped_storage().data_ptr()
            != global_tokens.untyped_storage().data_ptr()
        ), "aliasing between local_tokens and global_tokens not allowed"
        """
        memcpy_triton(
            local_tokens, global_tokens, 0, local_start_pos, local_num_tokens, True
        )


def tp_reduce_scatter(
    output: torch.Tensor,
    input_list: List[torch.Tensor],
):
    return get_attention_tp_group().reduce_scatter(output, input_list)


def tp_all_gather(output_list: List[torch.Tensor], input_: torch.Tensor):
    return get_attention_tp_group().all_gather(input_, tensor_list=output_list)


def get_dp_mla_tp_to_dp_plan_meta(
    forward_batch: "ForwardBatch",
    rank,
    world_size,
    dp_tp_size,
    group,
    output,
    input,
    tp_head,
    layer_id,
    mla_mask,
):
    if layer_id == 0:
        all_lens = forward_batch.global_num_tokens_gpu.to(torch.int64)

        dp_size = world_size // dp_tp_size  # 8//4=2
        dp_rank = rank // dp_tp_size  # 0,0,1,1,2,2,3,3
        new_seq = all_lens[dp_rank]
        # dp0~3[h0,h1,h2,h3,h4,h5,h6,h7] -> [dp0[h0~3],dp0[h4~7],dp1[h0~3],dp1[h4~7],...]
        dp_tp_rank = rank % dp_tp_size
        # recv from every rank, [world_size,2] 2:block_offset, block_cnt
        output_split_sizes = torch.zeros(
            world_size, dtype=torch.int64, device=all_lens.device
        )
        seq_with_pad = forward_batch.gathered_buffer_tp2dp.shape[0]
        output_split_sizes[dp_tp_rank * dp_size : (dp_tp_rank + 1) * dp_size] = new_seq
        output_split_offsets = torch.zeros(
            world_size, dtype=torch.int64, device=all_lens.device
        )
        if seq_with_pad > 0:
            output_split_offsets[dp_tp_rank * dp_size : (dp_tp_rank + 1) * dp_size] = (
                torch.arange(
                    0,
                    seq_with_pad * dp_size,
                    seq_with_pad,
                    dtype=torch.int64,
                    device=all_lens.device,
                )
            )

        # send to every rank, [world_size] block_cnt
        # mla_mask = torch.zeros(dp_tp_size, dtype=torch.int64, device=all_lens.device)
        # mla_mask[rank // dp_size] = 1  # to dp0~n tp dp_tp_rank
        input_split_sizes = all_lens.unsqueeze(-1) * mla_mask
        input_split_sizes = input_split_sizes.view(-1)

        forward_batch.dp_mla_tp2dp_plan_meta = group.custom_all_to_all_plan(
            output,
            input,
            output_split_sizes * tp_head,
            input_split_sizes * tp_head,
            output_split_offsets=output_split_offsets * tp_head,
        )

    return forward_batch.dp_mla_tp2dp_plan_meta


def all_to_all_tp_to_dp(
    input_tensor: torch.Tensor,
    k_input: torch.Tensor,
    forward_batch: "ForwardBatch",
    rank,
    world_size,
    dp_tp_size,
    group,
    layer_id,
    mla_mask,
):
    # input_tensor: [seq0+seq1+seq2+..., head/8, head_dim] -> [seq0, head/2, head_dim], [seq1, head/2, head_dim]
    if world_size == 1:
        return input_tensor, k_input

    total_seq, local_head, seq_head_dim = input_tensor.shape
    dp_size = world_size // dp_tp_size  # 8//4=2
    new_head = local_head * dp_size  # 16*4=64

    output_tensor = forward_batch.gathered_buffer_tp2dp
    # dp_size, seq+seq_pad, local_head * seq_head_dim
    output_tensor = output_tensor.view(dp_size, -1, local_head * seq_head_dim)

    block_size = seq_head_dim  # 576
    plan_meta = get_dp_mla_tp_to_dp_plan_meta(
        forward_batch,
        rank,
        world_size,
        dp_tp_size,
        group,
        output_tensor.view(-1, block_size),
        input_tensor.view(-1, block_size),
        local_head,
        layer_id,
        mla_mask,
    )

    group.custom_all_to_all(
        output_tensor.view(-1, block_size),
        input_tensor.view(-1, block_size),
        plan_meta,
    )

    output_tensor = (
        output_tensor.transpose(0, 1).contiguous().view(-1, new_head, seq_head_dim)
    )

    local_k_input = torch.empty(
        output_tensor.shape[0],
        *k_input.shape[1:],
        dtype=k_input.dtype,
        device=k_input.device,
    )
    dp_scatter(local_k_input, k_input, forward_batch)
    return output_tensor, local_k_input


def get_dp_mla_dp_to_tp_plan_meta(
    forward_batch: "ForwardBatch",
    rank,
    world_size,
    dp_tp_size,
    group,
    output,
    input,
    tp_head,
    layer_id,
    mla_mask,
):
    if layer_id == 0:
        dp_size = world_size // dp_tp_size  # 8//2==4

        all_lens = forward_batch.global_num_tokens_gpu.to(torch.int64)
        dp_rank = rank // dp_tp_size  # 0,0,1,1,2,2,3,3
        cur_seq = all_lens[dp_rank]

        # [dp0[h0~3],dp0[h4~7],dp1[h0~3],dp1[h4~7],...] -> dp0~3[h0,h1,h2,h3,h4,h5,h6,h7]
        dp_tp_rank = rank % dp_tp_size
        # recv from every rank, [world_size] block_cnt
        # mla_mask = torch.zeros(dp_tp_size, dtype=torch.int64, device=all_lens.device)
        # mla_mask[rank // dp_size] = 1  # to dp0~n tp dp_tp_rank
        output_split_sizes = all_lens.unsqueeze(-1) * mla_mask
        output_split_sizes = output_split_sizes.view(-1)
        # send to every rank, [world_size,2] 2:block_offset, block_cnt
        input_split_sizes = torch.zeros(
            world_size, dtype=torch.int64, device=all_lens.device
        )
        input_split_sizes[dp_tp_rank * dp_size : (dp_tp_rank + 1) * dp_size] = cur_seq
        seq_with_pad = forward_batch.gathered_buffer_tp2dp.shape[0]
        input_split_offsets = torch.zeros(
            world_size, dtype=torch.int64, device=all_lens.device
        )
        if seq_with_pad > 0:
            input_split_offsets[dp_tp_rank * dp_size : (dp_tp_rank + 1) * dp_size] = (
                torch.arange(
                    0,
                    seq_with_pad * dp_size,
                    seq_with_pad,
                    dtype=torch.int64,
                    device=all_lens.device,
                )
            )
        forward_batch.dp_mla_dp2tp_plan_meta = group.custom_all_to_all_plan(
            output,
            input,
            output_split_sizes * tp_head,
            input_split_sizes * tp_head,
            input_split_offsets=input_split_offsets * tp_head,
        )
    return forward_batch.dp_mla_dp2tp_plan_meta


def all_to_all_dp_to_tp(
    input_tensor: torch.Tensor,
    forward_batch: "ForwardBatch",
    rank,
    world_size,
    dp_tp_size,
    group,
    layer_id,
    mla_mask,
):
    if world_size == 1:
        return input_tensor
    # input_tensor: [seq_r0, head/2, 572], [seq_r1, head/2, 572], ... -> [seq_r0+seq_r1+seq_r2+...,head/8, 572]
    _, local_head, seq_head_dim = input_tensor.shape

    dp_size = world_size // dp_tp_size  # 8//2==4
    new_head = local_head // dp_size  # 64//4=16
    # to [dp_size, seq+seq_pad, new_head, seq_head_dim)
    input_tensor = (
        input_tensor.view(-1, dp_size, new_head * seq_head_dim)
        .transpose(0, 1)
        .contiguous()
    )

    output_tensor = forward_batch.gathered_buffer_dp2tp
    output_tensor = output_tensor.view(-1, new_head * seq_head_dim)

    block_size = seq_head_dim  # 512
    plan_meta = get_dp_mla_dp_to_tp_plan_meta(
        forward_batch,
        rank,
        world_size,
        dp_tp_size,
        group,
        output_tensor.view(-1, block_size),
        input_tensor.view(-1, block_size),
        new_head,
        layer_id,
        mla_mask,
    )

    group.custom_all_to_all(
        output_tensor.view(-1, block_size),
        input_tensor.view(-1, block_size),
        plan_meta,
    )
    return output_tensor.view(-1, new_head, seq_head_dim)
