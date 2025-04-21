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
        assert (
            local_tokens.untyped_storage() is not global_tokens.untyped_storage()
        ), "aliasing between global_tokens and local_tokens not allowed"
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
        assert (
            local_tokens.untyped_storage() is not global_tokens.untyped_storage()
        ), "aliasing between local_tokens and global_tokens not allowed"
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
