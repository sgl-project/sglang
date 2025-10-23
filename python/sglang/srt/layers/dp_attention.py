from __future__ import annotations

import functools
import logging
from contextlib import contextmanager
from enum import IntEnum, auto
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.srt.distributed import (
    GroupCoordinator,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.utils import get_bool_env_var, is_hip

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

_ATTN_TP_GROUP: Optional[GroupCoordinator] = None
_ATTN_TP_RANK: Optional[int] = None
_ATTN_TP_SIZE: Optional[int] = None
_ATTN_DP_RANK: Optional[int] = None
_ATTN_DP_SIZE: Optional[int] = None
_LOCAL_ATTN_DP_SIZE: Optional[int] = None
_LOCAL_ATTN_DP_RANK: Optional[int] = None
_ENABLE_DP_ATTENTION_FLAG: bool = False

_is_hip = is_hip()
_USE_ROCM700A_WA = _is_hip and get_bool_env_var("SGLANG_USE_ROCM700A")


class DpPaddingMode(IntEnum):

    # Padding tokens to max length and then gather tokens using `all_gather_into_tensor`
    MAX_LEN = auto()
    # Padding tokens to sum length and then gather tokens using `all_reduce`
    SUM_LEN = auto()

    def is_max_len(self):
        return self == DpPaddingMode.MAX_LEN

    def is_sum_len(self):
        return self == DpPaddingMode.SUM_LEN

    @classmethod
    def get_dp_padding_mode(
        cls, is_extend_in_batch, global_num_tokens: List[int]
    ) -> DpPaddingMode:
        if is_extend_in_batch:
            return DpPaddingMode.SUM_LEN

        # we choose the mode that minimizes the communication cost
        max_len = max(global_num_tokens)
        sum_len = sum(global_num_tokens)
        if sum_len * 2 > max_len * get_attention_dp_size():
            return cls.MAX_LEN
        else:
            return cls.SUM_LEN

    @classmethod
    def get_default_mode_in_cuda_graph(cls) -> DpPaddingMode:
        # TODO(kkhuang-amd): noqa, temporary work-around for rocm 7.0.0 alpha
        # it can be safely removed later, once RCCL fixed
        if _USE_ROCM700A_WA:
            return cls.SUM_LEN
        else:
            return cls.MAX_LEN


class _DpGatheredBufferWrapper:

    _hidden_size: int
    _dtype: torch.dtype
    _device: torch.device
    _global_dp_buffer_len: int
    _local_dp_buffer_len: int
    _global_num_tokens: Optional[List[int]]
    _is_extend_in_batch: bool

    @classmethod
    def set_metadata(cls, hidden_size: int, dtype: torch.dtype, device: torch.device):
        cls._hidden_size = hidden_size
        cls._dtype = dtype
        cls._device = device

    @classmethod
    def set_dp_buffer_len(
        cls,
        global_dp_buffer_len: int,
        local_dp_buffer_len: int,
        global_num_tokens: Optional[List[int]] = None,
    ):
        cls._global_dp_buffer_len = global_dp_buffer_len
        cls._local_dp_buffer_len = local_dp_buffer_len
        cls._global_num_tokens = global_num_tokens

    @classmethod
    def get_global_dp_buffer(cls) -> torch.Tensor:
        return torch.empty(
            (cls._global_dp_buffer_len, cls._hidden_size),
            dtype=cls._dtype,
            device=cls._device,
        )

    @classmethod
    def get_local_dp_buffer(cls) -> torch.Tensor:
        return torch.empty(
            (cls._local_dp_buffer_len, cls._hidden_size),
            dtype=cls._dtype,
            device=cls._device,
        )

    @classmethod
    def get_global_dp_buffer_len(cls) -> int:
        return cls._global_dp_buffer_len

    @classmethod
    def get_local_dp_buffer_len(cls) -> int:
        return cls._local_dp_buffer_len

    @classmethod
    def get_dp_global_num_tokens(cls) -> List[int]:
        return cls._global_num_tokens

    @classmethod
    def get_dp_hidden_size(cls) -> int:
        return cls._hidden_size

    @classmethod
    def get_dp_dtype(cls) -> torch.dtype:
        return cls._dtype

    @classmethod
    def get_dp_device(cls) -> torch.device:
        return cls._device

    @classmethod
    def set_is_extend_in_batch(cls, is_extend_in_batch: bool):
        cls._is_extend_in_batch = is_extend_in_batch

    @classmethod
    def get_is_extend_in_batch(cls) -> bool:
        return cls._is_extend_in_batch


def set_dp_buffer_len(
    global_dp_buffer_len: int,
    local_dp_buffer_len: int,
    global_num_tokens: Optional[List[int]] = None,
):
    _DpGatheredBufferWrapper.set_dp_buffer_len(
        global_dp_buffer_len, local_dp_buffer_len, global_num_tokens
    )


def get_global_dp_buffer() -> torch.Tensor:
    return _DpGatheredBufferWrapper.get_global_dp_buffer()


def get_local_dp_buffer() -> torch.Tensor:
    return _DpGatheredBufferWrapper.get_local_dp_buffer()


def get_global_dp_buffer_len() -> int:
    return _DpGatheredBufferWrapper.get_global_dp_buffer_len()


def get_local_dp_buffer_len() -> int:
    return _DpGatheredBufferWrapper.get_local_dp_buffer_len()


def get_dp_global_num_tokens() -> List[int]:
    return _DpGatheredBufferWrapper.get_dp_global_num_tokens()


def get_dp_hidden_size() -> int:
    return _DpGatheredBufferWrapper.get_dp_hidden_size()


def get_dp_dtype() -> torch.dtype:
    return _DpGatheredBufferWrapper.get_dp_dtype()


def get_dp_device() -> torch.device:
    return _DpGatheredBufferWrapper.get_dp_device()


def set_is_extend_in_batch(is_extend_in_batch: bool):
    _DpGatheredBufferWrapper.set_is_extend_in_batch(is_extend_in_batch)


def get_is_extend_in_batch() -> bool:
    return _DpGatheredBufferWrapper.get_is_extend_in_batch()


def compute_dp_attention_world_info(enable_dp_attention, tp_rank, tp_size, dp_size):
    if not enable_dp_attention:
        return tp_rank, tp_size, 0

    attn_tp_size = tp_size // dp_size
    attn_dp_rank = tp_rank // attn_tp_size
    attn_tp_rank = tp_rank % attn_tp_size

    return attn_tp_rank, attn_tp_size, attn_dp_rank


def compute_dp_attention_local_info(
    enable_dp_attention, tp_rank, tp_size, dp_size, moe_dense_tp_size
):
    if not enable_dp_attention:
        return tp_rank, tp_size, 0

    local_tp_size = moe_dense_tp_size if moe_dense_tp_size else tp_size
    local_tp_rank = tp_rank % local_tp_size
    local_dp_size = max(1, dp_size // (tp_size // local_tp_size))

    local_attn_tp_size = local_tp_size // local_dp_size
    local_attn_dp_rank = local_tp_rank // local_attn_tp_size
    local_attn_tp_rank = local_tp_rank % local_attn_tp_size

    return local_attn_tp_rank, local_attn_tp_size, local_attn_dp_rank


def initialize_dp_attention(
    server_args: ServerArgs,
    model_config: ModelConfig,
):
    global _ATTN_TP_GROUP, _ATTN_TP_RANK, _ATTN_TP_SIZE, _ATTN_DP_RANK, _ATTN_DP_SIZE
    global _LOCAL_ATTN_DP_SIZE, _LOCAL_ATTN_DP_RANK, _ENABLE_DP_ATTENTION_FLAG

    from sglang.srt.layers.sampler import SYNC_TOKEN_IDS_ACROSS_TP

    enable_dp_attention = server_args.enable_dp_attention
    tp_size = server_args.tp_size
    dp_size = server_args.dp_size
    moe_dense_tp_size = server_args.moe_dense_tp_size
    pp_size = server_args.pp_size

    tp_rank = get_tensor_model_parallel_rank()

    _ENABLE_DP_ATTENTION_FLAG = enable_dp_attention

    _ATTN_TP_RANK, _ATTN_TP_SIZE, _ATTN_DP_RANK = compute_dp_attention_world_info(
        enable_dp_attention, tp_rank, tp_size, dp_size
    )
    _, _, _LOCAL_ATTN_DP_RANK = compute_dp_attention_local_info(
        enable_dp_attention, tp_rank, tp_size, dp_size, moe_dense_tp_size
    )

    if enable_dp_attention:
        _ATTN_DP_SIZE = dp_size
        if moe_dense_tp_size is None:
            _LOCAL_ATTN_DP_SIZE = _ATTN_DP_SIZE
        else:
            _LOCAL_ATTN_DP_SIZE = max(1, dp_size // (tp_size // moe_dense_tp_size))
    else:
        _ATTN_DP_SIZE = 1
        _LOCAL_ATTN_DP_SIZE = 1

    tp_group = get_tp_group()
    _ATTN_TP_GROUP = GroupCoordinator(
        [
            list(range(head, head + _ATTN_TP_SIZE))
            for head in range(0, pp_size * tp_size, _ATTN_TP_SIZE)
        ],
        tp_group.local_rank,
        torch.distributed.get_backend(tp_group.device_group),
        use_pynccl=SYNC_TOKEN_IDS_ACROSS_TP,
        use_pymscclpp=False,
        use_custom_allreduce=False,
        use_torch_symm_mem=False,
        use_hpu_communicator=False,
        use_xpu_communicator=False,
        use_npu_communicator=False,
        group_name="attention_tp",
    )

    _DpGatheredBufferWrapper.set_metadata(
        hidden_size=model_config.hidden_size,
        dtype=model_config.dtype,
        device=torch.device(server_args.device),
    )


def is_dp_attention_enabled() -> bool:
    return _ENABLE_DP_ATTENTION_FLAG


def get_attention_tp_group() -> GroupCoordinator:
    assert _ATTN_TP_GROUP is not None, "dp attention not initialized!"
    return _ATTN_TP_GROUP


def get_attention_tp_rank() -> int:
    assert _ATTN_TP_RANK is not None, "dp attention not initialized!"
    return _ATTN_TP_RANK


def get_attention_tp_size() -> int:
    assert _ATTN_TP_SIZE is not None, "dp attention not initialized!"
    return _ATTN_TP_SIZE


def get_attention_dp_rank() -> int:
    assert _ATTN_DP_RANK is not None, "dp attention not initialized!"
    return _ATTN_DP_RANK


def get_attention_dp_size() -> int:
    assert _ATTN_DP_SIZE is not None, "dp attention not initialized!"
    return _ATTN_DP_SIZE


def get_local_attention_dp_rank() -> int:
    assert _LOCAL_ATTN_DP_RANK is not None, "dp attention not initialized!"
    return _LOCAL_ATTN_DP_RANK


def get_local_attention_dp_size() -> int:
    assert _LOCAL_ATTN_DP_SIZE is not None, "dp attention not initialized!"
    return _LOCAL_ATTN_DP_SIZE


@contextmanager
def disable_dp_size():
    """Patch the tp group temporarily until this function ends.

    This method is for draft workers of speculative decoding to run draft model
    with different tp degree from that of target model workers.

    Args:
        tp_group (GroupCoordinator): the tp group coordinator
    """
    global _ATTN_DP_SIZE
    assert _ATTN_DP_SIZE is not None, "dp attention not initialized!"

    old_dp_size = _ATTN_DP_SIZE
    _ATTN_DP_SIZE = 1
    try:
        yield
    finally:
        _ATTN_DP_SIZE = old_dp_size


def get_dp_local_info(forward_batch: ForwardBatch) -> Tuple[torch.Tensor, torch.Tensor]:
    # `get_dp_local_info` is only called in global DP gather and scatter. We use global DP rank here.
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
    offset_src: tl.constexpr,
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


def _dp_gather_via_all_reduce(
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

    # Input IDs are in int 32. We should use inplace_all_reduce for local case because of custom all reduce.
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


def _dp_gather_via_all_gather(
    global_tokens: torch.Tensor,
    local_tokens: torch.Tensor,
    forward_batch: ForwardBatch,
    is_partial: bool,
):
    if get_attention_tp_size() == 1:
        get_tp_group().all_gather_into_tensor(global_tokens, local_tokens)
        return

    if not is_partial:
        if get_attention_tp_rank() != 0:
            local_tokens.fill_(0)
    scattered_local_tokens = local_tokens.tensor_split(get_attention_tp_size())[
        get_attention_tp_rank()
    ]
    get_attention_tp_group().reduce_scatter_tensor(scattered_local_tokens, local_tokens)
    get_tp_group().all_gather_into_tensor(global_tokens, scattered_local_tokens)


def _dp_gather(
    global_tokens: torch.Tensor,
    local_tokens: torch.Tensor,
    forward_batch: ForwardBatch,
    is_partial: bool,
):
    if forward_batch.dp_padding_mode.is_max_len():
        _dp_gather_via_all_gather(
            global_tokens, local_tokens, forward_batch, is_partial
        )
    else:
        _dp_gather_via_all_reduce(
            global_tokens, local_tokens, forward_batch, is_partial
        )


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


def dp_reduce_scatter_tensor(output: torch.Tensor, input: torch.Tensor):
    if get_tensor_model_parallel_world_size() == get_attention_dp_size():
        get_tp_group().reduce_scatter_tensor(output, input)
    else:
        scattered_local_tokens = input.tensor_split(
            get_tensor_model_parallel_world_size()
        )[get_tensor_model_parallel_rank()]
        get_tp_group().reduce_scatter_tensor(scattered_local_tokens, input)
        get_attention_tp_group().all_gather_into_tensor(output, scattered_local_tokens)


def attn_tp_reduce_scatter_tensor(output: torch.Tensor, input: torch.Tensor):
    return get_attention_tp_group().reduce_scatter_tensor(output, input)


def attn_tp_all_gather_into_tensor(output: torch.Tensor, input: torch.Tensor):
    return get_attention_tp_group().all_gather_into_tensor(output, input)


def attn_tp_all_gather(output_list: List[torch.Tensor], input: torch.Tensor):
    return get_attention_tp_group().all_gather(input, output_tensor_list=output_list)
