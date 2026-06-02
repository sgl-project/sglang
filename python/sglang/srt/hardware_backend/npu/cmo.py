from typing import List

import torch

from sglang.srt.utils.custom_op import register_custom_op

cmo_stream = None
share_stream = None


def get_cmo_stream():
    """
    Cache Management Operation(CMO).
    Launch a new stream to prefetch the weight of matmul when running other
    AIV or communication kernels, aiming to overlap the memory access time.
    """
    global cmo_stream
    return cmo_stream


def set_cmo_stream(stream):
    global cmo_stream
    cmo_stream = stream


@register_custom_op()
def _prepare_weight_cache_list(
    handle: torch.Tensor, cache: List[torch.Tensor], max_size: int
) -> None:
    import torch_npu

    stream = get_cmo_stream()
    if stream is None:
        stream = torch.npu.Stream()
        set_cmo_stream(stream)
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for weight in cache:
            torch_npu.npu_prefetch(weight, handle, max_size)


@register_custom_op()
def _prepare_weight_cache_single(
    handle: torch.Tensor, cache: torch.Tensor, max_size: int
) -> None:
    import torch_npu

    stream = get_cmo_stream()
    if stream is None:
        stream = torch.npu.Stream()
        set_cmo_stream(stream)
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        torch_npu.npu_prefetch(cache, handle, max_size)


def prepare_weight_cache(handle, cache, PREFETCH_MAX_SIZE=1000000000):
    """
    PREFETCH_MAX_SIZE: maximum size (bytes) for each prefetch operation.
    This affects the time spent in prefetch:
        time ≈ PREFETCH_MAX_SIZE / system_bandwidth
    """
    if isinstance(cache, list):
        _prepare_weight_cache_list(handle, cache, PREFETCH_MAX_SIZE)
    else:
        _prepare_weight_cache_single(handle, cache, PREFETCH_MAX_SIZE)


@register_custom_op(out_shape=0)
def wait_cmo_stream(dummy: torch.Tensor) -> torch.Tensor:
    stream = get_cmo_stream()
    if stream is not None:
        cur_stream = torch.npu.current_stream()
        cur_stream.wait_stream(stream)
    return dummy


def get_share_stream():
    global share_stream
    return share_stream


def set_share_stream(stream):
    global share_stream
    share_stream = stream


@register_custom_op(out_shape=0)
def wait_share_stream(dummy: torch.Tensor) -> torch.Tensor:
    stream = get_share_stream()
    if stream is not None:
        cur_stream = torch.npu.current_stream()
        cur_stream.wait_stream(stream)
    return dummy


def shared_expert_on_independent_stream(hidden_states, forward_func):
    import torch.compiler

    if torch.compiler.is_compiling():
        return forward_func(hidden_states)

    stream = get_share_stream()
    if stream is None:
        stream = torch.npu.Stream()
        set_share_stream(stream)
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        shared_output = forward_func(hidden_states)
        return shared_output
