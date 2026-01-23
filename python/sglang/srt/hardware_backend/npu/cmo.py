import torch

cmo_stream = None

conv1d_stream = None
cache_update_stream = None
# The total number of cores varies across different device types and needs to be adjusted accordingly.
# The default core number here is set to that of the Ascend A3 (24 cores).
CORE_NUM_FOR_CONV1D = 15
CORE_NUM_FOR_CACHE_UPDATE = 9

share_stream = None
routed_stream = None
CORE_NUM_FOR_SHARE = 8
CORE_NUM_FOR_ROUTED = 16


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


def prepare_weight_cache(handle, cache, PREFETCH_MAX_SIZE=1000000000):
    """
    PREFETCH_MAX_SIZE: maximum size (bytes) for each prefetch operation.
    This affects the time spent in prefetch:
        time ≈ PREFETCH_MAX_SIZE / system_bandwidth
    """
    import torch_npu

    stream = get_cmo_stream()
    if stream is None:
        stream = torch.npu.Stream()
        set_cmo_stream(stream)
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        if isinstance(cache, list):
            for weight in cache:
                torch_npu.npu_prefetch(
                    weight,
                    handle,
                    PREFETCH_MAX_SIZE,
                )
        else:
            torch_npu.npu_prefetch(
                cache,
                handle,
                PREFETCH_MAX_SIZE,
            )


def wait_cmo_stream():
    stream = get_cmo_stream()
    if stream is not None:
        cur_stream = torch.npu.current_stream()
        cur_stream.wait_stream(stream)


def get_or_create_cache_update_stream():
    global cache_update_stream
    if cache_update_stream is None:
        cache_update_stream = torch.npu.Stream()
        torch.npu.set_stream_limit(
            cache_update_stream,
            CORE_NUM_FOR_CACHE_UPDATE,
            CORE_NUM_FOR_CACHE_UPDATE * 2,
        )
    return cache_update_stream


def get_or_create_conv1d_stream():
    global conv1d_stream
    if conv1d_stream is None:
        conv1d_stream = torch.npu.Stream()
        torch.npu.set_stream_limit(
            conv1d_stream, CORE_NUM_FOR_CONV1D, CORE_NUM_FOR_CONV1D * 2
        )
    return conv1d_stream


def get_share_stream():
    global share_stream
    return share_stream


def set_share_stream(stream):
    global share_stream
    share_stream = stream
    torch.npu.set_stream_limit(share_stream, CORE_NUM_FOR_SHARE, CORE_NUM_FOR_SHARE * 2)


def get_routed_stream():
    global routed_stream
    return routed_stream


def set_routed_stream(stream):
    global routed_stream
    routed_stream = stream
    torch.npu.set_stream_limit(
        routed_stream, CORE_NUM_FOR_ROUTED, CORE_NUM_FOR_ROUTED * 2
    )


def wait_share_stream():
    stream = get_share_stream()
    if stream is not None:
        cur_stream = torch.npu.current_stream()
        cur_stream.wait_stream(stream)


def wait_routed_stream():
    stream = get_routed_stream()
    if stream is not None:
        cur_stream = torch.npu.current_stream()
        cur_stream.wait_stream(stream)


def shared_expert_on_independent_stream(hidden_states, forward_func):
    stream = get_share_stream()
    if stream is None:
        stream = torch.npu.Stream()
        set_share_stream(stream)
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        shared_output = forward_func(hidden_states)
        return shared_output


def routed_expert_on_independent_stream(hidden_states, topk_output, forward_func):
    stream = get_routed_stream()
    if stream is None:
        stream = torch.npu.Stream()
        set_routed_stream(stream)
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        routed_output = forward_func(hidden_states, topk_output)
        return routed_output
