from .vllm_reduce_cuda import all_reduce as _vllm_all_reduce
from .vllm_reduce_cuda import dispose as _vllm_dispose
from .vllm_reduce_cuda import (
    get_graph_buffer_ipc_meta as _vllm_get_graph_buffer_ipc_meta,
)
from .vllm_reduce_cuda import init_custom_ar as _vllm_init_custom_ar
from .vllm_reduce_cuda import meta_size as _vllm_meta_size
from .vllm_reduce_cuda import register_buffer as _vllm_register_buffer
from .vllm_reduce_cuda import register_graph_buffers as _vllm_register_graph_buffers
from .warp_reduce_cuda import reduce as _reduce


def warp_reduce(input_tensor):
    return _reduce(input_tensor)


def vllm_init_custom_ar(ipc_tensors, rank_data, rank, full_nvlink):
    return _vllm_init_custom_ar(ipc_tensors, rank_data, rank, full_nvlink)


def vllm_all_reduce(fa, inp, out, reg_buffer, reg_buffer_sz_bytes):
    _vllm_all_reduce(fa, inp, out, reg_buffer, reg_buffer_sz_bytes)


def vllm_dispose(fa):
    _vllm_dispose(fa)


def vllm_meta_size():
    return _vllm_meta_size()


def vllm_register_buffer(fa, ipc_tensors):
    return _vllm_register_buffer(fa, ipc_tensors)


def vllm_get_graph_buffer_ipc_meta(fa):
    return _vllm_get_graph_buffer_ipc_meta(fa)


def vllm_register_graph_buffers(fa, handles, offsets):
    _vllm_register_graph_buffers(fa, handles, offsets)
