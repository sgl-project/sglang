from sgl_kernel.ops._kernels import all_reduce as _all_reduce
from sgl_kernel.ops._kernels import (
    batched_rotary_embedding as _batched_rotary_embedding,
)
from sgl_kernel.ops._kernels import dispose as _dispose
from sgl_kernel.ops._kernels import (
    get_graph_buffer_ipc_meta as _get_graph_buffer_ipc_meta,
)
from sgl_kernel.ops._kernels import init_custom_ar as _init_custom_ar
from sgl_kernel.ops._kernels import int8_scaled_mm as _int8_scaled_mm
from sgl_kernel.ops._kernels import moe_align_block_size as _moe_align_block_size
from sgl_kernel.ops._kernels import register_graph_buffers as _register_graph_buffers
from sgl_kernel.ops._kernels import rotary_embedding as _rotary_embedding
from sgl_kernel.ops._kernels import (
    sampling_scaling_penalties as _sampling_scaling_penalties,
)


def init_custom_reduce(
    rank_id, num_devices, rank_data, buffers, tmp_buffers, barrier_in, barrier_out
):
    return _init_custom_ar(
        rank_id, num_devices, rank_data, buffers, tmp_buffers, barrier_in, barrier_out
    )


def custom_dispose(fa):
    _dispose(fa)


def custom_reduce(fa, inp, out):
    _all_reduce(fa, inp, out)


def get_graph_buffer_ipc_meta(fa):
    return _get_graph_buffer_ipc_meta(fa)


def register_graph_buffers(fa, handles, offsets):
    _register_graph_buffers(fa, handles, offsets)


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


def sampling_scaling_penalties(logits, scaling_penalties):
    return _sampling_scaling_penalties(logits, scaling_penalties)


def int8_scaled_mm(mat_a, mat_b, scales_a, scales_b, out_dtype, bias=None):
    return _int8_scaled_mm(
        mat_a,
        mat_b,
        scales_a,
        scales_b,
        out_dtype,
        bias,
    )


def rotary_embedding(positions, query, key, head_size, cos_sin_cache, is_neox):
    return _rotary_embedding(positions, query, key, head_size, cos_sin_cache, is_neox)


def batched_rotary_embedding(
    positions,
    query,
    key,
    head_size,
    cos_sin_cache,
    is_neox,
    rot_dim,
    cos_sin_cache_offsets,
):
    return _batched_rotary_embedding(
        positions,
        query,
        key,
        head_size,
        cos_sin_cache,
        is_neox,
        rot_dim,
        cos_sin_cache_offsets,
    )
