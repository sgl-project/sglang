import torch
from sgl_kernel.ops._kernels import all_reduce as _all_reduce
from sgl_kernel.ops._kernels import dispose as _dispose
from sgl_kernel.ops._kernels import init_custom_ar as _init_custom_ar
from sgl_kernel.ops._kernels import int8_scaled_mm as _int8_scaled_mm
from sgl_kernel.ops._kernels import moe_align_block_size as _moe_align_block_size
from sgl_kernel.ops._kernels import reduce as _reduce


def warp_reduce(input_tensor):
    return _reduce(input_tensor)


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


def int8_scaled_mm(mat_a, mat_b, scales_a, scales_b, out_dtype, bias):
    # if bias is not None:
    #     bias = bias.unsqueeze(0).expand(out.shape[0], out.shape[1]).contiguous()
    M, N = mat_a.shape[0], mat_b.shape[1]
    out = torch.empty((M, N), dtype=out_dtype, device=mat_a.device)
    _int8_scaled_mm(
        out,
        mat_a,
        mat_b,
        scales_a,
        scales_b,
        bias,
    )
    return out
