import torch
import triton
import triton.language as tl


@triton.jit
def _quant_int8(val):
    val_min = tl.min(val, 1)
    val_max = tl.max(val, 1)
    scales = 255 / (val_max - val_min)
    zeros = -val_min / scales
    q_val = (val * scales[:, None] + zeros[:, None] + 0.5).to(tl.uint8)
    return q_val, scales, zeros


@triton.jit
def _per_token_quant_int8(
    x_ptr,
    xq_ptr,
    scale_ptr,
    stride_x,
    stride_xq,
    N,
    BLOCK: tl.constexpr,
):
    # Adapted from https://github.com/InternLM/lmdeploy/blob/086481ed84b59bee3b8e4274e5fc69620040c048/lmdeploy/pytorch/kernels/cuda/w8a8_triton_kernels.py#L282
    row_id = tl.program_id(0)

    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(x_ptr + row_id * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    absmax = tl.maximum(tl.max(tl.abs(x)), 1e-10)
    scale_x = absmax / 127
    x_q = x * (127 / absmax)
    x_q = tl.extra.cuda.libdevice.round(x_q).to(tl.int8)

    tl.store(xq_ptr + row_id * stride_xq + cols, x_q, mask=mask)
    tl.store(scale_ptr + row_id, scale_x)


def per_token_quant_int8(x):
    M = x.numel() // x.shape[-1]
    N = x.shape[-1]
    x_q = torch.empty_like(x, device=x.device, dtype=torch.int8)
    scales = torch.empty(x.shape[:-1] + (1,), device=x.device, dtype=torch.float32)
    BLOCK = triton.next_power_of_2(N)
    # heuristics for number of warps
    num_warps = min(max(BLOCK // 256, 1), 8)

    assert x.is_contiguous()
    _per_token_quant_int8[(M,)](
        x,
        x_q,
        scales,
        stride_x=x.stride(-2),
        stride_xq=x_q.stride(-2),
        N=N,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )

    return x_q, scales


@triton.jit
def quantize_and_store(
    cur_index,
    data_ptr,
    cache_ptr,
    scale_zeros_ptr,
    stride_bs,
    stride_h,
    stride_d,
    stride_c_bs,
    stride_c_h,
    stride_c_d,
    dest_index,
    offs_h,
    offs_d,
    head_num,
    szd_off,
):
    data = tl.load(
        data_ptr
        + cur_index * stride_bs
        + offs_h[:, None] * stride_h
        + offs_d[None, :] * stride_d,
        mask=offs_h[:, None] < head_num,
        other=0.0,
    )

    quant, scales, zeros = _quant_int8(data)
    o_ptrs = (
        cache_ptr
        + dest_index * stride_bs
        + offs_h[:, None] * stride_h
        + offs_d[None, :] * stride_d
    )
    sz_ptrs_k = (
        scale_zeros_ptr
        + dest_index * stride_c_bs
        + stride_c_h * offs_h[:, None] * stride_c_d
    )
    tl.store(o_ptrs, quant, mask=offs_h[:, None] < head_num)
    tl.store(
        sz_ptrs_k + szd_off[None, :] * 1,
        scales[:, None],
        mask=(offs_h[:, None] < head_num) & (szd_off[None, :] < 1),
    )
    tl.store(
        sz_ptrs_k + szd_off[None, :] * 1,
        zeros[:, None],
        mask=(offs_h[:, None] < head_num) & (szd_off[None, :] == 1),
    )


@triton.jit
def _fwd_kernel_quantize_cache_kv(
    K,
    V,
    Dest_Idx,
    K_Cache,
    V_Cache,
    K_Scale_Zeros,
    V_Scale_Zeros,
    stride_k_bs,
    stride_k_h,
    stride_k_d,
    stride_v_bs,
    stride_v_h,
    stride_v_d,
    stride_kv_sz_bs,
    stride_kv_sz_h,
    stride_kv_sz_d,
    head_num,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_HEAD: tl.constexpr,
):
    cur_index = tl.program_id(0)
    offs_h = tl.arange(0, BLOCK_HEAD)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    dest_index = tl.load(Dest_Idx + cur_index)
    szd_off = tl.arange(0, 2)

    # Process K
    quantize_and_store(
        cur_index,
        K,
        K_Cache,
        K_Scale_Zeros,
        stride_k_bs,
        stride_k_h,
        stride_k_d,
        stride_kv_sz_bs,
        stride_kv_sz_h,
        stride_kv_sz_d,
        dest_index,
        offs_h,
        offs_d,
        head_num,
        szd_off,
    )

    # Process V
    quantize_and_store(
        cur_index,
        V,
        V_Cache,
        V_Scale_Zeros,
        stride_v_bs,
        stride_v_h,
        stride_v_d,
        stride_kv_sz_bs,
        stride_kv_sz_h,
        stride_kv_sz_d,
        dest_index,
        offs_h,
        offs_d,
        head_num,
        szd_off,
    )


def quantize_cache_kv(
    k,
    v,
    dest_idx,
    k_quantized_out,
    k_scales_zeros,
    v_quantized_out,
    v_scales_zeros,
):
    bs = dest_idx.shape[0]
    k_head_num = k.shape[1]
    k_head_dim = k.shape[2]
    assert (
        k.shape[1] == k_quantized_out.shape[1]
        and k.shape[2] == k_quantized_out.shape[2]
    )
    BLOCK_HEAD = triton.next_power_of_2(k_head_num)
    grid = (bs,)
    num_warps = min(max(k_head_dim // 256, 1), 8)

    _fwd_kernel_quantize_cache_kv[grid](
        k,
        v,
        dest_idx,
        k_quantized_out,
        v_quantized_out,
        k_scales_zeros,
        v_scales_zeros,
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        k_scales_zeros.stride(0),
        k_scales_zeros.stride(1),
        k_scales_zeros.stride(2),
        k_head_num,
        BLOCK_DMODEL=k_head_dim,
        BLOCK_HEAD=BLOCK_HEAD,
        num_warps=num_warps,
        num_stages=1,
    )
    return
