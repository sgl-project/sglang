import logging
from typing import List, Optional

import torch
import triton
import triton.language as tl

from sglang.srt.distributed import get_tensor_model_parallel_rank
from sglang.srt.layers.quantization.fp8_kernel import per_token_group_quant_fp8
from sglang.srt.utils import is_cuda

_is_cuda = is_cuda()
if _is_cuda:
    from sglang.srt.layers.quantization.fp8_kernel import (
        sglang_per_token_group_quant_fp8,
    )
logger = logging.getLogger(__name__)


@triton.jit
def deepep_permute_triton_kernel(
    input_ptr,
    gateup_input_ptr,
    src2dst_ptr,
    topk_ids_ptr,
    a1_scales_ptr,
    topk,
    hidden_size,
    BLOCK_SIZE: tl.constexpr,
):
    OutDtype = gateup_input_ptr.dtype.element_ty

    src_idx = tl.program_id(0)
    src2dst_ptr = src2dst_ptr + src_idx * topk
    topk_ids_ptr = topk_ids_ptr + src_idx * topk

    src_ptr = input_ptr + src_idx * hidden_size

    for start_offset in tl.range(0, hidden_size, BLOCK_SIZE):
        offset = start_offset + tl.arange(0, BLOCK_SIZE)
        mask = offset < hidden_size
        in_data = tl.load(src_ptr + offset, mask=mask).to(OutDtype)

        for idx in range(topk):
            dst_idx = tl.load(src2dst_ptr + idx)
            if dst_idx >= 0:
                dst_ptr = gateup_input_ptr + dst_idx * hidden_size
                tl.store(dst_ptr + offset, in_data, mask=mask)


@triton.jit
def deepep_post_reorder_triton_kernel(
    down_output_ptr,
    output_ptr,
    src2dst_ptr,
    topk_ids_ptr,
    topk_weights_ptr,
    topk,
    hidden_size,
    BLOCK_SIZE: tl.constexpr,
):
    InDtype = down_output_ptr.dtype.element_ty

    src_idx = tl.program_id(0)
    src2dst_ptr = src2dst_ptr + src_idx * topk
    topk_ids_ptr = topk_ids_ptr + src_idx * topk
    topk_weights_ptr = topk_weights_ptr + src_idx * topk

    store_ptr = output_ptr + src_idx * hidden_size
    for start_offset in tl.range(0, hidden_size, BLOCK_SIZE):
        offset = start_offset + tl.arange(0, BLOCK_SIZE)
        mask = offset < hidden_size
        sum_vec = tl.zeros([BLOCK_SIZE], dtype=InDtype)
        for idx in range(topk):
            dst_idx = tl.load(src2dst_ptr + idx)
            if dst_idx >= 0:
                weigh_scale = tl.load(topk_weights_ptr + idx).to(InDtype)
                load_ptr = down_output_ptr + dst_idx * hidden_size
                in_data = tl.load(load_ptr + offset, mask=mask)
                sum_vec += in_data * weigh_scale
        tl.store(store_ptr + offset, sum_vec, mask=mask)


@triton.jit
def compute_src2dst_triton_kernel(
    reorder_ids, src2dst, num_toks, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    dst_id = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = dst_id < num_toks
    src_id = tl.load(reorder_ids + dst_id, mask=mask)
    tl.store(src2dst + src_id, dst_id, mask=mask)


@triton.jit
def deepep_compute_src2dst_triton_kernel(
    reorder_ids, src2dst, num_toks, num_minus_one, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    dst_id = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = dst_id < num_toks
    src_id = tl.load(reorder_ids + dst_id, mask=mask)
    num_invalid = tl.load(num_minus_one)
    tl.store(src2dst + src_id, dst_id - num_invalid, mask=mask)


def deepep_run_moe_deep_preprocess(topk_ids: torch.Tensor, num_experts: int):
    reorder_topk_ids, reorder_ids = torch.sort(topk_ids.view(-1), stable=True)
    seg_indptr = torch.empty(num_experts + 1, device=topk_ids.device, dtype=torch.int64)
    src2dst = torch.empty(topk_ids.numel(), device=topk_ids.device, dtype=torch.int64)

    # Find offet
    expert_ids = torch.arange(
        num_experts + 1, device=topk_ids.device, dtype=reorder_topk_ids.dtype
    )
    torch.searchsorted(reorder_topk_ids, expert_ids, out=seg_indptr)
    num_minus_one = seg_indptr[0]
    seg_indptr = seg_indptr - num_minus_one

    BLOCK_SIZE = 512
    grid = (triton.cdiv(topk_ids.numel(), BLOCK_SIZE),)
    deepep_compute_src2dst_triton_kernel[grid](
        reorder_ids, src2dst, topk_ids.numel(), num_minus_one, BLOCK_SIZE
    )
    reorder_topk_ids = reorder_topk_ids[num_minus_one:]
    return reorder_topk_ids, src2dst, seg_indptr


@triton.jit
def compute_seg_indptr_triton_kernel(reorder_topk_ids, seg_indptr, num_toks):
    expert = tl.program_id(0)
    low = 0
    high = num_toks - 1
    target_location = -1
    while low <= high:
        mid = (low + high) // 2

        if tl.load(reorder_topk_ids + mid) > expert:
            high = mid - 1
        else:
            low = mid + 1
            target_location = mid
    tl.store(seg_indptr + expert + 1, target_location + 1)


def run_moe_ep_preproess(topk_ids: torch.Tensor, num_experts: int):
    reorder_topk_ids, reorder_ids = torch.sort(topk_ids.view(-1), stable=True)
    seg_indptr = torch.zeros(num_experts + 1, device=topk_ids.device, dtype=torch.int64)
    src2dst = torch.empty(topk_ids.numel(), device=topk_ids.device, dtype=torch.int32)

    compute_seg_indptr_triton_kernel[(num_experts,)](
        reorder_topk_ids, seg_indptr, topk_ids.numel()
    )

    BLOCK_SIZE = 512
    grid = (triton.cdiv(topk_ids.numel(), BLOCK_SIZE),)
    compute_src2dst_triton_kernel[grid](
        reorder_ids, src2dst, topk_ids.numel(), BLOCK_SIZE
    )
    return reorder_topk_ids, src2dst, seg_indptr


@triton.jit
def pre_reorder_triton_kernel(
    input_ptr,
    gateup_input_ptr,
    src2dst_ptr,
    topk_ids_ptr,
    a1_scales_ptr,
    start_expert_id,
    end_expert_id,
    topk,
    hidden_size,
    BLOCK_SIZE: tl.constexpr,
):
    OutDtype = gateup_input_ptr.dtype.element_ty

    src_idx = tl.program_id(0)
    src2dst_ptr = src2dst_ptr + src_idx * topk
    topk_ids_ptr = topk_ids_ptr + src_idx * topk

    src_ptr = input_ptr + src_idx * hidden_size
    for idx in range(topk):
        expert_id = tl.load(topk_ids_ptr + idx)
        if expert_id >= start_expert_id and expert_id <= end_expert_id:
            if a1_scales_ptr is not None:
                scale = 1.0 / tl.load(a1_scales_ptr + expert_id - start_expert_id)
            else:
                scale = 1.0

            dst_idx = tl.load(src2dst_ptr + idx)
            dst_ptr = gateup_input_ptr + dst_idx * hidden_size
            for start_offset in tl.range(0, hidden_size, BLOCK_SIZE):
                offset = start_offset + tl.arange(0, BLOCK_SIZE)
                mask = offset < hidden_size
                in_data = tl.load(src_ptr + offset, mask=mask).to(tl.float32)
                out_data = (in_data * scale).to(OutDtype)
                tl.store(dst_ptr + offset, out_data, mask=mask)


@triton.jit
def silu_and_mul_triton_kernel(
    gateup_output,
    down_input,
    hidden_size,
    reorder_topk_ids,
    scales,
    start_expert_id,
    end_expert_id,
    BLOCK_SIZE: tl.constexpr,
):
    InDtype = gateup_output.dtype.element_ty
    OutDtype = down_input.dtype.element_ty

    half_hidden_size = hidden_size // 2

    pid = tl.program_id(0)
    expert_id = tl.load(reorder_topk_ids + pid)
    if expert_id >= start_expert_id and expert_id <= end_expert_id:
        gateup_output_ptr = gateup_output + pid * hidden_size
        gate_output_ptr = gateup_output_ptr
        up_output_ptr = gateup_output_ptr + half_hidden_size
        down_input_ptr = down_input + pid * half_hidden_size

        if scales is not None:
            scale = tl.load(scales + expert_id - start_expert_id)
            scale = (1 / scale).to(InDtype)
        else:
            scale = 1

        for start_offset in tl.range(0, half_hidden_size, BLOCK_SIZE):
            offset = start_offset + tl.arange(0, BLOCK_SIZE)
            mask = offset < half_hidden_size

            gate_output = tl.load(gate_output_ptr + offset, mask=mask).to(tl.float32)
            up_output = tl.load(up_output_ptr + offset, mask=mask)

            # silu & mul & quantize
            gate_output = gate_output * tl.sigmoid(gate_output)
            gate_output = gate_output.to(InDtype)

            silu_mul_output = gate_output * up_output * scale
            silu_mul_output = silu_mul_output.to(OutDtype)
            tl.store(down_input_ptr + offset, silu_mul_output, mask=mask)


# copy from https://github.com/ModelTC/lightllm/blob/a000ab69098654df4731f5b12587dd4e7f0a4f41/lightllm/common/fused_moe/moe_silu_and_mul_mix_quant_ep.py
@triton.jit
def _silu_and_mul_post_quant_kernel(
    input_ptr,
    stride_input_0,
    stride_input_1,
    stride_input_2,
    output_ptr,
    stride_output_0,
    stride_output_1,
    stride_output_2,
    output_scale_ptr,
    stride_output_scale_0,
    stride_output_scale_1,
    stride_output_scale_2,
    masked_m_ptr,
    size_n,
    fp8_max,
    fp8_min,
    BLOCK_N: tl.constexpr,
    NUM_STAGE: tl.constexpr,
):
    expert_id = tl.program_id(2)
    token_id = tl.program_id(1)
    hidden_dim_block_index = tl.program_id(0)

    block_num_per_expert = tl.num_programs(1)

    token_num_cur_expert = tl.load(masked_m_ptr + expert_id)

    stride_input_0 = tl.cast(stride_input_0, dtype=tl.int64)
    stride_output_0 = tl.cast(stride_output_0, dtype=tl.int64)
    stride_input_1 = tl.cast(stride_input_1, dtype=tl.int64)
    stride_output_1 = tl.cast(stride_output_1, dtype=tl.int64)

    offs_in_d = hidden_dim_block_index * BLOCK_N + tl.arange(0, BLOCK_N)
    input_ptr_offs = input_ptr + expert_id * stride_input_0 + offs_in_d
    output_ptr_offs = output_ptr + expert_id * stride_output_0 + offs_in_d
    output_scale_offs = (
        output_scale_ptr
        + expert_id * stride_output_scale_0
        + hidden_dim_block_index * stride_output_scale_2
    )

    for token_index in tl.range(
        token_id, token_num_cur_expert, block_num_per_expert, num_stages=NUM_STAGE
    ):
        gate = tl.load(
            input_ptr_offs + token_index * stride_input_1,
            mask=offs_in_d < size_n,
            other=0.0,
        ).to(tl.float32)
        up = tl.load(
            input_ptr_offs + token_index * stride_input_1 + size_n,
            mask=offs_in_d < size_n,
            other=0.0,
        )
        gate = gate / (1 + tl.exp(-gate))
        gate = gate.to(input_ptr.dtype.element_ty)
        gate_up = up * gate
        _absmax = tl.maximum(tl.max(tl.abs(gate_up)), 1e-10)
        output_s = _absmax / fp8_max
        output_q = tl.clamp(gate_up / output_s, fp8_min, fp8_max).to(
            output_ptr.dtype.element_ty
        )
        tl.store(
            output_ptr_offs + token_index * stride_output_1,
            output_q,
            mask=offs_in_d < size_n,
        )
        tl.store(
            output_scale_offs + token_index * stride_output_scale_1,
            output_s,
        )


def silu_and_mul_masked_post_quant_fwd(
    input: torch.Tensor,
    output: torch.Tensor,
    output_scale: torch.Tensor,
    quant_group_size: int,
    masked_m: torch.Tensor,
):
    """
    input shape [expert_num, token_num_padded, hidden_dim]
    output shape [expert_num, token_num_padded, hidden_dim // 2], dtype fp8
    output_scale [expert_num token_num_paddded, hidden_dim // 2 // 128] dtype float32
    quant_group_size  int,
    masked_m shape [expert_num],
    """

    assert input.is_contiguous()
    assert output.dtype == torch.float8_e4m3fn
    assert output.is_contiguous()
    assert len(input.shape) == 3
    assert input.shape[0] == masked_m.shape[0]
    assert input.shape[-1] % 2 == 0

    size_n = input.shape[-1] // 2
    assert size_n % quant_group_size == 0

    expert_num = len(masked_m)

    if expert_num < 4:
        BLOCK_NUM_PER_EXPERT = 64
    else:
        BLOCK_NUM_PER_EXPERT = 32

    BLOCK_N = quant_group_size
    num_warps = 1
    NUM_STAGES = 6
    hidden_dim_split_block_num = triton.cdiv(size_n, BLOCK_N)
    assert BLOCK_N % quant_group_size == 0

    grid = (
        hidden_dim_split_block_num,
        BLOCK_NUM_PER_EXPERT,
        expert_num,
    )

    finfo = torch.finfo(torch.float8_e4m3fn)
    fp8_max = finfo.max
    fp8_min = -fp8_max

    _silu_and_mul_post_quant_kernel[grid](
        input,
        *input.stride(),
        output,
        *output.stride(),
        output_scale,
        *output_scale.stride(),
        masked_m,
        size_n,
        fp8_max,
        fp8_min,
        BLOCK_N=BLOCK_N,
        NUM_STAGE=NUM_STAGES,
        num_warps=num_warps,
    )
    return


@triton.jit
def tanh(x):
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def gelu_and_mul_triton_kernel(
    gateup_output,
    down_input,
    hidden_size,
    reorder_topk_ids,
    scales,
    start_expert_id,
    end_expert_id,
    BLOCK_SIZE: tl.constexpr,
):
    InDtype = gateup_output.dtype.element_ty
    OutDtype = down_input.dtype.element_ty

    half_hidden_size = hidden_size // 2

    pid = tl.program_id(0)
    expert_id = tl.load(reorder_topk_ids + pid)
    if expert_id >= start_expert_id and expert_id <= end_expert_id:
        gateup_output_ptr = gateup_output + pid * hidden_size
        gate_output_ptr = gateup_output_ptr
        up_output_ptr = gateup_output_ptr + half_hidden_size
        down_input_ptr = down_input + pid * half_hidden_size

        if scales is not None:
            scale = tl.load(scales + expert_id - start_expert_id)
            scale = (1 / scale).to(InDtype)
        else:
            scale = 1

        for start_offset in tl.range(0, half_hidden_size, BLOCK_SIZE):
            offset = start_offset + tl.arange(0, BLOCK_SIZE)
            mask = offset < half_hidden_size

            gate_output = tl.load(gate_output_ptr + offset, mask=mask).to(tl.float32)
            up_output = tl.load(up_output_ptr + offset, mask=mask)

            # gelu & mul & quantize
            # https://pytorch.org/docs/stable/generated/torch.nn.GELU.html
            # sqrt(2/pi)
            kAlpha = 0.7978845608028654
            gate_output = (
                0.5
                * gate_output
                * (
                    1
                    + tanh(
                        kAlpha
                        * (
                            gate_output
                            + 0.044715 * gate_output * gate_output * gate_output
                        )
                    )
                )
            )
            gate_output = gate_output.to(InDtype)

            gelu_mul_output = gate_output * up_output * scale
            gelu_mul_output = gelu_mul_output.to(OutDtype)
            tl.store(down_input_ptr + offset, gelu_mul_output, mask=mask)


@triton.jit
def post_reorder_triton_kernel(
    down_output_ptr,
    output_ptr,
    src2dst_ptr,
    topk_ids_ptr,
    topk_weights_ptr,
    start_expert_id,
    end_expert_id,
    topk,
    hidden_size,
    BLOCK_SIZE: tl.constexpr,
):
    InDtype = down_output_ptr.dtype.element_ty

    src_idx = tl.program_id(0)
    src2dst_ptr = src2dst_ptr + src_idx * topk
    topk_ids_ptr = topk_ids_ptr + src_idx * topk
    topk_weights_ptr = topk_weights_ptr + src_idx * topk

    computed = False
    store_ptr = output_ptr + src_idx * hidden_size
    for start_offset in tl.range(0, hidden_size, BLOCK_SIZE):
        offset = start_offset + tl.arange(0, BLOCK_SIZE)
        mask = offset < hidden_size

        sum_vec = tl.zeros([BLOCK_SIZE], dtype=InDtype)
        for idx in range(topk):
            expert_id = tl.load(topk_ids_ptr + idx)
            if expert_id >= start_expert_id and expert_id <= end_expert_id:
                computed = True
                dst_idx = tl.load(src2dst_ptr + idx)
                weigh_scale = tl.load(topk_weights_ptr + idx).to(InDtype)
                load_ptr = down_output_ptr + dst_idx * hidden_size
                in_data = tl.load(load_ptr + offset, mask=mask)
                sum_vec += in_data * weigh_scale
        tl.store(store_ptr + offset, sum_vec, mask=mask)

    if computed == False:
        for start_offset in tl.range(0, hidden_size, BLOCK_SIZE):
            offset = start_offset + tl.arange(0, BLOCK_SIZE)
            mask = offset < hidden_size
            tl.store(
                store_ptr + offset, tl.zeros([BLOCK_SIZE], dtype=InDtype), mask=mask
            )


@triton.jit
def compute_m_range(
    pid,
    batch_size,
    seg_indptr,
    weight_indices,
    m_num_tiles_indptr,
    BLOCK_SIZE_M: tl.constexpr,
):
    idx = 0
    for bs in range(batch_size):
        tiles = tl.load(m_num_tiles_indptr + bs)
        if pid >= tiles:
            idx = bs

    idx_start = tl.load(m_num_tiles_indptr + idx)

    m_range_start = tl.load(seg_indptr + idx) + (pid - idx_start) * BLOCK_SIZE_M
    m_range_end = min(tl.load(seg_indptr + idx + 1), m_range_start + BLOCK_SIZE_M)
    expert_id = tl.load(weight_indices + idx)
    return m_range_start, m_range_end, expert_id


@triton.jit
def grouped_gemm_triton_kernel(
    a,
    b,
    c,
    batch_size,
    N,
    K,
    seg_indptr,
    weight_indices,
    m_num_tiles_indptr,
    scale_a,
    scale_b,
    use_fp8_w8a8: tl.constexpr,
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    a_stride_0: tl.constexpr,
    b_stride_0: tl.constexpr,
    b_stride_1: tl.constexpr,
    as_stride_0: tl.constexpr,
    as_stride_1: tl.constexpr,
    bs_stride_0: tl.constexpr,
    bs_stride_2: tl.constexpr,
    bs_stride_1: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    c_dtype = c.dtype.element_ty

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    total_m_block = tl.load(m_num_tiles_indptr + batch_size)
    if pid_m >= total_m_block:
        return

    m_range_start, m_range_end, expert_id = compute_m_range(
        pid_m, batch_size, seg_indptr, weight_indices, m_num_tiles_indptr, BLOCK_SIZE_M
    )
    if m_range_end - m_range_start == 0:
        return

    n_range_start = pid_n * BLOCK_SIZE_N
    n_range_end = min(n_range_start + BLOCK_SIZE_N, N)

    offs_am = tl.arange(0, BLOCK_SIZE_M)
    offs_bn = tl.arange(0, BLOCK_SIZE_N)

    offs_am = tl.where(offs_am < m_range_end - m_range_start, offs_am, 0)
    offs_bn = tl.where(offs_bn < n_range_end - n_range_start, offs_bn, 0)
    offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptr = a + (m_range_start + offs_am[:, None]) * a_stride_0 + offs_k[None, :]
    b_ptr = b + (
        (expert_id * b_stride_0)
        + (n_range_start + offs_bn[:, None]) * b_stride_1
        + offs_k[None, :]
    )

    if group_k > 0 and group_n > 0:
        a_scale_ptrs = scale_a + (m_range_start + offs_am[:, None]) * as_stride_0
        offs_bsn = (n_range_start + offs_bn) // group_n
        b_scale_ptrs = scale_b + (expert_id * bs_stride_0) + offs_bsn * bs_stride_1

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a_tile = tl.load(
            a_ptr, mask=offs_k[None, :] < (K - k * BLOCK_SIZE_K), other=0.0
        )
        b_tile = tl.load(
            b_ptr, mask=offs_k[None, :] < (K - k * BLOCK_SIZE_K), other=0.0
        )

        if group_k > 0 and group_n > 0:
            k_start = k * BLOCK_SIZE_K
            offs_ks = k_start // group_k
            a_scale = tl.load(a_scale_ptrs + offs_ks * as_stride_1)
            b_scale = tl.load(b_scale_ptrs + offs_ks * bs_stride_2)
            accumulator += tl.dot(a_tile, b_tile.T) * a_scale * b_scale[None, :]
        else:
            accumulator = tl.dot(a_tile, b_tile.T, accumulator)
        a_ptr += BLOCK_SIZE_K
        b_ptr += BLOCK_SIZE_K

    if use_fp8_w8a8 and not (group_k > 0 and group_n > 0):
        scale_a_value = tl.load(scale_a + expert_id)
        scale_b_value = tl.load(scale_b + expert_id)
        accumulator *= scale_a_value * scale_b_value

    c_tile = accumulator.to(c_dtype)

    offs_cm = m_range_start + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = n_range_start + tl.arange(0, BLOCK_SIZE_N)
    c_ptr = c + offs_cm[:, None] * N + offs_cn[None, :]
    c_mask = (offs_cm[:, None] < m_range_end) & (offs_cn[None, :] < n_range_end)
    tl.store(c_ptr, c_tile, mask=c_mask)


@triton.jit
def compute_m_num_tiles_indptr(
    m_num_tiles_indptr, seg_indptr, batch_size: tl.constexpr, BLOCK_SIZE_M: tl.constexpr
):
    for bs in range(batch_size):
        m = tl.load(seg_indptr + bs + 1) - tl.load(seg_indptr + bs)
        cur_num_tiles = tl.cdiv(m, BLOCK_SIZE_M)
        pre_num_tiles = tl.load(m_num_tiles_indptr + bs)
        tl.store(m_num_tiles_indptr + bs + 1, pre_num_tiles + cur_num_tiles)


def grouped_gemm_triton(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    batch_size: int,
    weight_column_major: bool,
    seg_indptr: Optional[torch.Tensor] = None,
    weight_indices: Optional[torch.Tensor] = None,
    use_fp8_w8a8: bool = False,
    scale_a: torch.Tensor = None,
    scale_b: torch.Tensor = None,
    block_shape: Optional[List[int]] = None,
):
    assert weight_column_major == True  # TODO: more
    if use_fp8_w8a8 and block_shape is None:
        assert scale_a is not None and scale_b is not None

    if block_shape is not None:
        assert len(block_shape) == 2
        block_n, block_k = block_shape[0], block_shape[1]
        if _is_cuda:
            a, scale_a = sglang_per_token_group_quant_fp8(a, block_k)
        else:
            a, scale_a = per_token_group_quant_fp8(a, block_k)

        assert triton.cdiv(a.shape[-1], block_k) == scale_a.shape[-1]
        assert triton.cdiv(b.shape[-2], block_n) == scale_b.shape[-2]
        assert triton.cdiv(b.shape[-1], block_k) == scale_b.shape[-1]

    # TODO: adjust config or tune kernel
    # Reduce block size to prevent L40 shared memory overflow.
    config = {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 32,
        "BLOCK_SIZE_K": 128,
    }

    m_num_tiles_indptr = torch.zeros(batch_size + 1, device=a.device, dtype=torch.int64)
    compute_m_num_tiles_indptr[(1,)](
        m_num_tiles_indptr, seg_indptr, batch_size, config["BLOCK_SIZE_M"]
    )

    grid = lambda META: (
        triton.cdiv(a.size(0), META["BLOCK_SIZE_M"]) + batch_size,
        triton.cdiv(b.size(1), META["BLOCK_SIZE_N"]),
    )

    grouped_gemm_triton_kernel[grid](
        a,
        b,
        c,
        batch_size,
        b.size(1),
        b.size(2),
        seg_indptr,
        weight_indices,
        m_num_tiles_indptr,
        scale_a,
        scale_b,
        use_fp8_w8a8,
        0 if block_shape is None else block_shape[0],
        0 if block_shape is None else block_shape[1],
        a.stride(0),
        b.stride(0),
        b.stride(1),
        scale_a.stride(0) if scale_a is not None and scale_a.ndim == 2 else 0,
        scale_a.stride(1) if scale_a is not None and scale_a.ndim == 2 else 0,
        scale_b.stride(0) if scale_b is not None and scale_b.ndim >= 2 else 0,
        scale_b.stride(2) if scale_b is not None and scale_b.ndim == 3 else 0,
        scale_b.stride(1) if scale_b is not None and scale_b.ndim >= 2 else 0,
        **config,
    )
    return c
