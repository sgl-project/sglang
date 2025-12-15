import math
import os

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from .common import _attn_bwd_preprocess, _attn_fwd_gating, configs_gating_preset
from .flash_attn_bsa_varlen_mask import (
    _attn_bwd_dkdv_bsa_varlen_wrapper,
    _attn_bwd_dq_bsa_varlen_align_wrapper,
    _attn_bwd_dq_bsa_varlen_wrapper,
    _attn_fwd_bsa_varlen,
    _attn_fwd_bsa_varlen_align,
    configs_bwd_dkdv_bsa_varlen_preset,
    configs_bwd_dq_bsa_varlen_align_preset,
    configs_bwd_dq_bsa_varlen_preset,
    configs_fwd_bsa_varlen_align_preset,
    configs_fwd_bsa_varlen_preset,
)

torch._dynamo.config.cache_size_limit = 32


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def supports_tma():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9


HAS_TMA_DESC = "nv_tma_desc_type" in dir(tl)

if HAS_TMA_DESC:
    print(
        "TMA benchmarks will be running with experimental grid constant TMA descriptor.",
    )
else:
    print(
        "TMA benchmarks will be running without grid constant TMA descriptor.",
    )


# TmaAutoTuneHelper used in htyu's PR #5622
class TmaAutoTuneHelper:
    # duck typing wrapper to implement the same interface as TmaDescKernelParam in Triton PR #4498
    class KernelParamWrapper:

        def __init__(self, desc):
            self.desc = desc

        def tma_desc_cpu_ptr(self):
            return self.desc.data_ptr()

    TMA_SIZE = 128

    def __init__(self):
        self.fill_1d_tma_descriptor_inner = (
            triton.runtime.driver.active.utils.fill_1d_tma_descriptor
        )
        self.fill_2d_tma_descriptor_inner = (
            triton.runtime.driver.active.utils.fill_2d_tma_descriptor
        )
        if HAS_TMA_DESC:
            self.descriptors = {}
        else:
            self.cuda_descriptors = {}

    # Call this method outside of the lambda function for grid size
    def init_tma_descriptor(self, name):
        if HAS_TMA_DESC:
            self.descriptors[name] = torch.empty(
                TmaAutoTuneHelper.TMA_SIZE, device="cpu", dtype=torch.int8
            )
        else:
            self.cuda_descriptors[name] = torch.empty(
                TmaAutoTuneHelper.TMA_SIZE, device="cuda", dtype=torch.int8
            )

    # Call this method inside the lambda function for grid size
    def fill_1d_tma_descriptor(self, name, ptr, dim, block_dim, element_size):
        if HAS_TMA_DESC:
            desc_x = self.descriptors[name]
            assert desc_x.data_ptr() % 64 == 0
            self.fill_1d_tma_descriptor_inner(
                ptr, dim, block_dim, element_size, desc_x.data_ptr()
            )
        else:
            desc_x = self.cuda_descriptors[name]
            buf_x = torch.empty_like(desc_x, device="cpu", pin_memory=True)
            self.fill_1d_tma_descriptor_inner(
                ptr, dim, block_dim, element_size, buf_x.data_ptr()
            )
            desc_x.copy_(buf_x, non_blocking=True)

    # Call this method inside the lambda function for grid size
    def fill_2d_tma_descriptor(
        self, name, ptr, dim1, dim0, block_dim1, block_dim0, element_size
    ):
        if HAS_TMA_DESC:
            desc_x = self.descriptors[name]
            assert desc_x.data_ptr() % 64 == 0
            self.fill_2d_tma_descriptor_inner(
                ptr, dim1, dim0, block_dim1, block_dim0, element_size, desc_x.data_ptr()
            )
        else:
            desc_x = self.cuda_descriptors[name]
            buf_x = torch.empty_like(desc_x, device="cpu", pin_memory=True)
            self.fill_2d_tma_descriptor_inner(
                ptr, dim1, dim0, block_dim1, block_dim0, element_size, buf_x.data_ptr()
            )
            desc_x.copy_(buf_x, non_blocking=True)

    def get_tma_descriptor_kernel_param(self, name):
        if HAS_TMA_DESC:
            assert self.descriptors[name] is not None
            return self.KernelParamWrapper(self.descriptors[name])
        else:
            assert self.cuda_descriptors[name] is not None
            return self.cuda_descriptors[name]


@triton.jit
def create_mask_from_indices_kernel(
    block_indices,
    block_mask,
    stride_bz,
    stride_bh,
    stride_bm,
    stride_bs,
    stride_mz,
    stride_mh,
    stride_mm,
    stride_mn,
    H,
):
    i_zh, i_m, i_s = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_z, i_h = i_zh // H, i_zh % H

    off_b = (
        i_z.to(tl.int64) * stride_bz
        + i_h.to(tl.int64) * stride_bh
        + i_m.to(tl.int64) * stride_bm
        + i_s.to(tl.int64) * stride_bs
    )

    b_i = tl.load(block_indices + off_b)

    off_m = (
        i_z.to(tl.int64) * stride_mz
        + i_h.to(tl.int64) * stride_mh
        + i_m.to(tl.int64) * stride_mm
        + b_i.to(tl.int64) * stride_mn
    )

    b_m = 1
    tl.store(block_mask + off_m, b_m.to(block_mask.dtype.element_ty))


def create_mask_from_indices_triton(block_indices, N_cols):
    B, H, N_rows, S = block_indices.shape
    block_mask = torch.zeros(
        (B, H, N_rows, N_cols), dtype=torch.bool, device=block_indices.device
    )
    create_mask_from_indices_kernel[(B * H, N_rows, S)](
        block_indices,
        block_mask,
        block_indices.stride(0),
        block_indices.stride(1),
        block_indices.stride(2),
        block_indices.stride(3),
        block_mask.stride(0),
        block_mask.stride(1),
        block_mask.stride(2),
        block_mask.stride(3),
        H,
    )
    return block_mask


@torch.compile
def create_mask_from_indices_varlen(block_indices, N_cols_mask):
    B, H, M, _ = block_indices.shape
    device = block_indices.device

    mask = torch.zeros((B, H, M, N_cols_mask), dtype=torch.bool, device=device)

    valid = block_indices < N_cols_mask

    b_idx = torch.arange(B, device=device)[:, None, None, None].expand_as(block_indices)
    h_idx = torch.arange(H, device=device)[None, :, None, None].expand_as(block_indices)
    m_idx = torch.arange(M, device=device)[None, None, :, None].expand_as(block_indices)

    valid_coords = (b_idx[valid], h_idx[valid], m_idx[valid], block_indices[valid])

    mask[valid_coords] = True

    return mask


@torch.compile
def create_indices_k_from_indices_q_varlen(
    block_indices,
    N_cols_mask,
    # indicate the number of the last dimension of the bool mask, since this information cannot be determined by block_indices, which may contain invalid elements
):
    block_mask_qk = create_mask_from_indices_varlen(block_indices, N_cols_mask)
    B, H, M, N = block_mask_qk.shape
    block_mask_kq = block_mask_qk.permute(0, 1, 3, 2)
    indices = (
        torch.arange(M, device=block_indices.device)
        .view(1, 1, 1, -1)
        .expand_as(block_mask_kq)
    )
    block_indices_k = torch.where(block_mask_kq, indices, M)
    block_indices_k, _ = torch.sort(block_indices_k, dim=-1)

    block_indices_k_lens = (block_indices_k < M).sum(dim=-1)

    return block_indices_k, block_indices_k_lens


@torch.compile
def mean_pooling_compression(x: torch.Tensor, block_size: int) -> torch.Tensor:
    B, H, S = x.shape[:3]
    num_block = math.ceil(S / block_size)
    if S % block_size != 0:
        x = F.pad(x, (0, 0, 0, num_block * block_size - S))
    x_cmp = x.view(B, H, num_block, block_size, -1).mean(dim=3)
    return x_cmp


@torch.compile
def cal_score(q, k):
    k_transposed = k.transpose(-1, -2)  # [b, h, d, s_k]
    score = torch.matmul(q, k_transposed)  # [b, h, s_q, s_k]
    return score


def cal_score_triton(q, k):
    B, H, s_q, D = q.shape
    s_k = k.shape[2]

    score = torch.empty(B, H, s_q, s_k, device=q.device, dtype=q.dtype)

    kernel_config = (
        {}
        if os.environ.get("TRITON_AUTOTUNE_ENBALE", "0") == "1"
        else configs_gating_preset["default"]
    )

    grid = lambda args: (triton.cdiv(s_q, args["BLOCK_M"]), B * H, 1)
    _attn_fwd_gating[grid](
        q,
        k,
        score,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        score.stride(0),
        score.stride(1),
        score.stride(2),
        score.stride(3),
        H,
        s_q,
        s_k,
        HEAD_DIM=D,
        **kernel_config
    )
    return score


@torch.compile
def get_select_indices_topk(q, k, sparsity):
    score = cal_score(q, k)
    block_indices, block_indices_lens = get_select_indices_topk_from_score(
        score, sparsity
    )
    return block_indices, block_indices_lens


@torch.compile
def get_select_indices_topk_from_score(score, sparsity):
    num_selected = int((1 - sparsity) * score.shape[-1])
    block_indices = torch.topk(score, num_selected)[1]

    block_indices_lens = torch.full(
        (block_indices.shape[0], block_indices.shape[1], block_indices.shape[2]),
        num_selected,
        dtype=torch.int32,
        device=block_indices.device,
    )

    return block_indices, block_indices_lens


@torch.compile
def get_select_indices_cdf(q, k, cdf_threshold):
    score = cal_score(q, k)
    head_dim = q.shape[-1]
    block_indices, block_indices_lens = get_select_indices_cdf_from_score(
        score, cdf_threshold, 1 / head_dim**0.5
    )
    return block_indices, block_indices_lens


@torch.compile
def get_select_indices_cdf_from_score(score, cdf_threshold, sm_scale):
    weights = torch.softmax(score * sm_scale, dim=-1)

    B, H, Sq, Sk = weights.shape
    cdf_threshold = (
        torch.full((H,), cdf_threshold, device=weights.device)
        .view(1, H, 1, 1)
        .expand(B, -1, Sq, -1)
    )
    weights_sorted = torch.sort(weights, dim=-1, descending=True)
    cdf = torch.cumsum(weights_sorted.values, dim=-1)
    num_selected = torch.searchsorted(cdf, cdf_threshold, right=True)

    return weights_sorted.indices, num_selected.squeeze(-1)


@torch.compile
def get_select_indices_cdf_topk(q, k, sparsity, cdf_threshold):
    score = cal_score(q, k)
    head_dim = q.shape[-1]
    block_indices, block_indices_lens = get_select_indices_cdf_topk_from_score(
        score, sparsity, cdf_threshold, 1 / head_dim**0.5
    )
    return block_indices, block_indices_lens


@torch.compile
def get_select_indices_cdf_topk_from_score(score, sparsity, cdf_threshold, sm_scale):
    weights = torch.softmax(score * sm_scale, dim=-1)

    B, H, Sq, Sk = weights.shape
    cdf_threshold = (
        torch.full((H,), cdf_threshold, device=weights.device)
        .view(1, H, 1, 1)
        .expand(B, -1, Sq, -1)
    )
    weights_sorted = torch.sort(weights, dim=-1, descending=True)
    cdf = torch.cumsum(weights_sorted.values, dim=-1)
    num_selected = torch.searchsorted(cdf, cdf_threshold, right=True)

    # max(cdf, topk)
    num_selected_topk = int((1 - sparsity) * score.shape[-1])
    num_selected[num_selected < num_selected_topk] = num_selected_topk

    return weights_sorted.indices, num_selected.squeeze(-1)


def get_select_indices(q, k, sparsity, cdf_threshold):
    if sparsity is not None and cdf_threshold is None:
        block_indices, block_indices_lens = get_select_indices_topk(q, k, sparsity)
    elif sparsity is None and cdf_threshold is not None:
        block_indices, block_indices_lens = get_select_indices_cdf(q, k, cdf_threshold)
    elif sparsity is not None and cdf_threshold is not None:
        block_indices, block_indices_lens = get_select_indices_cdf_topk(
            q, k, sparsity, cdf_threshold
        )
    else:
        raise ValueError
    return block_indices, block_indices_lens


def get_select_indices_from_score(score, sparsity, cdf_threshold):
    if sparsity is not None and cdf_threshold is None:
        block_indices, block_indices_lens = get_select_indices_topk_from_score(
            score, sparsity
        )
    elif sparsity is None and cdf_threshold is not None:
        block_indices, block_indices_lens = get_select_indices_cdf_from_score(
            score, cdf_threshold
        )
    elif sparsity is not None and cdf_threshold is not None:
        block_indices, block_indices_lens = get_select_indices_cdf_topk_from_score(
            score, sparsity, cdf_threshold
        )
    else:
        raise ValueError
    return block_indices, block_indices_lens


def attn_fwd_bsa_varlen_triton(
    q,
    k,
    v,
    sm_scale,
    block_indices,
    block_indices_lens,
    chunk_size_q,
    chunk_size_k,
    sparsity,
):
    B, H, Seq, D = q.shape

    o = torch.empty_like(q)
    M = torch.empty(
        (q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32
    )

    grid = lambda args: (
        triton.cdiv(q.shape[2], args["BLOCK_M"]),
        q.shape[0] * q.shape[1],
        1,
    )

    config_key = "BLOCK_N_LG=64" if chunk_size_k == 64 else "default"
    if chunk_size_k > 128:
        fwd_func = _attn_fwd_bsa_varlen
        kernel_config = (
            {}
            if os.environ.get("TRITON_AUTOTUNE_ENBALE", "0") == "1"
            else configs_fwd_bsa_varlen_preset[config_key]
        )
    else:
        fwd_func = _attn_fwd_bsa_varlen_align
        kernel_config = (
            {}
            if os.environ.get("TRITON_AUTOTUNE_ENBALE", "0") == "1"
            else configs_fwd_bsa_varlen_align_preset[config_key]
        )

    block_indices = block_indices.contiguous()
    block_indices_lens = block_indices_lens.contiguous()

    fwd_func[grid](
        q,
        k,
        v,
        sm_scale,
        M,
        o,
        block_indices,  # [B, H, M_COMPRESS, S]
        block_indices_lens,  # [B, H, M_COMPRESS, S_MAX]
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        block_indices.stride(0),
        block_indices.stride(1),
        block_indices.stride(2),
        block_indices.stride(3),
        block_indices_lens.stride(0),
        block_indices_lens.stride(1),
        block_indices_lens.stride(2),
        H,
        Seq,
        D,
        BLOCK_M=chunk_size_q,
        BLOCK_N_LG=chunk_size_k,
        SPARSITY=sparsity,
        **kernel_config
    )

    LN2 = 0.6931471824645996
    lse = M * LN2  # convert back to natural units (M is of base 2)

    return o, lse


def attn_bwd_bsa_varlen_triton(
    do,
    q,
    k,
    v,
    o,
    dq,
    dk,
    dv,
    sm_scale,
    M,
    block_indices,
    block_indices_lens,
    chunk_size_q,
    chunk_size_k,
    sparsity,
):
    RCP_LN2 = 1.4426950408889634
    M = M * RCP_LN2  # ln -> log2

    do = do.contiguous()
    # assert q.stride() == k.stride() == v.stride() == o.stride() == do.stride()

    BATCH, N_HEAD, N_CTX, HEAD_DIM = q.shape
    N_CTX_KV = k.shape[-2]

    RCP_LN2 = 1.4426950408889634  # = 1.0 / ln(2) # reciprocal
    arg_k = k
    arg_k = arg_k * (sm_scale * RCP_LN2)

    if min(chunk_size_q, chunk_size_k) >= 128:
        PRE_BLOCK = 128
    else:
        PRE_BLOCK = min(chunk_size_q, chunk_size_k)

    assert N_CTX % PRE_BLOCK == 0
    pre_grid = (N_CTX // PRE_BLOCK, BATCH * N_HEAD)
    delta = torch.empty_like(M)
    _attn_bwd_preprocess[pre_grid](
        o, do, delta, N_CTX, BLOCK_M=PRE_BLOCK, HEAD_DIM=HEAD_DIM
    )

    block_indices_k, block_indices_k_lens = create_indices_k_from_indices_q_varlen(
        block_indices=block_indices, N_cols_mask=N_CTX_KV // chunk_size_k
    )

    block_indices = block_indices.contiguous()
    block_indices_lens = block_indices_lens.contiguous()
    block_indices_k = block_indices_k.contiguous()
    block_indices_k_lens = block_indices_k_lens.contiguous()

    config_key = "BLOCK_N_DQ_LG=64" if chunk_size_k == 64 else "default"
    kernel_config = (
        {}
        if os.environ.get("TRITON_AUTOTUNE_ENBALE", "0") == "1"
        else configs_bwd_dkdv_bsa_varlen_preset[config_key]
    )

    grid_dkdv = lambda args: (
        triton.cdiv(arg_k.shape[2], args["BLOCK_N"]),
        1,
        arg_k.shape[0] * arg_k.shape[1],
    )
    _attn_bwd_dkdv_bsa_varlen_wrapper[grid_dkdv](
        q,
        arg_k,
        v,
        sm_scale,  # softmax scale
        do,
        dk,
        dv,
        M,  # lse (log2)
        delta,
        block_indices_k,
        block_indices_k_lens,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        dk.stride(0),
        dk.stride(1),
        dk.stride(2),
        dk.stride(3),
        dv.stride(0),
        dv.stride(1),
        dv.stride(2),
        dv.stride(3),
        do.stride(0),
        do.stride(1),
        do.stride(2),
        do.stride(3),
        M.stride(0),
        M.stride(1),
        M.stride(2),
        delta.stride(0),
        delta.stride(1),
        delta.stride(2),
        block_indices_k.stride(0),
        block_indices_k.stride(1),
        block_indices_k.stride(2),
        block_indices_k.stride(3),
        block_indices_k_lens.stride(0),
        block_indices_k_lens.stride(1),
        block_indices_k_lens.stride(2),
        N_HEAD,
        N_CTX,
        BLOCK_M=chunk_size_q,
        BLOCK_N_DQ_LG=chunk_size_k,
        HEAD_DIM=HEAD_DIM,
        SPARSITY=sparsity,
        **kernel_config
    )

    config_key = "BLOCK_N_DQ_LG=64" if chunk_size_k == 64 else "default"
    if chunk_size_k > 128:
        bwd_dq_func = _attn_bwd_dq_bsa_varlen_wrapper
        kernel_config = (
            {}
            if os.environ.get("TRITON_AUTOTUNE_ENBALE", "0") == "1"
            else configs_bwd_dq_bsa_varlen_preset[config_key]
        )
    else:
        bwd_dq_func = _attn_bwd_dq_bsa_varlen_align_wrapper
        kernel_config = (
            {}
            if os.environ.get("TRITON_AUTOTUNE_ENBALE", "0") == "1"
            else configs_bwd_dq_bsa_varlen_align_preset[config_key]
        )

    grid_dq = lambda args: (
        triton.cdiv(q.shape[2], args["BLOCK_M"]),
        1,
        q.shape[0] * q.shape[1],
    )
    bwd_dq_func[grid_dq](
        q,
        arg_k,
        v,
        do,
        dq,
        M,  # lse (log2)
        delta,
        block_indices,
        block_indices_lens,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        dq.stride(0),
        dq.stride(1),
        dq.stride(2),
        dq.stride(3),
        do.stride(0),
        do.stride(1),
        do.stride(2),
        do.stride(3),
        M.stride(0),
        M.stride(1),
        M.stride(2),
        delta.stride(0),
        delta.stride(1),
        delta.stride(2),
        block_indices.stride(0),
        block_indices.stride(1),
        block_indices.stride(2),
        block_indices.stride(3),
        block_indices_lens.stride(0),
        block_indices_lens.stride(1),
        block_indices_lens.stride(2),
        N_HEAD,
        N_CTX,
        BLOCK_M=chunk_size_q,
        BLOCK_N_DQ_LG=chunk_size_k,
        HEAD_DIM=HEAD_DIM,
        SPARSITY=sparsity,
        **kernel_config
    )


@torch.compile
def make_block_indices_varlen_cp_list(block_indices, cp_size, num_blocks_k_full):
    """
    Args:
        block_indices: [B, H, num_blocks_q_per_cp_rank, num_blocks_k_full]

    Return:
        a list of [block_indices, block_indices_lens] for k from each cp_rank
            - each block_indices starts from zero
            - block_indices_lens indicates the valid number of elements in the last dimension of block_indices
    """
    res = []
    num_blocks_per_rank = num_blocks_k_full // cp_size
    for i in range(cp_size):
        block_indices_tmp = block_indices.clone()
        min_block_idx = i * num_blocks_per_rank
        block_indices_tmp -= min_block_idx
        block_indices_tmp[block_indices_tmp < 0] = (
            num_blocks_per_rank  # block_indices_tmp < 0 indicate invalid indices, set them to num_blocks_per_rank in order to sort them to the tail, so that the first N elements of the block_indices indicated by block_indices_lens are valid
        )

        block_indices_tmp, _ = torch.sort(block_indices_tmp, dim=-1)

        block_indices_tmp_lens = (block_indices_tmp < num_blocks_per_rank).sum(dim=-1)

        res.append([block_indices_tmp, block_indices_tmp_lens])

    return res


@torch.compile
def flash_attn_fwd_softmax_lse_correction(
    softmax_lse: torch.Tensor,
    softmax_lse_per_step: torch.Tensor,
):
    """Merge softmax stats of each step in Attention with context parallelism"""
    max_scale = torch.max(softmax_lse, softmax_lse_per_step)
    min_scale = torch.min(softmax_lse, softmax_lse_per_step)
    lse_diff = min_scale - max_scale
    lse_diff = lse_diff.nan_to_num(
        nan=0.0
    )  # handle cases: tensor(-inf) - tensor(-inf) = tensor(nan); In the current cp implementation, it is possible that lses of 2 cp ranks are both -inf, if no block is selected from both cp ranks. In such cases, the finally corrected lse should remain -inf.
    new_scale = max_scale + torch.log1p(
        torch.exp(lse_diff)
    )  # a + ln(1 + e^(b - a)) = ln(e^a) + ln(1 + e^(b - a)) = ln(e^a + e^b)
    softmax_lse.copy_(new_scale)


@torch.compile
def flash_attn_fwd_out_correction_init(
    out_init_step: torch.Tensor,  # b h s d
    softmax_lse: torch.Tensor,  # b h s
    softmax_lse_init_step: torch.Tensor,
):
    """Merge partial outputs of the first step in Attention with context parallelism"""
    softmax_lse_corrected_exp = torch.exp(softmax_lse_init_step - softmax_lse)
    softmax_lse_corrected_exp = softmax_lse_corrected_exp.unsqueeze(-1)
    out_corrected = out_init_step * softmax_lse_corrected_exp
    return out_corrected.to(out_init_step.dtype)


@torch.compile
def flash_attn_fwd_out_correction(
    out: torch.Tensor,
    out_per_step: torch.Tensor,
    softmax_lse: torch.Tensor,
    softmax_lse_per_step: torch.Tensor,
):
    """Merge partial outputs of each step in Attention with context parallelism"""
    softmax_lse_corrected_exp = torch.exp(softmax_lse_per_step - softmax_lse)
    softmax_lse_corrected_exp = softmax_lse_corrected_exp.unsqueeze(-1)
    out_corrected = out_per_step * softmax_lse_corrected_exp
    out.add_(out_corrected)


@torch.compile
def topk_sort(score, num_chunks_selected):
    block_indices = torch.topk(score, num_chunks_selected)[1]
    block_indices, _ = torch.sort(block_indices, dim=-1)
    return block_indices


class _attention_bsa(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        chunk_size_q,
        chunk_size_k,
        sparsity,
        cdf_threshold,
        sm_scale,
        use_tma=False,
    ):
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}

        # ---------------------- gating ----------------------
        q_cmp = mean_pooling_compression(q, chunk_size_q)
        k_cmp = mean_pooling_compression(k, chunk_size_k)
        block_indices, block_indices_lens = get_select_indices(
            q_cmp, k_cmp, sparsity, cdf_threshold
        )

        # ---------------------- bsa ----------------------

        o, lse = attn_fwd_bsa_varlen_triton(
            q,
            k,
            v,
            sm_scale,
            block_indices,
            block_indices_lens,
            chunk_size_q,
            chunk_size_k,
            sparsity,
        )

        ctx.save_for_backward(q, k, v, o, lse, block_indices, block_indices_lens)
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.chunk_size_q = chunk_size_q
        ctx.chunk_size_k = chunk_size_k
        ctx.use_tma = use_tma
        ctx.sparsity = sparsity

        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse, block_indices, block_indices_lens = ctx.saved_tensors

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)

        attn_bwd_bsa_varlen_triton(
            do,
            q,
            k,
            v,
            o,
            dq,
            dk,
            dv,
            ctx.sm_scale,
            lse,
            block_indices,
            block_indices_lens,
            ctx.chunk_size_q,
            ctx.chunk_size_k,
            ctx.sparsity,
        )

        return dq, dk, dv, None, None, None, None, None, None


flash_attn_bsa = _attention_bsa.apply


def rearrange_THW_to_3d_block(x, Nt, Nh, Nw, t, h, w, D):
    B, H, _, D = x.shape
    x = x.view(B, H, Nt, t, Nh, h, Nw, w, D)
    x = x.permute(0, 1, 2, 4, 6, 3, 5, 7, 8)  # B H Nt Nh Nw t h w D
    return x.contiguous().view(B, H, Nt * Nh * Nw * t * h * w, D)


def rearrange_3d_block_to_THW(x, Nt, Nh, Nw, t, h, w, D):
    B, H, _, D = x.shape
    x = x.view(B, H, Nt, Nh, Nw, t, h, w, D)
    x = x.permute(0, 1, 2, 5, 3, 6, 4, 7, 8)  # B H Nt t Nh h Nw w D
    return x.contiguous().view(B, H, Nt * t * Nh * h * Nw * w, D)


def flash_attn_bsa_3d(
    q: torch.Tensor,  # [B, H, Sq, D]
    k: torch.Tensor,  # [B, H, Skv, D]
    v: torch.Tensor,  # [B, H, Skv, D]
    latent_shape_q,
    latent_shape_k,
    # bsa_params
    sparsity=0.875,
    cdf_threshold=None,
    chunk_3d_shape_q=[4, 4, 8],
    chunk_3d_shape_k=[4, 4, 8],
) -> torch.Tensor:
    _, _, Sq, head_dim_q = q.shape
    _, _, Sk, head_dim_k = k.shape

    assert head_dim_q == head_dim_k
    head_dim = head_dim_q

    Tq, Hq, Wq = latent_shape_q
    Tk, Hk, Wk = latent_shape_k

    assert Tq * Hq * Wq == Sq
    assert Tk * Hk * Wk == Sk

    tq, hq, wq = chunk_3d_shape_q
    tk, hk, wk = chunk_3d_shape_k

    assert Tq % tq == 0 and Hq % hq == 0 and Wq % wq == 0
    assert Tk % tk == 0 and Hk % hk == 0 and Wk % wk == 0

    Ntq = Tq // tq
    Nhq = Hq // hq
    Nwq = Wq // wq

    Ntk = Tk // tk
    Nhk = Hk // hk
    Nwk = Wk // wk

    q = rearrange_THW_to_3d_block(q, Ntq, Nhq, Nwq, tq, hq, wq, q.shape[-1])
    k = rearrange_THW_to_3d_block(k, Ntk, Nhk, Nwk, tk, hk, wk, k.shape[-1])
    v = rearrange_THW_to_3d_block(v, Ntk, Nhk, Nwk, tk, hk, wk, v.shape[-1])

    chunk_size_q = tq * hq * wq
    chunk_size_k = tk * hk * wk

    output = flash_attn_bsa(
        q, k, v, chunk_size_q, chunk_size_k, sparsity, cdf_threshold, 1 / head_dim**0.5
    )

    output = rearrange_3d_block_to_THW(
        output, Ntq, Nhq, Nwq, tq, hq, wq, output.shape[-1]
    )
    return output
