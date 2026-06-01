from __future__ import annotations

import torch
import triton
import triton.language as tl
from tvm_ffi.module import Module

from sglang.jit_kernel.dsv4.utils import make_name
from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args


@triton.jit
def _online_c128_mtp_prepare_kernel(
    seq_lens_ptr,
    req_pool_indices_ptr,
    req_to_token_ptr,
    full_to_swa_ptr,
    main_state_ptr,
    pre_state_ptr,
    req_to_token_stride_b: tl.constexpr,
    main_state_stride_b: tl.constexpr,
    pre_state_stride_b: tl.constexpr,
    bs,
    swa_page_size: tl.constexpr,
    temp_state_slot_offset: tl.constexpr,
    state_width: tl.constexpr,
    block_d: tl.constexpr,
):
    bid = tl.program_id(0)
    if bid >= bs:
        return

    d = tl.arange(0, block_d)
    d_mask = d < state_width
    seq_len = tl.load(seq_lens_ptr + bid).to(tl.int64)
    has_partial = (seq_len > 0) & ((seq_len % 128) != 0)

    chunk_start = tl.where(has_partial, ((seq_len - 1) // 128) * 128, 0)
    req_idx = tl.load(req_pool_indices_ptr + bid).to(tl.int64)
    full_loc = tl.load(
        req_to_token_ptr + req_idx * req_to_token_stride_b + chunk_start,
        mask=has_partial,
        other=0,
    )
    swa_loc = tl.load(full_to_swa_ptr + full_loc, mask=has_partial, other=0).to(
        tl.int64
    )
    slot = tl.where(has_partial, swa_loc // swa_page_size, 0)
    temp_slot = slot + temp_state_slot_offset
    value = tl.load(
        main_state_ptr + slot * main_state_stride_b + d,
        mask=d_mask & has_partial,
        other=0.0,
    )
    tl.store(
        main_state_ptr + temp_slot * main_state_stride_b + d,
        value,
        mask=d_mask & has_partial,
    )

    tl.store(pre_state_ptr + bid * pre_state_stride_b + d, value, mask=d_mask)


@triton.jit
def _online_c128_mtp_accept_lens_kernel(
    ctx_req_pool_indices_ptr,
    ctx_seq_lens_ptr,
    cur_req_pool_indices_ptr,
    cur_seq_lens_ptr,
    accept_lens_ptr,
    cur_bs: tl.constexpr,
    max_draft_tokens: tl.constexpr,
    block_b: tl.constexpr,
):
    bid = tl.program_id(0)
    offsets = tl.arange(0, block_b)
    cur_mask = offsets < cur_bs

    ctx_req = tl.load(ctx_req_pool_indices_ptr + bid)
    old_seq_len = tl.load(ctx_seq_lens_ptr + bid).to(tl.int64)
    cur_req = tl.load(cur_req_pool_indices_ptr + offsets, mask=cur_mask, other=-1)
    cur_seq_len = tl.load(
        cur_seq_lens_ptr + offsets,
        mask=cur_mask,
        other=old_seq_len,
    ).to(tl.int64)

    matched_seq_len = tl.max(
        tl.where(cur_req == ctx_req, cur_seq_len, old_seq_len), axis=0
    )
    delta_len = matched_seq_len - old_seq_len
    # The verify seq_len delta includes the bonus token; the commit kernel only
    # has kv/score rows for draft tokens, so clamp instead of dropping full accepts.
    accept_len = tl.minimum(tl.maximum(delta_len, 0), max_draft_tokens)
    tl.store(accept_lens_ptr + bid, accept_len)


def online_c128_mtp_compute_accept_lens(
    *,
    ctx_req_pool_indices: torch.Tensor,
    ctx_seq_lens: torch.Tensor,
    cur_req_pool_indices: torch.Tensor,
    cur_seq_lens: torch.Tensor,
    max_draft_tokens: int,
) -> torch.Tensor:
    ctx_bs = ctx_seq_lens.shape[0]
    cur_bs = cur_seq_lens.shape[0]
    accept_lens = torch.empty((ctx_bs,), dtype=torch.int64, device=ctx_seq_lens.device)
    if ctx_bs <= 0:
        return accept_lens
    if cur_bs <= 0:
        accept_lens.zero_()
        return accept_lens

    block_b = triton.next_power_of_2(cur_bs)
    _online_c128_mtp_accept_lens_kernel[(ctx_bs,)](
        ctx_req_pool_indices,
        ctx_seq_lens,
        cur_req_pool_indices,
        cur_seq_lens,
        accept_lens,
        cur_bs,
        max_draft_tokens,
        block_b,
    )
    return accept_lens


@cache_once
def _jit_online_c128_mtp_commit_module(head_dim: int) -> Module:
    args = make_cpp_args(head_dim)
    return load_jit(
        make_name(f"online_c128_mtp_commit_{head_dim}"),
        *args,
        cuda_files=["deepseek_v4/online_c128_mtp.cuh"],
        cuda_wrappers=[
            ("commit", f"OnlineC128MTPCommitKernel<{args}>::run"),
        ],
        extra_cuda_cflags=["-use_fast_math"],
    )


def online_c128_mtp_commit(
    *,
    kv_score_input: torch.Tensor,
    pre_state: torch.Tensor,
    accept_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    full_to_swa_index_mapping: torch.Tensor,
    ape: torch.Tensor,
    main_state: torch.Tensor,
    layer_bs: int,
    swa_page_size: int,
    num_verify_tokens: int,
    head_dim: int,
) -> None:
    if layer_bs <= 0:
        return
    _jit_online_c128_mtp_commit_module(head_dim).commit(
        kv_score_input,
        pre_state,
        accept_lens,
        seq_lens,
        req_pool_indices,
        req_to_token,
        full_to_swa_index_mapping,
        ape,
        main_state,
        layer_bs,
        swa_page_size,
        num_verify_tokens,
    )


def _online_c128_mtp_prepare_triton(
    *,
    seq_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    full_to_swa_index_mapping: torch.Tensor,
    main_state: torch.Tensor,
    pre_state: torch.Tensor,
    bs: int,
    swa_page_size: int,
    temp_state_slot_offset: int,
    state_width: int,
) -> None:
    if bs <= 0:
        return
    assert state_width <= 2048
    block_d = triton.next_power_of_2(state_width)
    _online_c128_mtp_prepare_kernel[(bs,)](
        seq_lens,
        req_pool_indices,
        req_to_token,
        full_to_swa_index_mapping,
        main_state,
        pre_state,
        req_to_token.stride(0),
        main_state.stride(0),
        pre_state.stride(0),
        bs,
        swa_page_size,
        temp_state_slot_offset,
        state_width,
        block_d,
    )


def online_c128_mtp_prepare(
    *,
    seq_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    full_to_swa_index_mapping: torch.Tensor,
    main_state: torch.Tensor,
    pre_state: torch.Tensor,
    bs: int,
    swa_page_size: int,
    temp_state_slot_offset: int,
    state_width: int,
) -> None:
    _online_c128_mtp_prepare_triton(
        seq_lens=seq_lens,
        req_pool_indices=req_pool_indices,
        req_to_token=req_to_token,
        full_to_swa_index_mapping=full_to_swa_index_mapping,
        main_state=main_state,
        pre_state=pre_state,
        bs=bs,
        swa_page_size=swa_page_size,
        temp_state_slot_offset=temp_state_slot_offset,
        state_width=state_width,
    )
