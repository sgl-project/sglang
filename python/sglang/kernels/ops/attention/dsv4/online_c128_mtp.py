from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args
from sglang.kernels.ops.attention.dsv4.utils import make_name

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_online_c128_mtp_module(
    head_dim: int,
    seq_dtype: torch.dtype,
    req_dtype: torch.dtype,
    dtype_buffer: torch.dtype,
) -> Module:
    args = make_cpp_args(head_dim, seq_dtype, req_dtype, dtype_buffer)
    return load_jit(
        make_name(f"online_c128_mtp_{head_dim}"),
        *args,
        cuda_files=["deepseek_v4/online_c128_mtp.cuh"],
        cuda_wrappers=[
            ("write_prefix_states", f"OnlineC128MTPWritePrefixKernel<{args}>::run"),
            ("mark_pending", f"OnlineC128MTPMarkPendingKernel<{args}>::run"),
            ("commit_pending", f"OnlineC128MTPCommitPendingKernel<{args}>::run"),
        ],
        extra_cuda_cflags=["-use_fast_math"],
    )


def online_c128_mtp_write_prefix_states(
    kv_score_input: torch.Tensor,
    seq_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    ape: torch.Tensor,
    state: torch.Tensor,
    batch_size: int,
    num_verify_tokens: int,
    state_slot_stride: int,
    *,
    head_dim: int,
) -> None:
    """Stage the c128 compressor state after each verify token in that token's
    draft state slot."""
    _jit_online_c128_mtp_module(
        head_dim, seq_lens.dtype, req_pool_indices.dtype, state.dtype
    ).write_prefix_states(
        kv_score_input,
        seq_lens,
        req_pool_indices,
        req_to_token,
        ape,
        state,
        batch_size,
        num_verify_tokens,
        state_slot_stride,
    )


def online_c128_mtp_mark_pending(
    seq_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    pending_seq_lens: torch.Tensor,
    batch_size: int,
    num_req_slots: int,
    *,
    head_dim: int,
    state_dtype: torch.dtype,
) -> None:
    """Reset all pending verify lengths, then record ``seq_lens`` for the requests
    of this verify batch."""
    _jit_online_c128_mtp_module(
        head_dim, seq_lens.dtype, req_pool_indices.dtype, state_dtype
    ).mark_pending(
        seq_lens,
        req_pool_indices,
        pending_seq_lens,
        batch_size,
        num_req_slots,
    )


def online_c128_mtp_commit_pending(
    seq_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    pending_seq_lens: torch.Tensor,
    state: torch.Tensor,
    batch_size: int,
    num_verify_tokens: int,
    state_slot_stride: int,
    num_req_slots: int,
    *,
    head_dim: int,
) -> None:
    """Copy the draft state of each request's accepted prefix into its main c128
    state slot."""
    _jit_online_c128_mtp_module(
        head_dim, seq_lens.dtype, req_pool_indices.dtype, state.dtype
    ).commit_pending(
        seq_lens,
        req_pool_indices,
        req_to_token,
        pending_seq_lens,
        state,
        batch_size,
        num_verify_tokens,
        state_slot_stride,
        num_req_slots,
    )
