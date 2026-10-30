"""KDA chain spec-verify on gfx950 with the ReplaySSM ring, in one HIP kernel.

Conv1d, recurrence, ring and conv-window writes all run in
csrc/attention/kda_verify_gfx950.cuh. Takes the arguments of
``fused_kda_conv_gating_verify`` so the KDA backend can dispatch either.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_module(bv: int, t: int, lower_bound: bool, has_bias: bool) -> Module:
    args = make_cpp_args(bv, t, lower_bound, has_bias)
    return load_jit(
        "kda_verify_gfx950",
        *args,
        cuda_files=["attention/kda_verify_gfx950.cuh"],
        cuda_wrappers=[("run", f"KdaVerifyGfx950Kernel<{args}>::run")],
    )


def _octet_rows(t: torch.Tensor) -> torch.Tensor:
    # the kernel reads 8 bf16 channels per 16-byte load, so rows must be 16-byte aligned
    if t.stride(1) == 1 and t.stride(0) % 8 == 0 and t.data_ptr() % 16 == 0:
        return t
    return t.contiguous()


def kda_verify_block_v(batch_size: int) -> int:
    # Measured on MI350X, Kimi-K3 TP8 (12 heads, T=8): BV=4 wins at B=1, BV=8 from B=4;
    # BV=16 spills.
    return 4 if batch_size <= 2 else 8


def supports_kda_verify_gfx950(
    *,
    num_q_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    T: int,
    mixed_qkv: torch.Tensor,
    conv_weight: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    ssm_states: torch.Tensor,
    replayssm_rawv: Optional[torch.Tensor],
    intermediate_states_buffer: Optional[torch.Tensor],
    intermediate_conv_window: Optional[torch.Tensor],
) -> bool:
    return (
        replayssm_rawv is not None
        and intermediate_states_buffer is None
        and intermediate_conv_window is not None
        and num_q_heads == num_v_heads
        and head_k_dim == head_v_dim == 128
        and T == 8  # the 4-wave gate mapping splits 256 lanes into 8 tokens x 32
        and mixed_qkv.dtype == torch.bfloat16
        and a.dtype == b.dtype == torch.bfloat16
        and conv_weight.dtype == torch.float32
        and ssm_states.dtype == torch.float32
        and replayssm_rawv.dtype == torch.bfloat16
    )


def kda_verify_gfx950(
    mixed_qkv: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: Optional[torch.Tensor],
    conv_state: torch.Tensor,
    conv_state_indices: torch.Tensor,
    intermediate_conv_window: torch.Tensor,
    intermediate_state_indices: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
    intermediate_states_buffer: Optional[torch.Tensor],
    scale: float,
    T: int,
    num_q_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    lower_bound: Optional[float] = None,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
    use_qk_l2norm_in_kernel: bool = True,
    num_warps: Optional[int] = None,
    cache_ring: bool = False,
    replayssm_rawv: Optional[torch.Tensor] = None,
    replayssm_rawk: Optional[torch.Tensor] = None,
    replayssm_g: Optional[torch.Tensor] = None,
    replayssm_beta: Optional[torch.Tensor] = None,
    block_v: Optional[int] = None,
) -> torch.Tensor:
    """Returns ``o`` of shape [1, seq_len, HV, V]."""
    assert cache_ring and use_qk_l2norm_in_kernel and intermediate_states_buffer is None
    seq_len = mixed_qkv.shape[0]
    HV, V = num_v_heads, head_v_dim
    batch_size = seq_len // T
    bv = block_v or kda_verify_block_v(batch_size)
    o = mixed_qkv.new_empty(seq_len, HV, V)
    module = _jit_module(bv, T, lower_bound is not None, conv_bias is not None)
    module.run(
        _octet_rows(mixed_qkv),
        conv_weight,
        conv_bias,
        conv_state,
        cache_indices,
        intermediate_conv_window,
        intermediate_state_indices,
        _octet_rows(a.reshape(seq_len, HV * head_k_dim)),
        _octet_rows(b.reshape(seq_len, HV)),
        A_log,
        dt_bias,
        ssm_states,
        replayssm_rawv,
        replayssm_rawk,
        replayssm_g,
        replayssm_beta,
        o,
        num_q_heads,
        float(scale),
        float(lower_bound) if lower_bound is not None else 0.0,
        float(softplus_beta),
        float(softplus_threshold),
    )
    return o.view(1, seq_len, HV, V)
