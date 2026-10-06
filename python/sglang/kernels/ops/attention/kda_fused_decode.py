"""Fully fused KDA decode step (Kimi K3 batched decode fast path).

One kernel replaces the three-kernel decode chain
``causal_conv1d_update -> kda_packed_decode -> rms_norm_gated``: it reads the
raw (pre-conv) qkv slice straight out of the fused projection GEMM output,
does the causal conv1d update (conv state shifted in the pool in place), the
delta-rule recurrence (l2-normed q/k, softplus forget gate, sigmoid beta),
and the sigmoid-gated output RMSNorm.

Kernel body vendored from the NVIDIA x Moonshot Kimi K3 optimization package
(see csrc/attention/kda_fused_decode.cuh for provenance and the list of
integration patches). Specialized for the K3 KDA decode regime:
K = V = 128, kernel width 4, no lower bound, T = 1 per request.
The JIT currently instantiates local head counts H = HV in {12, 6, 3}
(TP8, TP16, and TP32).

The model must hand off the output-norm gate (attempt-and-verify stash on the
attention layer, see kimi_k3.py), and a covered() check gates supported inputs.
Everything else falls back to the unfused chain.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_SUPPORTED_HEADS = {3, 6, 12}
_CONV_STATE_W = 3  # kernel width 4 -> 3 cached tokens


@cache_once
def _jit_kda_fused_decode_module() -> Module:
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        "kda_fused_decode",
        *args,
        cuda_files=["attention/kda_fused_decode.cuh"],
        cuda_wrappers=[("run", f"KdaFusedDecodeKernel<{args}>::run")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
    )


def covered(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    conv_states: torch.Tensor,
    ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
    onorm_g: torch.Tensor,
) -> bool:
    """The kernel is compiled for the K3 KDA decode regime: H heads of 128,
    packed [T, 3*H*128] qkv rows, transposed [slots, 3, 3*H*128] conv pool, fp32
    [slots, H, 128, 128] ssm pool (inner-contiguous, any slot pitch — the
    kernel reads the real slot stride), one token per request."""
    if ssm_states.ndim < 4:
        return False
    H, V, K = ssm_states.shape[-3:]
    if H not in _SUPPORTED_HEADS:
        return False
    seg = H * 128
    conv_dim = 3 * seg
    if mixed_qkv.ndim != 2 or mixed_qkv.shape[-1] != conv_dim:
        return False
    return (
        V == 128
        and K == 128
        and a.ndim == 2
        and a.shape[-1] == seg
        and b.ndim == 2
        and b.shape[-1] == H
        and onorm_g.ndim == 2
        and onorm_g.shape[-1] == seg
        and conv_states.ndim == 3
        and conv_states.shape[-2:] == (_CONV_STATE_W, conv_dim)
        and mixed_qkv.dtype == torch.bfloat16
        and a.dtype == torch.bfloat16
        and b.dtype == torch.bfloat16
        and onorm_g.dtype == torch.bfloat16
        and conv_states.dtype == torch.bfloat16
        and ssm_states.dtype == torch.float32
        and cache_indices.dtype == torch.int32
        and mixed_qkv.stride(-1) == 1
        and a.stride(-1) == 1
        and b.stride(-1) == 1
        and onorm_g.stride(-1) == 1
        and conv_states.stride(-1) == 1
        # Inner [HV, V, K] must be contiguous (the kernel float4-loads V*K
        # chunks); the slot pitch (stride(-4)) is arbitrary — a locally
        # allocated pool packs it at HV*V*K, the unified / page-major pools at
        # the multi-layer envelope. The kernel reads ssm_states.stride(0), so
        # any slot pitch is fine. (Do NOT use .view(-1, HV, V, K): that fails /
        # copies on an envelope-strided view.)
        and ssm_states.stride(-1) == 1
        and ssm_states.stride(-2) == K
        and ssm_states.stride(-3) == V * K
        and cache_indices.is_contiguous()
    )


def kda_fused_decode(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    conv_states: torch.Tensor,
    w_q_t: torch.Tensor,
    w_k_t: torch.Tensor,
    w_v_t: torch.Tensor,
    conv_bias: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    onorm_g: torch.Tensor,
    onorm_weight: torch.Tensor,
    ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
    scale: float,
    onorm_eps: float,
    lower_bound: Optional[float] = None,
) -> torch.Tensor:
    """In-place fused decode step: shifts `conv_states` and updates
    `ssm_states` rows selected by `cache_indices` (rows < 0 are padded
    cuda-graph slots and only zero their output), returns the gated-normed
    attention output [1, B, HV, V] (the packed-decode output layout).
    Caller must have checked covered()."""
    B = mixed_qkv.shape[0]
    H = ssm_states.shape[-3]
    seg = H * 128
    out = torch.empty((B, seg), dtype=torch.bfloat16, device=mixed_qkv.device)
    _jit_kda_fused_decode_module().run(
        mixed_qkv,
        a,
        b,
        conv_states,
        w_q_t,
        w_k_t,
        w_v_t,
        conv_bias,
        A_log,
        dt_bias,
        onorm_g,
        onorm_weight,
        # Pass the pool view as-is (already [slots, HV, V, K]); the kernel
        # binding reads its real slot stride via state.stride(0). A
        # .view(-1, H, 128, 128) here would break on envelope-strided pools
        # (unified / page-major) — the reshape can't fold a non-dense slot
        # pitch and would raise / silently copy.
        ssm_states,
        cache_indices,
        out,
        float(scale),
        float(onorm_eps),
        float(lower_bound) if lower_bound is not None else 0.0,
        lower_bound is not None,
    )
    return out.view(1, B, H, 128)
