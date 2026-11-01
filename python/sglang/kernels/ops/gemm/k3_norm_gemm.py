# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3 ROCm KDA gated o_norm fused into the o_proj GEMM prologue (M <= 16).

The normalized operand is bit-identical to layer_norm_gated_fwd_kernel's; the
output differs from the unfused pair only by GEMM summation order.
"""

from __future__ import annotations

import functools

import torch

_MODE_GATED_HEAD = 0
HIP_MAX_M = 16

# (waves, n_tiles_per_block); K fixes the per-wave K blocks
HIP_ONORM_CFG = (12, 2)


@functools.cache
def _hip_module(mode: int, waves: int, nt: int, kb: int):
    from sglang.kernels.jit.utils import load_jit, make_cpp_args

    args = make_cpp_args(mode, waves, nt, kb)
    return load_jit(
        f"k3_norm_gemm_{mode}_{waves}_{nt}_{kb}",
        *args,
        cuda_files=["kimi_k3/norm_gemm_hip.cuh"],
        cuda_wrappers=[("run", f"k3_norm_gemm::NormGemm<{args}>::run")],
        extra_cuda_cflags=["-O3"],
    )


def hip_onorm_supported(M: int, K: int, N: int, D: int) -> bool:
    waves, nt = HIP_ONORM_CFG
    return 1 <= M <= HIP_MAX_M and D == 128 and K == waves * 128 and N % (16 * nt) == 0


def kda_onorm_oproj_hip(
    x: torch.Tensor,  # [M, HV*D] contiguous rows
    gate: torch.Tensor,  # [M, HV*D] unit inner stride, row stride % 8 == 0
    norm_weight: torch.Tensor,  # [D] bf16 or fp32
    eps: float,
    w: torch.Tensor,  # [N, HV*D] bf16 contiguous
) -> torch.Tensor:
    x = x.reshape(-1, w.shape[1])
    M, K = x.shape
    N = w.shape[0]
    waves, nt = HIP_ONORM_CFG
    assert hip_onorm_supported(M, K, N, norm_weight.shape[0])
    out = torch.empty((M, N), dtype=x.dtype, device=x.device)
    if norm_weight.dtype != torch.float32:  # callers on the hot path pass fp32
        norm_weight = norm_weight.float()
    flags = 1 | 4
    assert norm_weight.is_contiguous()
    _hip_module(_MODE_GATED_HEAD, waves, nt, 4).run(
        x, gate, norm_weight, w, out, flags, float(eps)
    )
    return out
