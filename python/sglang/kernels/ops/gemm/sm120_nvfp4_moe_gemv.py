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
    pass

# Decode (M=1) NVFP4 W4A4 grouped GEMV pair for the non-gated ReLU^2 latent
# MoE (see csrc/gemm/sm120_nvfp4_moe_gemv.cuh). CUTLASS 128-row tiles waste
# most of their work at M=1; this path streams the packed fp4 weights once
# at DRAM bandwidth, quantizes activations in-kernel with the same
# cvt.rn.satfinite hardware ops, and absorbs the expand / doActivation /
# finalize glue kernels.

_buffer_cache: dict = {}


@cache_once
def _jit_nvfp4_moe_gemv_module(n1, k1, n2, k2, topk):
    args = make_cpp_args(n1, k1, n2, k2, topk, is_arch_support_pdl())
    return load_jit(
        "sm120_nvfp4_moe_gemv",
        *args,
        cuda_files=["gemm/sm120_nvfp4_moe_gemv.cuh"],
        cuda_wrappers=[
            ("run_fc1", f"sglang::Sm120Nvfp4MoeGemvKernel<{args}>::run_fc1"),
            ("run_fc2", f"sglang::Sm120Nvfp4MoeGemvKernel<{args}>::run_fc2"),
        ],
        extra_cuda_cflags=["-O3"],
    )


def use_nvfp4_moe_gemv(num_tokens, n1, k1, n2, k2):
    return (
        num_tokens == 1
        and k1 % 512 == 0
        and 512 <= k1 <= 8192
        and n1 % 16 == 0
        and k2 == n1
        and n2 % 8 == 0
    )


def _z_buffers(topk, n1, device):
    key = (topk, n1, device.index)
    bufs = _buffer_cache.get(key)
    if bufs is None:
        zev = torch.empty(topk * (n1 // 8), dtype=torch.int32, device=device)
        zod = torch.empty(topk * (n1 // 8), dtype=torch.int32, device=device)
        zsf = torch.empty(topk * (n1 // 16), dtype=torch.float32, device=device)
        bufs = (zev, zod, zsf)
        _buffer_cache[key] = bufs
    return bufs


def nvfp4_moe_gemv_decode(
    x,
    w13_weight,
    w13_sf,
    w2_weight,
    w2_sf,
    g1_alphas,
    g2_alphas,
    a1_gscale,
    a2_gscale,
    topk_ids,
    topk_weights,
    out: Optional[torch.Tensor] = None,
):
    n1 = w13_weight.shape[1]
    k1 = x.shape[-1]
    n2 = w2_weight.shape[1]
    k2 = n1
    topk = topk_ids.numel()
    module = _jit_nvfp4_moe_gemv_module(n1, k1, n2, k2, topk)
    zev, zod, zsf = _z_buffers(topk, n1, x.device)
    ids = topk_ids.view(-1)
    if ids.dtype != torch.int32:
        ids = ids.to(torch.int32)
    tw = topk_weights.view(-1)
    if tw.dtype != torch.float32:
        tw = tw.to(torch.float32)
    module.run_fc1(
        x.view(-1),
        w13_weight.view(torch.uint8).view(-1),
        w13_sf.view(torch.uint8).view(-1),
        g1_alphas.view(-1),
        a1_gscale.view(-1),
        a2_gscale.view(-1),
        ids,
        zev,
        zod,
        zsf,
    )
    if out is None:
        out = torch.empty((1, n2), dtype=torch.bfloat16, device=x.device)
    module.run_fc2(
        zev,
        zod,
        zsf,
        w2_weight.view(torch.uint8).view(-1),
        w2_sf.view(torch.uint8).view(-1),
        g2_alphas.view(-1),
        ids,
        tw,
        out.view(-1),
    )
    return out
