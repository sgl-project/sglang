from __future__ import annotations

from typing import Optional

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

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


@cache_once
def _mtp_module(pair_layout: bool = False):
    args = make_cpp_args(2688, 1024, 1024, 2688, 22, False, pair_layout)
    return load_jit(
        "sm120_nvfp4_moe_mtp_mma",
        *args,
        cuda_files=["gemm/sm120_nvfp4_moe_mtp_mma.cuh"],
        cuda_wrappers=[("run", f"sglang::Sm120Nvfp4MoeMtpMma<{args}>::run")],
        extra_cuda_cflags=["-O3"],
    )


def nvfp4_moe_mtp_mma(
    x, w1, s1, w2, s2, a1, a2, g1, g2, ids, routing, out=None, *, pair_layout=False
):
    m = x.shape[0]
    module = _mtp_module(pair_layout)
    if out is None:
        out = torch.empty((m, 1024), dtype=torch.bfloat16, device=x.device)
    z = torch.empty(m * 22 * 2688 // 8, dtype=torch.int32, device=x.device)
    zs = torch.empty(m * 22 * 2688 // 16, dtype=torch.uint8, device=x.device)
    partial = torch.empty(m * 22 * 1024, dtype=torch.bfloat16, device=x.device)
    module.run(
        x,
        w1.view(torch.uint8),
        s1.view(torch.uint8).view(-1),
        w2.view(torch.uint8),
        s2.view(torch.uint8).view(-1),
        a1.view(-1),
        a2.view(-1),
        g1.view(-1),
        g2.view(-1),
        ids.to(torch.int32),
        routing.float(),
        z,
        zs,
        partial,
        out,
    )
    return out


def try_nvfp4_moe(
    x,
    w13,
    w2,
    scales,
    ids,
    routing,
    *,
    output_dtype,
    out=None,
    pair_w2_weight=None,
    w13_blockscale_plain=None,
    w2_blockscale_plain=None,
    use_mtp=True,
    use_gemv=True,
):
    """Select native small-M kernels; unsupported shapes retain CUTLASS."""
    if (
        use_mtp
        and x.ndim == 2
        and 1 <= x.shape[0] <= 4
        and x.shape[1] == 1024
        and x.dtype == output_dtype == torch.bfloat16
        and w13.shape == (512, 2688, 512)
        and w2.shape == (512, 1024, 1344)
        and ids.shape == (x.shape[0], 22)
    ):
        return nvfp4_moe_mtp_mma(
            x,
            w13,
            scales[1],
            pair_w2_weight if pair_w2_weight is not None else w2,
            scales[4],
            scales[2],
            scales[5],
            scales[0],
            scales[3],
            ids,
            routing,
            out=out,
            pair_layout=pair_w2_weight is not None,
        )
    if pair_w2_weight is not None:
        raise RuntimeError("A paired NVFP4 weight reached an unsupported MoE path")
    if (
        use_gemv
        and w13_blockscale_plain is not None
        and x.ndim == 2
        and x.dtype == torch.bfloat16
        and use_nvfp4_moe_gemv(
            x.shape[0], w13.shape[1], w13.shape[2] * 2, w2.shape[1], w2.shape[2] * 2
        )
    ):
        if out is None:
            out = torch.empty((1, x.shape[1]), dtype=output_dtype, device=x.device)
        return nvfp4_moe_gemv_decode(
            x,
            w13,
            w13_blockscale_plain,
            w2,
            w2_blockscale_plain,
            scales[2],
            scales[5],
            scales[0],
            scales[3],
            ids,
            routing,
            out=out,
        )
    return None


@cache_once
def _transpose_module(rows: int, cols: int):
    return load_jit(
        "sm120_nvfp4_weight_transpose",
        rows,
        cols,
        cuda_files=["gemm/sm120_nvfp4_weight_transpose.cuh"],
        cuda_wrappers=[("run", f"sglang::Nvfp4WeightTranspose<{rows},{cols}>::run")],
        extra_cuda_cflags=["-O3"],
    )


class Nvfp4LayoutCycle:
    """Own the two views of a fixed, serial set of [512,1024,1344] weights."""

    def __init__(self, weights):
        if not weights:
            raise ValueError("A layout cycle requires at least one expert weight")
        first = weights[0]
        if first.device.type != "cuda":
            raise ValueError("The layout cycle requires CUDA weights")
        for weight in weights:
            if (
                weight.shape != (512, 1024, 1344)
                or weight.dtype != torch.uint8
                or not weight.is_contiguous()
                or weight.device != first.device
            ):
                raise ValueError("Incompatible NVFP4 FC2 weight layout")
        if len({weight.data_ptr() for weight in weights}) != len(weights):
            raise ValueError("Expert weights in a layout cycle must not alias")
        self.original = tuple(weights)
        self.workspace_bytes = first.numel()
        self.scratch = None
        self.pair = ()
        self.is_pair = False
        self.switches = 0
        # Compile before any graph capture can reach the state transition.
        self._forward = _transpose_module(1024, 1344)
        self._inverse = _transpose_module(1344, 1024)

    def prepare_workspace(self):
        if self.scratch is not None:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("NVFP4 workspace must be prepared before graph capture")
        # Use runtime workspace headroom, like graph/kernel scratch allocations.
        # Allocating this before Mamba/KV pool sizing incorrectly charges it as
        # persistent checkpoint storage and can round the state pool below one
        # request. The normal graph warmup prepares stable pointers before use.
        self.scratch = torch.empty_like(self.original[0])
        self.pair = tuple(
            weight.view(512, 1344, 1024)
            for weight in (self.scratch, *self.original[:-1])
        )

    def ensure(self, pair: bool):
        if pair == self.is_pair:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("NVFP4 layout transitions must run before graph capture")
        if pair:
            self.prepare_workspace()
            for original, paired in zip(self.original, self.pair):
                self._forward.run(original, paired)
        else:
            for original, paired in reversed(tuple(zip(self.original, self.pair))):
                self._inverse.run(paired, original)
        self.is_pair = pair
        self.switches += 1
