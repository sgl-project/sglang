from __future__ import annotations

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    load_jit,
    make_cpp_args,
)


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
    use_mtp=True,
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
