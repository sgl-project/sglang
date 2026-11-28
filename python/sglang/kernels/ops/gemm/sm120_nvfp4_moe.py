from __future__ import annotations

from typing import Optional

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


# Above this many tokens FlashInfer's grouped CUTLASS GEMMs are as fast as the fused kernel.
SWIGLU_MAX_TOKENS = 512


@cache_once
def _swiglu_module(num_experts: int, top_k: int, hidden: int, inter: int):
    args = make_cpp_args(num_experts, top_k, hidden, inter)
    return load_jit(
        "sm120_nvfp4_moe_swiglu",
        *args,
        cuda_files=["gemm/sm120_nvfp4_moe_swiglu.cuh"],
        cuda_wrappers=[("run", f"sglang::Sm120Nvfp4MoeSwiglu<{args}>::run")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        extra_dependencies=["cutlass"],
    )


def _swiglu_workspace_sizes(
    num_tokens: int, num_experts: int, top_k: int, hidden: int, inter: int
) -> tuple[int, int]:
    """Workspace sizes (uint8 bytes, int32 count); mirrors ``Shape`` in the kernel."""

    def align(n):
        return (n + 255) // 256 * 256

    routes = num_tokens * top_k
    units = min(routes, num_experts) + (routes + 31) // 32
    u8 = (
        align(num_tokens * hidden // 2)
        + align(num_tokens * hidden // 16)
        + align(units * 32 * inter // 2)
        + align(units * (inter // 128) * 32 * 8)
    )
    i32 = (
        2 * routes
        + 2 * units
        + 2 * num_experts
        + num_tokens * (hidden // 128)
        + num_tokens
        + 2
    )
    return u8, i32


def nvfp4_moe_swiglu(
    x: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    alpha1: torch.Tensor,
    alpha2: torch.Tensor,
    input_scale1: torch.Tensor,
    input_scale2: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused NVFP4 SwiGLU MoE on SM120, bitwise equal to FlashInfer's CUTLASS
    ``cutlass_fused_moe`` with BF16 input, ``w13 = [up, gate]`` and the unfused
    finalize. Experts outside ``[0, num_experts)`` are skipped."""
    num_tokens, hidden = x.shape
    num_experts, inter = w13.shape[0], w13.shape[1] // 2
    top_k = topk_ids.shape[1]
    module = _swiglu_module(num_experts, top_k, hidden, inter)
    if out is None:
        out = torch.empty_like(x)
    u8, i32 = _swiglu_workspace_sizes(num_tokens, num_experts, top_k, hidden, inter)
    module.run(
        x,
        w13.view(torch.uint8),
        w13_scale.view(torch.uint8).view(-1),
        w2.view(torch.uint8),
        w2_scale.view(torch.uint8).view(-1),
        alpha1.view(-1),
        alpha2.view(-1),
        input_scale1.view(-1),
        input_scale2.view(-1),
        topk_ids.to(torch.int32).contiguous(),
        topk_weights.float().contiguous(),
        torch.empty(u8, dtype=torch.uint8, device=x.device),
        torch.empty(
            (num_tokens * top_k, hidden), dtype=torch.bfloat16, device=x.device
        ),
        torch.empty(i32, dtype=torch.int32, device=x.device),
        out,
    )
    return out


def try_nvfp4_moe_swiglu(
    x, w13, w2, scales, ids, routing, *, output_dtype, out=None
) -> Optional[torch.Tensor]:
    """Run the fused SwiGLU kernel when the shapes allow it; return None to
    keep the CUTLASS path."""
    if not (
        x.ndim == 2
        and 1 <= x.shape[0] <= SWIGLU_MAX_TOKENS
        and x.dtype == output_dtype == torch.bfloat16
        and x.is_contiguous()
        and w13.dim() == 3
        and w13.shape[0] % 32 == 0
        and x.shape[1] % 256 == 0
        and w13.shape[1] % 256 == 0
        and w13.shape[2] * 2 == x.shape[1]
        and w2.shape == (w13.shape[0], x.shape[1], w13.shape[1] // 4)
        and scales[0].numel() == 1
        and scales[3].numel() == 1
        and (out is None or out.is_contiguous())
    ):
        return None
    return nvfp4_moe_swiglu(
        x,
        w13,
        scales[1],
        w2,
        scales[4],
        scales[2],
        scales[5],
        scales[0],
        scales[3],
        ids,
        routing,
        out=out,
    )


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
