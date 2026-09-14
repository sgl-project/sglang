"""Fused DeepSeek-V4.1 mHC sublayer boundary: post-mix + pre-combine + RMSNorm.

One launch replaces the Triton pair ``mhc_post_split_h`` + ``hc_combine_norm``
for HC = 4 and hidden 5120::

    residual[t, i, :] = bf16(post[t, i] * x[t, :] + sum_j comb[t, j, i] * residual[t, j, :])
    y[t, :]           = RMSNorm_w(bf16(sum_i pre[t, i] * residual[t, i, :]))

``residual`` is rewritten in place. A cluster of ``cluster_size`` CTAs owns one
token; the size trades CTA count against per-CTA work.
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

from .utils import make_name

if TYPE_CHECKING:
    from tvm_ffi.module import Module

HC_WIDTH = 4
HIDDEN_DIM = 5120


@cache_once
def _jit_mhc_module(cluster_size: int, vec_size: int) -> Module:
    if vec_size == 16 and torch.cuda.get_device_capability()[0] < 10:
        raise RuntimeError("vec_size=16 (32 B per thread) needs Blackwell")
    if cluster_size > 1 and torch.cuda.get_device_capability()[0] < 9:
        raise RuntimeError("thread block clusters need SM90 or later")
    args = make_cpp_args(cluster_size, vec_size, is_arch_support_pdl())
    return load_jit(
        make_name("mhc_post_combine_norm"),
        *args,
        cuda_files=["deepseek_v4/mhc.cuh"],
        cuda_wrappers=[
            ("post_combine_norm", f"MHCFusedKernel<{args}>::post_combine_norm")
        ],
    )


def mhc_post_combine_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    pre: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    cluster_size: Optional[int] = None,
    vec_size: int = 8,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused mHC post-mix (in place into ``residual``) + pre-combine + RMSNorm.

    ``x`` ``[m, 5120]`` bf16, ``residual`` ``[m, 4, 5120]`` bf16 (updated in
    place), ``post``/``pre`` ``[m, 4]`` fp32, ``comb`` ``[m, 4, 4]`` fp32 indexed
    ``[token, source stream, target stream]``, ``weight`` ``[5120]`` bf16. All
    contiguous. Returns the normalized ``[m, 5120]`` bf16 sublayer input.
    """
    if out is None:
        out = torch.empty_like(x)
    if cluster_size is None:
        cluster_size = 5 if x.shape[0] <= 33 else 1
    module = _jit_mhc_module(cluster_size, vec_size)
    module.post_combine_norm(x, residual, post, comb, pre, weight, out, eps)
    return out


def mhc_post_combine_norm_reference(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    pre: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 torch reference with the same two bf16 roundings as the kernel.

    Out of place: returns ``(new_residual, y)`` and leaves ``residual`` untouched.
    """
    new_residual = (
        torch.einsum("tji,tjh->tih", comb, residual.float())
        + post[:, :, None] * x[:, None, :].float()
    ).to(residual.dtype)
    v = (pre[:, :, None] * new_residual.float()).sum(dim=1).to(x.dtype).float()
    inv_rms = torch.rsqrt(v.square().mean(dim=-1, keepdim=True) + eps)
    y = (v * inv_rms * weight.float()).to(x.dtype)
    return new_residual, y


def hc_boundary_fused(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    pre: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int,
    sinkhorn_iters: int,
    rms_eps: float,
    hc_eps: float,
    stats_stream: Optional[torch.cuda.Stream] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One mHC sublayer boundary: close the previous sublayer and open the next.

    ``residual`` is updated in place to the post-mixed streams, ``y`` is the
    normalized input of the next sublayer (combine with ``pre`` + RMSNorm), and
    the next sublayer's coefficients ``(pre, post, comb)`` are computed from the
    updated streams, on ``stats_stream`` when given so they overlap the sublayer.
    The caller joins ``stats_stream`` before consuming them (and before the next
    write to ``residual``). Returns ``(y, pre, post, comb)``.
    """
    from contextlib import nullcontext

    from sglang.kernels.ops.layernorm.mhc import hc_mix_stats_sinkhorn

    y = mhc_post_combine_norm(x, residual, post, comb, pre, weight, eps)
    main_stream = torch.cuda.current_stream()
    if stats_stream is not None:
        stats_stream.wait_stream(main_stream)
        residual.record_stream(stats_stream)
    with torch.cuda.stream(stats_stream) if stats_stream is not None else nullcontext():
        next_pre, next_post, next_comb = hc_mix_stats_sinkhorn(
            residual.flatten(1),
            hc_fn,
            hc_scale,
            hc_base,
            hc_mult,
            sinkhorn_iters,
            rms_eps,
            hc_eps,
        )
    if stats_stream is not None:
        for coefficient in (next_pre, next_post, next_comb):
            coefficient.record_stream(main_stream)
    return y, next_pre, next_post, next_comb
