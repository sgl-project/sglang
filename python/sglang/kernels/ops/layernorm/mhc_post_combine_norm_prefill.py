"""Prefill post/combine/RMSNorm with the original BF16 boundaries and norm tree."""

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.srt.utils.custom_op import register_custom_op


@cache_once
def _module():
    return load_jit(
        "mhc_post_combine_norm_prefill",
        cuda_files=["deepseek_v4/mhc_post_combine_norm_prefill.cuh"],
        cuda_wrappers=[("run", "MhcPostCombineNormPrefill<128>::run")],
    )


@register_custom_op(mutates_args=["updated", "normalized"])
def _mhc_post_combine_norm_prefill(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    pre: torch.Tensor,
    weight: torch.Tensor,
    updated: torch.Tensor,
    normalized: torch.Tensor,
    eps: float,
) -> None:
    _module().run(x, residual, post, comb, pre, weight, updated, normalized, eps)


def mhc_post_combine_norm_prefill(x, residual, post, comb, pre, weight, eps):
    updated, normalized = torch.empty_like(residual), torch.empty_like(x)
    _mhc_post_combine_norm_prefill(
        x, residual, post, comb, pre, weight, updated, normalized, eps
    )
    return updated, normalized
