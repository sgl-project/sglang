# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from functools import cache

import torch
import torch.nn as nn

from sglang.kernels.ops.diffusion.sites.quality_gate import QualityGatedFusion


@cache
def _get_qk_rmsnorm_cute():
    try:
        # Use FlashInfer's re-exported CuTe entry point directly: the public
        # ``rmsnorm`` wrapper adds custom-op dispatch to every Hunyuan block.
        from flashinfer.norm import qk_rmsnorm_cute
    except ImportError:
        return None
    return qk_rmsnorm_cute


def _site_reject_reason(_site: nn.Module) -> str | None:
    if _get_qk_rmsnorm_cute() is None:
        return "FlashInfer CuTe QK RMSNorm unavailable"
    return None


HUNYUAN_QKNORM = QualityGatedFusion(
    name="HunyuanVideo strided QK RMSNorm",
    marker_attr="_sgl_hunyuan_qknorm_site",
    enabled_attr="_sgl_hunyuan_qknorm_enabled",
    reject_reason=_site_reject_reason,
)


def try_hunyuan_qknorm(
    site: nn.Module,
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Normalize strided Hunyuan Q/K views without materializing inputs."""
    if not (
        HUNYUAN_QKNORM.is_enabled(site)
        and not torch.compiler.is_compiling()
        and q.is_cuda
        and q.dtype == torch.bfloat16
        and k.dtype == q.dtype
        and q_weight.dtype == q.dtype
        and k_weight.dtype == q.dtype
        and q.stride(-1) == 1
        and k.stride(-1) == 1
    ):
        return None

    qk_rmsnorm_cute = _get_qk_rmsnorm_cute()
    if qk_rmsnorm_cute is None:
        return None

    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    q_shape = q.shape
    k_shape = k.shape
    qk_rmsnorm_cute(
        q.reshape(-1, q_shape[-2], q_shape[-1]),
        q_weight,
        q_out.reshape(-1, q_shape[-2], q_shape[-1]),
        eps,
        enable_pdl=True,
    )
    qk_rmsnorm_cute(
        k.reshape(-1, k_shape[-2], k_shape[-1]),
        k_weight,
        k_out.reshape(-1, k_shape[-2], k_shape[-1]),
        eps,
        enable_pdl=True,
    )
    return q_out, k_out
