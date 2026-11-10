# SPDX-License-Identifier: Apache-2.0
"""Bit-exact channel-first RMSNorm finish + SiLU (Qwen-Image 2.1 VAE).

The reference is ``F.silu(F.normalize(x.float(), dim=1).to(x.dtype) * scale * gamma + 0.0)``.
The fp32 L2 norm over channels stays with aten (``x.float().norm(p=2, dim=1, keepdim=True)``)
so the reduction order is the reference's; one CUDA launch then replays the pointwise
tail with the eager rounding points (bf16 after the divide, the scale multiply, the gamma
multiply and the ``+ 0.0`` that folds ``-0.0`` to ``+0.0``) and aten's SiLU formula
``v / (1 + expf(-v))`` on the bf16 value. Contiguous NCHW / NCDHW bf16 activations with a
spatial size that is a multiple of 8. CUDA only; callers verify once through a
``BitExactFusionGate`` and keep the eager chain as the fallback.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_SPATIAL_MULTIPLE = 8


@cache_once
def _jit_module() -> Module:
    return load_jit(
        "channel_rmsnorm_finish_silu",
        cuda_files=["diffusion/channel_rmsnorm_finish_silu.cuh"],
        cuda_wrappers=[("run", "channel_rmsnorm_finish_silu::Kernel::run")],
    )


def can_use_channel_rmsnorm_finish_silu(x: torch.Tensor, gamma: torch.Tensor) -> bool:
    """Contiguous bf16 NCHW / NCDHW ``x`` with ``gamma`` shaped ``(C, 1, ...)`` on the same device."""
    if not (
        x.is_cuda
        and torch.version.hip is None
        and not torch.compiler.is_compiling()
        and x.dtype is torch.bfloat16
        and x.ndim in (4, 5)
        and x.numel() > 0
        and x.is_contiguous()
        and x.data_ptr() % 16 == 0
    ):
        return False
    spatial = x.numel() // (x.shape[0] * x.shape[1])
    return (
        spatial % _SPATIAL_MULTIPLE == 0
        and gamma.device == x.device
        and gamma.dtype is torch.bfloat16
        and gamma.shape == (x.shape[1],) + (1,) * (x.ndim - 2)
        and gamma.is_contiguous()
    )


def _fake(x, norm, gamma, scale):
    return torch.empty_like(x)


@register_custom_op(
    op_name="channel_rmsnorm_finish_silu", mutates_args=[], fake_impl=_fake
)
def channel_rmsnorm_finish_silu(
    x: torch.Tensor, norm: torch.Tensor, gamma: torch.Tensor, scale: float
) -> torch.Tensor:
    """``silu(rmsnorm_finish(x))`` given aten's fp32 channel norm ``norm = x.float().norm(dim=1, keepdim=True)``."""
    batch, channels = x.shape[0], x.shape[1]
    out = torch.empty_like(x)
    with torch.cuda.device(x.device):
        _jit_module().run(
            out.view(batch, channels, -1),
            x.view(batch, channels, -1),
            norm.reshape(batch, -1),
            gamma.reshape(-1),
            float(scale),
        )
    return out


__all__ = ["can_use_channel_rmsnorm_finish_silu", "channel_rmsnorm_finish_silu"]
