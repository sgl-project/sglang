# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Fused operators for normalization layers, with ROCm AITer support."""

import logging
import os
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from sglang.srt.custom_op import CustomOp
from sglang.srt.utils import is_cuda

_is_cuda = is_cuda()

if _is_cuda:
    from sgl_kernel import (
        fused_add_rmsnorm,
        gemma_fused_add_rmsnorm,
        gemma_rmsnorm,
        rmsnorm,
    )


logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# ROCm AITer integration -------------------------------------------------------
# -----------------------------------------------------------------------------


def _is_rocm() -> bool:
    """Return True when running on ROCm (HIP) runtime."""
    return hasattr(torch.version, "hip") and torch.version.hip is not None  # type: ignore[attr-defined]


def is_rocm_aiter_rmsnorm_enabled() -> bool:  # noqa: N802  (follow existing naming)
    """Heuristic to decide whether to use AITer kernels on ROCm.

    The behaviour mirrors vLLM's env‑flag logic so existing deployment flags
    continue to work:
        * VLLM_ROCM_USE_AITER_RMSNORM=1 enables the specialised RMSNorm kernels.
        * VLLM_ROCM_USE_AITER=1 must also be set (or the old flag combination
          used by vLLM).
    """
    return _is_rocm() and (
        os.environ.get("VLLM_ROCM_USE_AITER_RMSNORM", "0") == "1"
        and os.environ.get("VLLM_ROCM_USE_AITER", "0") == "1"
    )

# ---- Safe import of AITer ----------------------------------------------------

try:
    import aiter as _rocm_aiter  # lightweight import, only when available

    def rocm_aiter_rms_norm(  # noqa: N802
        x: torch.Tensor, weight: torch.Tensor, variance_epsilon: float
    ) -> torch.Tensor:
        """Thin wrapper that calls AITer's RMSNorm implementation."""
        return _rocm_aiter.rms_norm(x, weight, variance_epsilon)  # type: ignore[attr-defined]

    def rocm_aiter_fused_add_rms_norm(  # noqa: N802
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        variance_epsilon: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fused residual‑add‑plus‑RMSNorm for AITer on ROCm."""
        _rocm_aiter.rmsnorm2d_fwd_with_add(  # type: ignore[attr-defined]
            x,          # output tensor (in‑place on *x*)
            x,          # input tensor
            residual,   # residual input
            residual,   # residual output (in‑place)
            weight,
            variance_epsilon,
        )
        return x, residual

except ModuleNotFoundError:

    def rocm_aiter_rms_norm(*args, **kwargs):  # type: ignore[return‑type]
        raise RuntimeError(
            "AITer not found. Install ROCm AITer or unset VLLM_ROCM_USE_AITER_RMSNORM.")

    def rocm_aiter_fused_add_rms_norm(*args, **kwargs):  # type: ignore[return‑type]
        raise RuntimeError(
            "AITer not found. Install ROCm AITer or unset VLLM_ROCM_USE_AITER_RMSNORM.")

# -----------------------------------------------------------------------------
#  Helper to pick the fastest kernel at runtime (CUDA / ROCm) ------------------
# -----------------------------------------------------------------------------

def dispatch_cuda_rmsnorm_func(add_residual: bool):
    """Return the best available kernel for (fused‑)RMSNorm on the current HW."""

    # Prefer ROCm AITer when explicitly enabled.
    if add_residual:
        if is_rocm_aiter_rmsnorm_enabled():
            return rocm_aiter_fused_add_rms_norm
        return fused_add_rmsnorm  # type: ignore[name‑defined]

    # No residual path
    if is_rocm_aiter_rmsnorm_enabled():
        return rocm_aiter_rms_norm
    return rmsnorm  # type: ignore[name‑defined]

# -----------------------------------------------------------------------------
#  Core classes ----------------------------------------------------------------
# -----------------------------------------------------------------------------


class RMSNorm(CustomOp):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        add_residual = residual is not None
        norm_fn = dispatch_cuda_rmsnorm_func(add_residual)

        if add_residual:
            return norm_fn(x, residual, self.weight.data, self.variance_epsilon)
        return norm_fn(x, self.weight.data, self.variance_epsilon)

    def forward_native(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        if residual is not None:
            x = x + residual.to(torch.float32)
            residual = x.to(orig_dtype)

        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = (x * self.weight).to(orig_dtype)
        if residual is None:
            return x
        else:
            return x, residual


class GemmaRMSNorm(CustomOp):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward_native(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        orig_dtype = x.dtype
        if residual is not None:
            x = x + residual
            residual = x

        x = x.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = x * (1.0 + self.weight.float())
        x = x.to(orig_dtype)
        return x if residual is None else (x, residual)

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if residual is not None:
            gemma_fused_add_rmsnorm(
                x, residual, self.weight.data, self.variance_epsilon
            )
            return x, residual
        out = gemma_rmsnorm(x, self.weight.data, self.variance_epsilon)
        return out


class Gemma3RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        # Llama does x.to(float16) * w whilst Gemma3 is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"
