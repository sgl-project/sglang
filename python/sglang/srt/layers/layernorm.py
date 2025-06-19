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
"""Fused operators for normalization layers."""

import logging
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from sglang.srt.custom_op import CustomOp
from sglang.srt.utils import (
    cpu_has_amx_support,
    get_bool_env_var,
    is_cpu,
    is_cuda,
    is_hip,
    is_hpu,
    is_npu,
)

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_hpu = is_hpu()
_is_npu = is_npu()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
_is_cpu_amx_available = cpu_has_amx_support()
_is_cpu = is_cpu()

if _is_cuda:
    from sgl_kernel import (
        fused_add_rmsnorm,
        gemma_fused_add_rmsnorm,
        gemma_rmsnorm,
        rmsnorm,
    )

if _use_aiter:
    from aiter import rmsnorm2d_fwd as rms_norm
    from aiter import rmsnorm2d_fwd_with_add as fused_add_rms_norm
elif _is_hip:
    from vllm._custom_ops import fused_add_rms_norm, rms_norm

logger = logging.getLogger(__name__)


class RMSNorm(CustomOp):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        if _use_aiter:
            self._forward_method = self.forward_aiter

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if residual is not None:
            fused_add_rmsnorm(x, residual, self.weight.data, self.variance_epsilon)
            return x, residual
        out = rmsnorm(x, self.weight.data, self.variance_epsilon)
        return out

    def forward_aiter(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if residual is not None:
            residual_out = torch.empty_like(x)
            output = torch.empty_like(x)
            fused_add_rms_norm(
                output,
                x,
                residual,
                residual_out,
                self.weight.data,
                self.variance_epsilon,
            )
            return output, residual_out
        return rms_norm(x, self.weight.data, self.variance_epsilon)

    def forward_hip(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if not x.is_contiguous():
            # NOTE: Remove this if aiter kernel supports discontinuous input
            x = x.contiguous()
        if residual is not None:
            fused_add_rms_norm(x, residual, self.weight.data, self.variance_epsilon)
            return x, residual
        out = torch.empty_like(x)
        rms_norm(out, x, self.weight.data, self.variance_epsilon)
        return out

    def forward_native(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if not x.is_contiguous():
            x = x.contiguous()
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

    def forward_cpu(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if _is_cpu_amx_available:
            if residual is not None:
                torch.ops.sgl_kernel.fused_add_rmsnorm_cpu(
                    x, residual, self.weight.data, self.variance_epsilon
                )
                return x, residual
            return torch.ops.sgl_kernel.rmsnorm_cpu(
                x, self.weight.data, self.variance_epsilon
            )
        else:
            return self.forward_native(x, residual)


class GemmaRMSNorm(CustomOp):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

        # Re-dispatch
        if _is_hip:
            self._forward_method = self.forward_native

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


if not (
    _is_cuda or _is_hip or _is_hpu or _is_npu or (_is_cpu and _is_cpu_amx_available)
):
    logger.info(
        "sgl-kernel layernorm implementation is not available on current platform. Fallback to other kernel libraries."
    )
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNorm
