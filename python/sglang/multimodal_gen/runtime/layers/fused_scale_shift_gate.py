# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.layers.custom_op import CustomOp
from sglang.multimodal_gen.runtime.platforms import current_platform

_is_cuda = current_platform.is_cuda()
if _is_cuda:
    from sglang.jit_kernel.diffusion.triton.scale_shift import (
        fuse_layernorm_scale_shift_gate_select01_kernel,
        fuse_residual_layernorm_scale_shift_gate_select01_kernel,
    )


@CustomOp.register("fuse_layernorm_scale_shift_gate_select01")
class FusedLayerNormScaleShiftGateSelect01(CustomOp):
    """Fused layernorm + scale/shift + gate with binary index selection.

    CUDA path uses a Triton kernel; other platforms fall back to PyTorch ops.
    """

    def forward_cuda(
        self,
        x: torch.Tensor,
        weight: Optional[torch.Tensor],
        bias: Optional[torch.Tensor],
        scale0: torch.Tensor,
        shift0: torch.Tensor,
        gate0: torch.Tensor,
        scale1: torch.Tensor,
        shift1: torch.Tensor,
        gate1: torch.Tensor,
        index: torch.Tensor,
        eps: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not x.is_contiguous():
            x = x.contiguous()
        if not index.is_contiguous():
            index = index.contiguous()
        return fuse_layernorm_scale_shift_gate_select01_kernel(
            x,
            weight=weight,
            bias=bias,
            scale0=scale0.contiguous(),
            shift0=shift0.contiguous(),
            gate0=gate0.contiguous(),
            scale1=scale1.contiguous(),
            shift1=shift1.contiguous(),
            gate1=gate1.contiguous(),
            index=index,
            eps=eps,
        )

    def forward_hip(self, *args, **kwargs):
        return self.forward_native(*args, **kwargs)

    def forward_native(
        self,
        x: torch.Tensor,
        weight: Optional[torch.Tensor],
        bias: Optional[torch.Tensor],
        scale0: torch.Tensor,
        shift0: torch.Tensor,
        gate0: torch.Tensor,
        scale1: torch.Tensor,
        shift1: torch.Tensor,
        gate1: torch.Tensor,
        index: torch.Tensor,
        eps: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = index.to(dtype=torch.bool).unsqueeze(-1)
        shift = torch.where(idx, shift1.unsqueeze(1), shift0.unsqueeze(1))
        scale = torch.where(idx, scale1.unsqueeze(1), scale0.unsqueeze(1))
        gate = torch.where(idx, gate1.unsqueeze(1), gate0.unsqueeze(1))
        x = F.layer_norm(x, (x.shape[-1],), weight=weight, bias=bias, eps=eps)
        x = x * (1 + scale) + shift
        return x, gate


@CustomOp.register("fuse_residual_layernorm_scale_shift_gate_select01")
class FusedResidualLayerNormScaleShiftGateSelect01(CustomOp):
    """Fused residual + layernorm + scale/shift + gate with binary index selection.

    CUDA path uses a Triton kernel; other platforms fall back to PyTorch ops.
    """

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        residual_gate: torch.Tensor,
        weight: Optional[torch.Tensor],
        bias: Optional[torch.Tensor],
        scale0: torch.Tensor,
        shift0: torch.Tensor,
        gate0: torch.Tensor,
        scale1: torch.Tensor,
        shift1: torch.Tensor,
        gate1: torch.Tensor,
        index: torch.Tensor,
        eps: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not x.is_contiguous():
            x = x.contiguous()
        if not index.is_contiguous():
            index = index.contiguous()
        if not residual.is_contiguous():
            residual = residual.contiguous()
        if not residual_gate.is_contiguous():
            residual_gate = residual_gate.contiguous()
        return fuse_residual_layernorm_scale_shift_gate_select01_kernel(
            x,
            residual=residual,
            residual_gate=residual_gate,
            weight=weight,
            bias=bias,
            scale0=scale0.contiguous(),
            shift0=shift0.contiguous(),
            gate0=gate0.contiguous(),
            scale1=scale1.contiguous(),
            shift1=shift1.contiguous(),
            gate1=gate1.contiguous(),
            index=index,
            eps=eps,
        )

    def forward_hip(self, *args, **kwargs):
        return self.forward_native(*args, **kwargs)

    def forward_native(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        residual_gate: torch.Tensor,
        weight: Optional[torch.Tensor],
        bias: Optional[torch.Tensor],
        scale0: torch.Tensor,
        shift0: torch.Tensor,
        gate0: torch.Tensor,
        scale1: torch.Tensor,
        shift1: torch.Tensor,
        gate1: torch.Tensor,
        index: torch.Tensor,
        eps: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = index.to(dtype=torch.bool).unsqueeze(-1)
        shift = torch.where(idx, shift1.unsqueeze(1), shift0.unsqueeze(1))
        scale = torch.where(idx, scale1.unsqueeze(1), scale0.unsqueeze(1))
        gate = torch.where(idx, gate1.unsqueeze(1), gate0.unsqueeze(1))
        residual_out = residual_gate * x + residual
        x = F.layer_norm(
            residual_out, (residual_out.shape[-1],), weight=weight, bias=bias, eps=eps
        )
        x = x * (1 + scale) + shift
        return x, residual_out, gate
