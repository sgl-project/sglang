# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/model_executor/layers/layernorm.py
"""Custom normalization layers."""
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.layers.custom_op import CustomOp
from sglang.multimodal_gen.runtime.layers.triton_ops import (
    fuse_scale_shift_kernel,
    norm_infer,
    rms_norm_fn,
)
from sglang.multimodal_gen.runtime.utils.common import (
    get_bool_env_var,
    is_cpu,
    is_cuda,
    is_hip,
    is_npu,
    is_xpu,
)

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_npu = is_npu()
_is_cpu = is_cpu()
_is_xpu = is_xpu()

from sgl_kernel import fused_add_rmsnorm, rmsnorm


# Copied and adapted from sglang
@CustomOp.register("rms_norm")
class RMSNorm(CustomOp):
    """Root mean square normalization.

    Computes x -> w * x / sqrt(E[x^2] + eps) where w is the learned weight.
    Refer to https://arxiv.org/abs/1910.07467
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        dtype: torch.dtype = torch.float32,
        var_hidden_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.hidden_size = hidden_size
        self.variance_size_override = (
            None if var_hidden_size == hidden_size else var_hidden_size
        )
        if get_bool_env_var("SGLANG_ENABLE_DETERMINISTIC_INFERENCE"):
            self._forward_method = self.forward_native

    def forward_triton(self, x: torch.Tensor, residual: Optional[torch.Tensor] = None):
        return rms_norm_fn(
            x, self.weight, bias=None, residual=residual, eps=self.variance_epsilon
        )

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        shape = x.shape
        x = x.view(-1, shape[-1])
        if residual is not None:
            residual_shape = residual.shape
            residual = residual.view(-1, shape[-1])

        if x.dtype == torch.float:
            # fp32
            out = self.forward_triton(x, residual)
        elif self.variance_size_override is not None:
            return self.forward_native(x, residual)
        elif residual is not None:
            fused_add_rmsnorm(x, residual, self.weight.data, self.variance_epsilon)
            return x.view(shape), residual.view(residual_shape)
        else:
            out = rmsnorm(x, self.weight.data, self.variance_epsilon)
        out = out.view(shape)
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

        hidden_size = x.shape[-1]
        if hidden_size != self.hidden_size:
            raise ValueError(
                "Expected hidden_size to be "
                f"{self.hidden_size}, but found: {hidden_size}"
            )

        if self.variance_size_override is None:
            x_var = x
        else:
            if hidden_size < self.variance_size_override:
                raise ValueError(
                    "Expected hidden_size to be at least "
                    f"{self.variance_size_override}, but found: {hidden_size}"
                )

            x_var = x[..., : self.variance_size_override]

        variance = x_var.pow(2).mean(dim=-1, keepdim=True)
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
        return self.forward_native(x, residual)

    def extra_repr(self) -> str:
        s = f"hidden_size={self.weight.data.size(0)}"
        s += f", eps={self.variance_epsilon}"
        return s


# Copied and adapted from sglang
@CustomOp.register("layer_norm")
class LayerNorm(CustomOp):
    def __init__(
        self,
        hidden_size: int,
        eps=1e-5,
        bias: bool = True,
        elementwise_affine=True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.eps = eps
        factory_kwargs = {"device": device, "dtype": dtype}
        self.hidden_size = hidden_size
        if elementwise_affine:
            self.weight = torch.nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
            self.bias = (
                torch.nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
                if bias
                else None
            )
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
            # Lazy cache for ones vector (not a registered buffer to avoid FSDP/meta issues)
            self._weight_fallback_cache = None

    def _get_weight_fallback(self, x: torch.Tensor) -> torch.Tensor:
        wf = getattr(self, "_weight_fallback_cache", None)
        if (
            wf is None
            or wf.device != x.device
            or wf.dtype != x.dtype
            or wf.numel() != self.hidden_size
        ):
            wf = torch.ones(self.hidden_size, device=x.device, dtype=x.dtype)
            self._weight_fallback_cache = wf
        return wf

    def forward_triton(self, x: torch.Tensor):
        # Fast inference kernel without residual/dropout branches
        return norm_infer(
            x.view(-1, self.hidden_size),
            self.weight,
            self.bias,
            eps=self.eps,
            is_rms_norm=False,
        ).view(x.shape)

    def forward_cuda(
        self,
        x: torch.Tensor,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        shape = x.shape
        x = x.view(-1, self.hidden_size)
        return self.forward_triton(x).view(shape)

    @torch.compile(backend="inductor")
    def forward_native(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        input_dtype = x.dtype
        mean = x.mean(-1, keepdim=True)
        variance = (x - mean).pow(2).mean(-1, keepdim=True)
        x = (x - mean) * torch.rsqrt(variance + self.eps)
        if self.weight is not None:
            x = self.weight * x
        # if no affine, this is a no-op
        if self.bias is not None:
            x = x + self.bias
        return x.to(input_dtype)

    def forward_cpu(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        return self.forward_native(x, residual)

    def extra_repr(self) -> str:
        s = f"hidden_size={self.weight.data.size(0)}"
        s += f", eps={self.variance_epsilon}"
        return s


class ScaleResidual(nn.Module):
    """
    Applies gated residual connection.
    """

    def __init__(self, prefix: str = ""):
        super().__init__()

    def forward(
        self, residual: torch.Tensor, x: torch.Tensor, gate: torch.Tensor
    ) -> torch.Tensor:
        """Apply gated residual connection."""
        # x.shape: [batch_size, seq_len, inner_dim]
        if gate.dim() == 4:
            # gate.shape: [batch_size, num_frames, 1, inner_dim]
            num_frames = gate.shape[1]
            frame_seqlen = x.shape[1] // num_frames
            return residual + (
                x.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * gate
            ).flatten(1, 2)
        else:
            # gate.shape: [batch_size, 1, inner_dim]
            return residual + x * gate


# adapted from Diffusers: https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/normalization.py
# NOTE(will): Needed to match behavior of diffusers and wan2.1 even while using
# FSDP's MixedPrecisionPolicy
class FP32LayerNorm(nn.LayerNorm):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        origin_dtype = inputs.dtype
        return F.layer_norm(
            inputs.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        ).to(origin_dtype)


class ScaleResidualLayerNormScaleShift(nn.Module):
    """
    Fused operation that combines:
    1. Gated residual connection
    2. LayerNorm
    3. Scale and shift operations

    This reduces memory bandwidth by combining memory-bound operations.
    """

    def __init__(
        self,
        hidden_size: int,
        norm_type: str = "rms",
        eps: float = 1e-6,
        elementwise_affine: bool = False,
        dtype: torch.dtype = torch.float32,
        compute_dtype: torch.dtype | None = None,
        prefix: str = "",
    ):
        super().__init__()
        if norm_type == "rms":
            self.norm = RMSNorm(
                hidden_size, has_weight=elementwise_affine, eps=eps, dtype=dtype
            )
        elif norm_type == "layer":
            if compute_dtype == torch.float32:
                self.norm = FP32LayerNorm(
                    hidden_size, elementwise_affine=elementwise_affine, eps=eps
                )
            else:
                self.norm = LayerNorm(
                    hidden_size,
                    elementwise_affine=elementwise_affine,
                    eps=eps,
                    dtype=dtype,
                )
        else:
            raise NotImplementedError(f"Norm type {norm_type} not implemented")

    def forward(
        self,
        residual: torch.Tensor,
        x: torch.Tensor,
        gate: torch.Tensor | int,
        shift: torch.Tensor,
        scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply gated residual connection, followed by layernorm and
        scale/shift in a single fused operation.

        Returns:
            Tuple containing:
            - normalized and modulated output of shape: [batch_size, seq_len, inner_dim]
            - residual value (value after residual connection
              but before normalization)
        """
        # x.shape: [batch_size, seq_len, inner_dim]
        # Apply residual connection with gating
        if isinstance(gate, int):
            # used by cross-attention, should be 1
            assert gate == 1
            residual_output = residual + x
        elif isinstance(gate, torch.Tensor):
            if gate.dim() == 4:
                # gate.shape: [batch_size, num_frames, 1, inner_dim]
                num_frames = gate.shape[1]
                frame_seqlen = x.shape[1] // num_frames
                residual_output = residual + (
                    x.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * gate
                ).flatten(1, 2)
            else:
                # used by bidirectional self attention
                # gate.shape: [batch_size, 1, inner_dim]
                residual_output = residual + x * gate
        else:
            raise ValueError(f"Gate type {type(gate)} not supported")
        # residual_output.shape: [batch_size, seq_len, inner_dim]

        # Apply normalization
        normalized = self.norm(residual_output)

        # modulated = fused_scale_shift(
        #     normalized,
        #     scale,
        #     shift,
        # )
        modulated = fuse_scale_shift_kernel(
            normalized,
            scale,
            shift,
        )
        return modulated, residual_output


class LayerNormScaleShift(nn.Module):
    """
    Fused operation that combines LayerNorm with scale and shift operations.
    This reduces memory bandwidth by combining memory-bound operations.
    """

    def __init__(
        self,
        hidden_size: int,
        norm_type: str = "rms",
        eps: float = 1e-6,
        elementwise_affine: bool = False,
        dtype: torch.dtype = torch.float32,
        compute_dtype: torch.dtype | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.compute_dtype = compute_dtype
        if norm_type == "rms":
            self.norm = RMSNorm(hidden_size, has_weight=elementwise_affine, eps=eps)
        elif norm_type == "layer":
            if self.compute_dtype == torch.float32:
                self.norm = FP32LayerNorm(
                    hidden_size, elementwise_affine=elementwise_affine, eps=eps
                )
            else:
                self.norm = nn.LayerNorm(
                    hidden_size,
                    elementwise_affine=elementwise_affine,
                    eps=eps,
                    dtype=dtype,
                )
        else:
            raise NotImplementedError(f"Norm type {norm_type} not implemented")

    def forward(
        self, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        """Apply ln followed by scale and shift in a single fused operation."""
        # x.shape: [batch_size, seq_len, inner_dim]
        normalized = self.norm(x)
        if self.compute_dtype == torch.float32:
            normalized = normalized.float()

        if scale.dim() == 4:
            # scale.shape: [batch_size, num_frames, 1, inner_dim]
            num_frames = scale.shape[1]
            frame_seqlen = normalized.shape[1] // num_frames
            output = (
                normalized.unflatten(dim=1, sizes=(num_frames, frame_seqlen))
                * (1.0 + scale)
                + shift
            ).flatten(1, 2)
        else:
            # scale.shape: [batch_size, 1, inner_dim]
            # shift.shape: [batch_size, 1, inner_dim]
            output = normalized * (1.0 + scale) + shift

        if self.compute_dtype == torch.float32:
            output = output.to(x.dtype)

        return output
