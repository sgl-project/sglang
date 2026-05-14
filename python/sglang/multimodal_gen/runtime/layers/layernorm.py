# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/model_executor/layers/layernorm.py
"""Custom normalization layers."""

import os
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.jit_kernel.diffusion.qknorm_rope import (
    can_use_fused_inplace_qknorm_rope,
    fused_inplace_qknorm_rope,
)
from sglang.jit_kernel.diffusion.triton.rmsnorm_onepass import triton_one_pass_rms_norm
from sglang.jit_kernel.diffusion.triton.scale_shift import fuse_scale_shift_kernel
from sglang.jit_kernel.norm import can_use_fused_inplace_qknorm, fused_inplace_qknorm
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from sglang.multimodal_gen.runtime.layers.custom_op import CustomOp
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.utils.common import get_bool_env_var

_is_cuda = current_platform.is_cuda()
_is_hip = current_platform.is_hip()
_is_npu = current_platform.is_npu()
_is_musa = current_platform.is_musa()
_is_cpu = current_platform.is_cpu()
_is_xpu = current_platform.is_xpu()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

if _is_cuda or _is_xpu:
    from sgl_kernel import fused_add_rmsnorm, rmsnorm

if _is_npu:
    import torch_npu

if _is_musa:
    from sgl_kernel import fused_add_rmsnorm

if _use_aiter:
    from aiter import rmsnorm2d_fwd as rms_norm
    from aiter import rmsnorm2d_fwd_with_add as fused_add_rms_norm

if not _is_cpu:
    from sglang.jit_kernel.diffusion.triton.norm import norm_infer, rms_norm_fn


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
        elif _use_aiter:
            self._forward_method = self.forward_aiter

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
        x = x.reshape(-1, shape[-1])
        if residual is not None:
            residual_shape = residual.shape
            residual = residual.view(-1, shape[-1])

        if x.dtype == torch.float:
            if residual is None and self.variance_size_override is None:
                return self.forward_native(x).view(shape)
            out = self.forward_triton(x, residual)
            if residual is not None:
                return out[0].view(shape), out[1].view(residual_shape)
            out = out.view(shape)
            return out
        elif self.variance_size_override is not None:
            return self.forward_native(x, residual)
        elif residual is not None:
            fused_add_rmsnorm(x, residual, self.weight.data, self.variance_epsilon)
            return x.view(shape), residual.view(residual_shape)
        else:
            if x.shape[-1] <= 128:
                out = triton_one_pass_rms_norm(
                    x, self.weight.data, self.variance_epsilon
                )
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

    def forward_npu(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if residual is not None:
            out, _, residual_out = torch_npu.npu_add_rms_norm(
                residual, x, self.weight.data, self.variance_epsilon
            )
            return out, residual_out
        return torch_npu.npu_rms_norm(x, self.weight.data, self.variance_epsilon)[0]

    def forward_hip(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # ROCm builds of sgl-kernel do not expose rmsnorm custom ops yet.
        return self.forward_native(x, residual)

    def forward_aiter(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Fall back to the native fp32 path for cases aiter cannot serve:
        #   - fp32 input  (CK kernel is templated on fp16/bf16 only;
        #                  out.dtype check rejects fp32 with "not support output type: float")
        if (
            x.dtype not in (torch.float16, torch.bfloat16)
            or self.variance_size_override is not None
        ):
            return self.forward_native(x, residual)

        weight = self._get_weight(x.dtype)

        shape = x.shape
        x_2d = x.reshape(
            -1, shape[-1]
        )  # (bs, seq_len, hidden_size) -> (bs*seq_len, hidden_size)
        if not x_2d.is_contiguous():
            x_2d = x_2d.contiguous()

        if residual is not None:
            residual_shape = residual.shape
            residual_2d = residual.reshape(-1, shape[-1])
            if not residual_2d.is_contiguous():
                residual_2d = residual_2d.contiguous()
            output = torch.empty_like(x_2d)
            residual_out = torch.empty_like(x_2d)
            fused_add_rms_norm(
                output,
                x_2d,
                residual_2d,
                residual_out,
                weight,
                self.variance_epsilon,
            )
            return output.view(shape), residual_out.view(residual_shape)
        return rms_norm(x_2d, weight, self.variance_epsilon).view(shape)

    def _get_weight(self, dtype: torch.dtype) -> torch.Tensor:
        """Return weight matched to *dtype*.

        MUSA kernels require input and weight to share the same dtype,
        unlike CUDA kernels which may handle mixed dtypes internally.
        """
        weight = self.weight.data
        if weight.dtype != dtype:
            weight = weight.to(dtype=dtype)
        return weight

    def forward_musa(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        if residual is not None:
            residual_shape = residual.shape
            residual = residual.view(-1, shape[-1])

        if self.variance_size_override is not None:
            return self.forward_native(x, residual)
        elif residual is not None:
            # fused_add_rmsnorm requires contiguous inputs.
            if not x.is_contiguous():
                x = x.contiguous()
            if not residual.is_contiguous():
                residual = residual.contiguous()
            weight = self._get_weight(x.dtype)
            fused_add_rmsnorm(x, residual, weight, self.variance_epsilon)
            return x.view(shape), residual.view(residual_shape)
        else:
            weight = self._get_weight(x.dtype)
            out = F.rms_norm(x, (self.hidden_size,), weight, self.variance_epsilon)
        out = out.view(shape)
        return out

    def forward_xpu(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        if residual is not None:
            residual_shape = residual.shape
            residual = residual.view(-1, shape[-1])

        if self.variance_size_override is not None:
            return self.forward_native(x, residual)
        elif residual is not None:
            fused_add_rmsnorm(x, residual, self.weight.data, self.variance_epsilon)
            return x.view(shape), residual.view(residual_shape)
        else:
            out = rmsnorm(x, self.weight.data, self.variance_epsilon)
        out = out.view(shape)
        return out

    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, eps={self.variance_epsilon}"


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

    @torch.compile(backend="inductor", disable=current_platform.is_npu())
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

    def forward_musa(self, x: torch.Tensor):
        return F.layer_norm(x, (self.hidden_size,), self.weight, self.bias, self.eps)

    def extra_repr(self) -> str:
        s = f"hidden_size={self.weight.data.size(0)}"
        s += f", eps={self.variance_epsilon}"
        return s


# adapted from Diffusers: https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/normalization.py
# NOTE(will): Needed to match behavior of diffusers and wan2.1 even while using
# FSDP's MixedPrecisionPolicy
class FP32LayerNorm(nn.LayerNorm):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        origin_dtype = inputs.dtype
        device = inputs.device
        return F.layer_norm(
            inputs.float(),
            self.normalized_shape,
            self.weight.float().to(device=device) if self.weight is not None else None,
            self.bias.float().to(device=device) if self.bias is not None else None,
            self.eps,
        ).to(origin_dtype)


################################################################################
# Fused norm kernel
################################################################################
def _ensure_contiguous(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    return tensor.contiguous() if tensor is not None else None


class _ScaleResidualNormScaleShift(CustomOp):
    """
    Fused kernel that combines:
    1. residual_out = residual + gate * x
    2. normed = layernorm(residual_out) or rmsnorm(residual_out)
    3. out = normed * (1 + scale) + shift
    compute_dtype is always fp32 for higher precision.
    """

    norm_type: str

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        elementwise_affine: bool = False,
        dtype: torch.dtype = torch.float32,
        prefix: str = "",
    ):
        super().__init__()
        self.eps = eps
        self.dtype = dtype
        if self.norm_type == "rms":
            self.norm = RMSNorm(hidden_size, eps=eps, dtype=dtype)
        elif self.norm_type == "layer":
            self.norm = FP32LayerNorm(
                hidden_size, elementwise_affine=elementwise_affine, eps=eps, dtype=dtype
            )
        else:
            raise NotImplementedError(f"Norm type {self.norm_type} not implemented")

    def forward_cuda(
        self,
        residual: torch.Tensor,
        x: torch.Tensor,
        gate: torch.Tensor | int,
        shift: torch.Tensor,
        scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.shape[-1] % 256 != 0 and x.shape[-1] <= 8192:
            import warnings

            warnings.warn(
                "FusedScaleResidualNormScaleShift cuda not available, using native fallback",
                stacklevel=2,
            )
            return self.forward_native(residual, x, gate, shift, scale)

        from sglang.jit_kernel.diffusion.cutedsl.scale_residual_norm_scale_shift import (
            fused_scale_residual_norm_scale_shift,
        )

        if isinstance(gate, int) and gate != 1:
            raise ValueError(
                f"Only gate value of 1 is supported for int type, but got {gate}"
            )

        return fused_scale_residual_norm_scale_shift(
            residual.contiguous(),
            x.contiguous(),
            gate.contiguous() if isinstance(gate, torch.Tensor) else None,
            _ensure_contiguous(getattr(self.norm, "weight", None)),
            _ensure_contiguous(getattr(self.norm, "bias", None)),
            scale.contiguous(),
            shift.contiguous(),
            self.norm_type,
            self.eps,
        )

    def forward_hip(self, *args, **kwargs):
        # ROCm does not support CUDA/CUTLASS-based fused kernels yet,
        # so we fall back to the native PyTorch implementation.
        return self.forward_native(*args, **kwargs)

    def forward_musa(self, *args, **kwargs):
        # MUSA does not support CUDA/CUTLASS-based fused kernels yet,
        # so we fall back to the native PyTorch implementation.
        return self.forward_native(*args, **kwargs)

    def forward_xpu(self, *args, **kwargs):
        # XPU does not support CUDA/CUTLASS-based fused kernels yet,
        # so we fall back to the native PyTorch implementation.
        return self.forward_native(*args, **kwargs)

    def forward_native(
        self,
        residual: torch.Tensor,
        x: torch.Tensor,
        gate: torch.Tensor | int,
        shift: torch.Tensor,
        scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # x.shape: [batch_size, seq_len, inner_dim]
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
                # gate.shape: [batch_size, 1, inner_dim]
                residual_output = residual + x * gate
        else:
            raise ValueError(f"Gate type {type(gate)} not supported")
        normalized = self.norm(residual_output)
        modulated = fuse_scale_shift_kernel(normalized, scale, shift)
        return modulated, residual_output

    def forward_npu(
        self,
        residual: torch.Tensor,
        x: torch.Tensor,
        gate: torch.Tensor | int,
        shift: torch.Tensor,
        scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from sgl_kernel_npu.norm.scale_shift import fused_scale_shift

        # x.shape: [batch_size, seq_len, inner_dim]
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
                # gate.shape: [batch_size, 1, inner_dim]
                residual_output = residual + x * gate
        else:
            raise ValueError(f"Gate type {type(gate)} not supported")
        normalized = self.norm(residual_output)
        modulated = fused_scale_shift(normalized, scale, shift)
        return modulated, residual_output


class ScaleResidualLayerNormScaleShift(_ScaleResidualNormScaleShift):
    norm_type = "layer"


class ScaleResidualRMSNormScaleShift(_ScaleResidualNormScaleShift):
    norm_type = "rms"


class _NormScaleShift(CustomOp):
    """
    Fused kernel that combines:
    1. normed = layernorm(x) or rmsnorm(x)
    2. out = normed * (1 + scale) + shift
    compute_dtype is always fp32 for higher precision.
    """

    norm_type: str

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        elementwise_affine: bool = False,
        dtype: torch.dtype = torch.float32,
        prefix: str = "",
    ):
        super().__init__()
        self.eps = eps
        if self.norm_type == "rms":
            self.norm = RMSNorm(hidden_size, eps=eps, dtype=dtype)
        elif self.norm_type == "layer":
            self.norm = FP32LayerNorm(
                hidden_size, elementwise_affine=elementwise_affine, eps=eps, dtype=dtype
            )
        else:
            raise NotImplementedError(f"Norm type {self.norm_type} not implemented")

    def forward_cuda(
        self, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        if x.shape[-1] % 256 != 0 and x.shape[-1] <= 8192:
            import warnings

            warnings.warn(
                "FusedNormScaleShift cuda not available, using native fallback",
                stacklevel=2,
            )
            return self.forward_native(x, shift, scale)

        from sglang.jit_kernel.diffusion.cutedsl.scale_residual_norm_scale_shift import (
            fused_norm_scale_shift,
        )

        return fused_norm_scale_shift(
            x.contiguous(),
            _ensure_contiguous(getattr(self.norm, "weight", None)),
            _ensure_contiguous(getattr(self.norm, "bias", None)),
            scale.contiguous(),
            shift.contiguous(),
            self.norm_type,
            self.eps,
        )

    def forward_hip(self, *args, **kwargs):
        # ROCm does not support CUDA/CUTLASS-based fused kernels yet,
        # so we fall back to the native PyTorch implementation.
        return self.forward_native(*args, **kwargs)

    def forward_musa(self, *args, **kwargs):
        # MUSA does not support CUDA/CUTLASS-based fused kernels yet,
        # so we fall back to the native PyTorch implementation.
        return self.forward_native(*args, **kwargs)

    def forward_xpu(self, *args, **kwargs):
        # XPU does not support CUDA/CUTLASS-based fused kernels yet,
        # so we fall back to the native PyTorch implementation.
        return self.forward_native(*args, **kwargs)

    def forward_native(
        self, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        normalized = self.norm(x)
        modulated = fuse_scale_shift_kernel(normalized, scale, shift)
        return modulated.to(x.dtype)

    def forward_npu(
        self, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        from sgl_kernel_npu.norm.scale_shift import fused_scale_shift

        normalized = self.norm(x)
        modulated = fused_scale_shift(normalized, scale, shift)
        return modulated.to(x.dtype)


class LayerNormScaleShift(_NormScaleShift):
    norm_type = "layer"


class RMSNormScaleShift(_NormScaleShift):
    norm_type = "rms"


################################################################################
# NormTanhMulAdd
# y = norm(x) * tanh(scale) + shift (where norm is layernorm or rmsnorm)
# See details in norm_tanh_mul_add_norm_scale.py
################################################################################
class _NormTanhMulAdd(CustomOp):
    norm_type: str

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        affine: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.eps = eps
        if self.norm_type == "rms":
            self.norm = RMSNorm(hidden_size, eps=eps, dtype=dtype)
        elif self.norm_type == "layer":
            self.norm = FP32LayerNorm(
                hidden_size, elementwise_affine=affine, eps=eps, dtype=dtype
            )
        else:
            raise NotImplementedError(f"Norm type {self.norm_type} not implemented")

    def forward_cuda(
        self, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor
    ) -> torch.Tensor:
        if x.shape[-1] % 256 != 0 and x.shape[-1] <= 8192:
            import warnings

            warnings.warn(
                "FusedNormScaleShift cuda not available, using native fallback",
                stacklevel=2,
            )
            return self.forward_native(x, scale, shift)

        from sglang.jit_kernel.diffusion.cutedsl.norm_tanh_mul_add_norm_scale import (
            fused_norm_tanh_mul_add,
        )

        x, scale, shift = x.contiguous(), scale.contiguous(), shift.contiguous()
        weight = _ensure_contiguous(getattr(self.norm, "weight", None))
        bias = _ensure_contiguous(getattr(self.norm, "bias", None))
        return fused_norm_tanh_mul_add(
            x,
            weight,
            bias,
            scale,
            shift,
            self.norm_type,
            self.eps,
        )

    def forward_hip(self, *args, **kwargs):
        # Fallback to native because ROCm does not support CuTeDSL.
        return self.forward_native(*args, **kwargs)

    def forward_native(
        self, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor
    ) -> torch.Tensor:
        y = self.norm(x) * torch.tanh(scale) + shift
        return y.to(x.dtype)


class LayerNormTanhMulAdd(_NormTanhMulAdd):
    norm_type = "layer"


class RMSNormTanhMulAdd(_NormTanhMulAdd):
    norm_type = "rms"


def apply_qk_norm(
    q: torch.Tensor,
    k: torch.Tensor,
    q_norm: "RMSNorm",
    k_norm: "RMSNorm",
    head_dim: int,
    allow_inplace: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply QK normalization for query and key tensors.

    Uses JIT fused inplace kernel when available, falls back to standard RMSNorm.
    """

    batch_size = q.size(0)
    q_eps = q_norm.variance_epsilon
    k_eps = k_norm.variance_epsilon
    # Only try fused path on CUDA and when it won't introduce implicit copies.
    if (
        _is_cuda
        and allow_inplace
        and (q_eps == k_eps)
        and q.dtype in (torch.float16, torch.bfloat16)
        and q_norm.weight.dtype == q.dtype
        and k_norm.weight.dtype == k.dtype
        and can_use_fused_inplace_qknorm(head_dim, q.dtype)
    ):
        fused_inplace_qknorm(
            q=q.view(batch_size, -1, head_dim),
            k=k.view(batch_size, -1, head_dim),
            q_weight=q_norm.weight,
            k_weight=k_norm.weight,
            head_dim=head_dim,
            eps=q_eps,
        )
        return q, k

    q_shape = q.shape
    k_shape = k.shape
    q_out = q_norm(q.view(-1, head_dim)).view(q_shape)
    k_out = k_norm(k.view(-1, head_dim)).view(k_shape)
    return q_out, k_out


def apply_qk_norm_with_optional_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_norm: "RMSNorm",
    k_norm: "RMSNorm",
    head_dim: int,
    cos_sin_cache: Optional[torch.Tensor] = None,
    *,
    is_neox: bool = False,
    positions: Optional[torch.Tensor] = None,
    position_offset: int = 0,
    allow_inplace: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply QK RMSNorm and optionally RoPE when a cos/sin cache is provided."""

    if cos_sin_cache is None:
        return apply_qk_norm(
            q=q,
            k=k,
            q_norm=q_norm,
            k_norm=k_norm,
            head_dim=head_dim,
            allow_inplace=allow_inplace,
        )

    return apply_qk_norm_rope(
        q=q,
        k=k,
        q_norm=q_norm,
        k_norm=k_norm,
        head_dim=head_dim,
        cos_sin_cache=cos_sin_cache,
        is_neox=is_neox,
        positions=positions,
        position_offset=position_offset,
        allow_inplace=allow_inplace,
    )


def apply_qk_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_norm: "RMSNorm",
    k_norm: "RMSNorm",
    head_dim: int,
    cos_sin_cache: torch.Tensor,
    *,
    is_neox: bool = False,
    positions: Optional[torch.Tensor] = None,
    position_offset: int = 0,
    allow_inplace: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply QK RMSNorm followed by RoPE, fusing both on supported CUDA shapes."""

    from sglang.multimodal_gen.runtime.layers.rotary_embedding import (
        apply_flashinfer_rope_qk_inplace,
    )

    if q.dim() != 4 or k.dim() != 4:
        raise ValueError(
            f"apply_qk_norm_rope expects 4D q/k tensors, got q:{tuple(q.shape)} k:{tuple(k.shape)}"
        )
    if q.shape != k.shape:
        raise ValueError(
            f"apply_qk_norm_rope expects q/k to have the same shape, got {q.shape} vs {k.shape}"
        )

    batch_size, seq_len, _, _ = q.shape
    q_eps = q_norm.variance_epsilon
    k_eps = k_norm.variance_epsilon
    rope_dim = cos_sin_cache.size(-1)
    fused_enabled = os.getenv("SGLANG_ENABLE_FUSED_QKNORM_ROPE", "1").lower() not in {
        "0",
        "false",
        "off",
        "no",
    }

    if positions is None:
        pos_1d = torch.arange(
            position_offset,
            position_offset + seq_len,
            device=q.device,
            dtype=torch.int64,
        )
        positions = pos_1d if batch_size == 1 else pos_1d.repeat(batch_size)
    else:
        if positions.dim() != 1 or positions.numel() != batch_size * seq_len:
            raise ValueError(
                f"positions must be 1D of length {batch_size * seq_len}, got shape={tuple(positions.shape)}"
            )

    if (
        fused_enabled
        and _is_cuda
        and allow_inplace
        and (q_eps == k_eps)
        and q.dtype in (torch.float16, torch.bfloat16)
        and q_norm.weight.dtype == q.dtype
        and k_norm.weight.dtype == k.dtype
        and q.is_contiguous()
        and k.is_contiguous()
        and can_use_fused_inplace_qknorm_rope(head_dim, rope_dim, is_neox, q.dtype)
    ):
        fused_inplace_qknorm_rope(
            q=q.reshape(-1, q.shape[-2], head_dim),
            k=k.reshape(-1, k.shape[-2], head_dim),
            q_weight=q_norm.weight,
            k_weight=k_norm.weight,
            cos_sin_cache=cos_sin_cache,
            positions=positions,
            is_neox=is_neox,
            eps=q_eps,
            head_dim=head_dim,
            rope_dim=rope_dim,
        )
        return q, k

    q, k = apply_qk_norm(
        q=q,
        k=k,
        q_norm=q_norm,
        k_norm=k_norm,
        head_dim=head_dim,
        allow_inplace=allow_inplace,
    )
    return apply_flashinfer_rope_qk_inplace(
        q=q,
        k=k,
        cos_sin_cache=cos_sin_cache,
        head_size=head_dim,
        is_neox=is_neox,
        positions=positions,
    )


def apply_rmsnorm_tanh_mul_add(
    x: torch.Tensor,
    gate: torch.Tensor,
    residual: torch.Tensor,
    norm: "RMSNorm",
) -> torch.Tensor:
    """Compute residual + tanh(gate) * rmsnorm(x), with a fused CUDA fast path."""
    if get_bool_env_var("SGLANG_ENABLE_DETERMINISTIC_INFERENCE"):
        return residual + torch.tanh(gate) * norm(x)

    if _is_cuda and x.is_cuda and x.shape[-1] % 256 == 0 and x.shape[-1] <= 8192:
        from sglang.jit_kernel.diffusion.cutedsl.norm_tanh_mul_add_norm_scale import (
            fused_norm_tanh_mul_add,
        )

        return fused_norm_tanh_mul_add(
            x.contiguous(),
            norm.weight.data.contiguous(),
            None,
            gate.contiguous(),
            residual.contiguous(),
            "rms",
            norm.variance_epsilon,
        )

    return residual + torch.tanh(gate) * norm(x)


def tensor_parallel_rms_norm(x: torch.Tensor, norm: "RMSNorm") -> torch.Tensor:
    tp_rank = get_tensor_model_parallel_rank()
    tp_size = get_tensor_model_parallel_world_size()
    src_dtype = x.dtype
    weight = norm.weight.tensor_split(tp_size)[tp_rank].float()
    x_fp32 = x.float()
    if _is_npu:
        from sgl_kernel_npu.norm.rmsnorm_split import fused_rsqrt_mul, fused_variance

        variance = fused_variance(x_fp32)
    else:
        variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)

    variance = get_tp_group().all_reduce(
        variance, op=torch._C._distributed_c10d.ReduceOp.AVG
    )

    if _is_npu:
        output = fused_rsqrt_mul(x_fp32, variance, weight, norm.variance_epsilon)
    else:
        output = x_fp32 * torch.rsqrt(variance + norm.variance_epsilon) * weight
    return output.to(dtype=src_dtype)
