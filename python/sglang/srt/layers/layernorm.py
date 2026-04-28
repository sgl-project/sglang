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
import torch.nn.functional as F

from sglang.srt.batch_invariant_ops import (
    is_batch_invariant_mode_enabled,
    rms_norm_batch_invariant,
)
from sglang.srt.environ import envs
from sglang.srt.layers.utils import MultiPlatformOp
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import (
    cpu_has_amx_support,
    get_bool_env_var,
    is_cpu,
    is_cuda,
    is_flashinfer_available,
    is_hip,
    is_musa,
    is_npu,
    is_xpu,
)

_is_cuda = is_cuda()
_is_flashinfer_available = is_flashinfer_available()
_is_hip = is_hip()
_is_musa = is_musa()
_is_npu = is_npu()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
_is_cpu_amx_available = cpu_has_amx_support()
_is_cpu = is_cpu()
_is_xpu = is_xpu()
_flashinfer_layernorm_available = False

if _is_cuda or _is_xpu or _is_musa:
    if _is_flashinfer_available:
        try:
            from flashinfer.norm import layernorm

            _flashinfer_layernorm_available = True
        except (ImportError, AttributeError):
            _flashinfer_layernorm_available = False
    else:
        _flashinfer_layernorm_available = False

    from sgl_kernel import (
        fused_add_rmsnorm,
        gemma_fused_add_rmsnorm,
        gemma_rmsnorm,
        rmsnorm,
    )
_has_aiter_layer_norm = False
_has_vllm_rms_norm = False
if _use_aiter:
    from aiter import layernorm2d_fwd as layer_norm
    from aiter import rmsnorm2d_fwd as rms_norm
    from aiter import rmsnorm2d_fwd_with_add as fused_add_rms_norm

    _has_aiter_layer_norm = True  # aiter provides the layer_norm functions
    _has_vllm_rms_norm = True  # aiter provides the rms_norm functions
elif _is_hip:
    try:
        from vllm._custom_ops import fused_add_rms_norm, rms_norm

        _has_vllm_rms_norm = True
    except ImportError:
        # Fallback: vllm not available, will use forward_native
        _has_vllm_rms_norm = False

if _is_cuda:
    # HF-semantics RMSNorm kernel (JIT-compiled).  Used when `cast_x_before_out_mul=True`
    # (the transformers backend path) to produce outputs that are numerically identical
    # to HuggingFace `LlamaRMSNorm`: the cast from fp32 to the activation dtype happens
    # BEFORE the weight multiply, so the multiply is done in the narrow dtype.
    _jit_rmsnorm_hf_available = False
    try:
        from sglang.jit_kernel.rmsnorm_hf import (
            is_supported_rmsnorm_hf_hidden_size,
        )
        from sglang.jit_kernel.rmsnorm_hf import rmsnorm_hf as _jit_rmsnorm_hf

        _jit_rmsnorm_hf_available = True
    except ImportError:

        def is_supported_rmsnorm_hf_hidden_size(d: int) -> bool:
            return False

        _jit_rmsnorm_hf = None


logger = logging.getLogger(__name__)

if _is_npu:
    import torch_npu
    from sgl_kernel_npu.norm.add_rmsnorm_bias import add_gemma_rms_norm


def _forward_with_allreduce_fusion(
    norm_module,
    x: torch.Tensor,
    residual: Optional[torch.Tensor],
    post_residual_addition: Optional[torch.Tensor],
    weight: torch.Tensor,
    use_attn_tp_group: bool = True,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Shared allreduce-fused RMSNorm logic usable by any norm."""
    if residual is not None:
        from sglang.srt.distributed import (
            get_attn_tensor_model_parallel_world_size,
            get_moe_expert_parallel_world_size,
            get_moe_tensor_parallel_world_size,
            tensor_model_parallel_all_reduce,
            tensor_model_parallel_fused_allreduce_rmsnorm,
        )
        from sglang.srt.layers.flashinfer_comm_fusion import (
            flashinfer_allreduce_residual_rmsnorm,
        )

        if use_attn_tp_group:
            world_size = get_attn_tensor_model_parallel_world_size()
        else:
            if get_moe_expert_parallel_world_size() > 1:
                world_size = get_moe_expert_parallel_world_size()
            else:
                world_size = get_moe_tensor_parallel_world_size()

        if world_size > 1:
            if post_residual_addition is not None:
                residual = residual + post_residual_addition

            # Prefer AITER fused AR+RMSNorm when enabled on AMD.
            if _use_aiter:
                fused_result = tensor_model_parallel_fused_allreduce_rmsnorm(
                    x, residual, weight, norm_module.variance_epsilon
                )
                if fused_result is not None:
                    return fused_result
            else:
                fused_result = flashinfer_allreduce_residual_rmsnorm(
                    input_tensor=x,
                    residual=residual,
                    weight=weight,
                    eps=norm_module.variance_epsilon,
                    use_attn_tp_group=use_attn_tp_group,
                )
                if fused_result[0] is not None:
                    return fused_result

            # For AITER route, preserve correctness when fused path is unavailable.
            if _use_aiter and get_global_server_args().enable_aiter_allreduce_fusion:
                x = tensor_model_parallel_all_reduce(x)
                return norm_module.forward(x, residual, None)

    return norm_module.forward(x, residual, post_residual_addition)


class RMSNorm(MultiPlatformOp):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        var_hidden_size: Optional[int] = None,
        cast_x_before_out_mul: bool = False,
        fp32_residual: bool = False,
        has_weight: bool = True,
        weight_dtype: Optional = None,
        override_orig_dtype: Optional = None,
    ) -> None:
        super().__init__()
        self.has_weight = has_weight
        self.cast_x_before_out_mul = cast_x_before_out_mul
        self.fp32_residual = fp32_residual
        self.override_orig_dtype = override_orig_dtype
        if self.has_weight:
            self.weight = nn.Parameter(torch.ones(hidden_size, dtype=weight_dtype))
        else:
            self.weight = torch.ones(hidden_size, dtype=weight_dtype)
        self.variance_epsilon = eps
        self.hidden_size = hidden_size
        self.variance_size_override = (
            None if var_hidden_size == hidden_size else var_hidden_size
        )
        if _use_aiter:
            self._forward_method = self.forward_aiter

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        post_residual_addition: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if x.numel() == 0:
            if residual is not None:
                if post_residual_addition is not None:
                    residual = residual + post_residual_addition
                return x, residual
            return x
        # sgl_kernel rmsnorm requires 2D input; reshape higher-rank tensors
        needs_reshape = x.dim() != 2 and residual is None
        if needs_reshape:
            original_shape = x.shape
            x = x.contiguous().reshape(-1, original_shape[-1])
        if self.variance_size_override is not None:
            return self.forward_native(x, residual, post_residual_addition)
        if is_batch_invariant_mode_enabled():
            if (
                residual is not None
                or self.cast_x_before_out_mul
                or get_global_server_args().rl_on_policy_target == "fsdp"
            ):
                return self.forward_native(x, residual, post_residual_addition)
            return rms_norm_batch_invariant(
                x,
                self.weight.data,
                self.variance_epsilon,
            )
        if self.cast_x_before_out_mul and residual is None:
            # Use HF-semantics kernel (cast to dtype before weight multiply).
            if (
                _jit_rmsnorm_hf_available
                and x.dtype in (torch.float16, torch.bfloat16)
                and self.weight.data.dtype == x.dtype
                and is_supported_rmsnorm_hf_hidden_size(x.shape[-1])
            ):
                out = _jit_rmsnorm_hf(
                    x.contiguous(), self.weight.data, self.variance_epsilon
                )
            else:
                # Fallback: pure-Python HF semantics (already implemented in forward_native).
                out = self.forward_native(x, None, None)
            if needs_reshape:
                out = out.reshape(original_shape)
            return out
        if residual is not None:
            # TODO: Ideally we want to have (hidden_states+residual)+post_residual_addition.
            # but right now we can only have hidden_states+(residual+post_residual_addition).
            # (hidden_states+residual)+post_residual_addition != hidden_states+(residual+post_residual_addition),
            # we probably need to add another parameter to fused_add_rmsnorm
            if post_residual_addition is not None:
                residual = residual + post_residual_addition
            fused_add_rmsnorm(x, residual, self.weight.data, self.variance_epsilon)
            return x, residual
        out = rmsnorm(x, self.weight.data, self.variance_epsilon)
        if needs_reshape:
            out = out.reshape(original_shape)
        return out

    def forward_npu(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        post_residual_addition: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if residual is not None:
            if post_residual_addition is not None:
                residual = residual + post_residual_addition
            out, _, residual_out = torch_npu.npu_add_rms_norm(
                residual, x, self.weight.data, self.variance_epsilon
            )
            return out, residual_out
        return torch_npu.npu_rms_norm(x, self.weight.data, self.variance_epsilon)[0]

    def forward_aiter(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        post_residual_addition: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if residual is not None:
            residual_out = torch.empty_like(x)
            output = torch.empty_like(x)
            if post_residual_addition is not None:
                residual = residual + post_residual_addition
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
        post_residual_addition: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Fallback to native implementation if vllm is not available
        if not _has_vllm_rms_norm:
            return self.forward_native(x, residual, post_residual_addition)

        if not x.is_contiguous():
            # NOTE: Remove this if aiter kernel supports discontinuous input
            x = x.contiguous()
        if residual is not None:
            out = torch.empty_like(x)
            residual_out = torch.empty_like(x)
            if post_residual_addition is not None:
                residual = residual + post_residual_addition
            fused_add_rms_norm(
                out, x, residual_out, residual, self.weight.data, self.variance_epsilon
            )
            return out, residual_out
        out = torch.empty_like(x)
        rms_norm(out, x, self.weight.data, self.variance_epsilon)
        return out

    def forward_musa(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        post_residual_addition: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if not get_global_server_args().disable_piecewise_cuda_graph:
            return self.forward_native(x, residual, post_residual_addition)

        if not x.is_contiguous():
            x = x.contiguous()

        if residual is not None:
            if post_residual_addition is not None:
                residual = residual + post_residual_addition
            fused_add_rmsnorm(x, residual, self.weight.data, self.variance_epsilon)
            return x, residual

        out = nn.functional.rms_norm(
            x, (self.hidden_size,), self.weight.data, self.variance_epsilon
        )
        return out

    def forward_native(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        post_residual_addition: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if not x.is_contiguous():
            x = x.contiguous()
        orig_dtype = self.override_orig_dtype or x.dtype
        x = x.to(torch.float32)
        if residual is not None:
            x = x + residual.to(torch.float32)
            if post_residual_addition is not None:
                x = x + post_residual_addition.to(torch.float32)
            if self.fp32_residual:
                residual = x.clone()
            else:
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

        if self.cast_x_before_out_mul:
            x = self.weight * x.to(orig_dtype)
        else:
            x = (x * self.weight).to(orig_dtype)

        if residual is None:
            return x
        else:
            return x, residual

    def forward_cpu(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        post_residual_addition: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if _is_cpu_amx_available:
            if residual is not None:
                if post_residual_addition is not None:
                    residual = residual + post_residual_addition
                torch.ops.sgl_kernel.fused_add_rmsnorm_cpu(
                    x, residual, self.weight.data, self.variance_epsilon
                )
                return x, residual
            return torch.ops.sgl_kernel.rmsnorm_cpu(
                x, self.weight.data, self.variance_epsilon
            )
        else:
            return self.forward_native(x, residual, post_residual_addition)

    def forward_xpu(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        post_residual_addition: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.variance_size_override is not None:
            return self.forward_native(x, residual, post_residual_addition)
        if residual is not None:
            if post_residual_addition is not None:
                residual = residual + post_residual_addition
            fused_add_rmsnorm(x, residual, self.weight.data, self.variance_epsilon)
            return x, residual
        out = rmsnorm(x, self.weight.data, self.variance_epsilon)
        return out

    def forward_with_allreduce_fusion(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        post_residual_addition: Optional[torch.Tensor] = None,
        use_attn_tp_group: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward with allreduce fusion, prioritizing flashinfer fused operations."""
        return _forward_with_allreduce_fusion(
            self, x, residual, post_residual_addition, self.weight, use_attn_tp_group
        )


class LayerNorm(MultiPlatformOp):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
        bias: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.variance_epsilon = eps
        self.elementwise_affine = elementwise_affine
        self.use_bias = bias
        self.dtype = dtype

        self.bias = nn.Parameter(torch.zeros(hidden_size, dtype=self.dtype))
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=self.dtype))

    def forward_cuda(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if (
            _flashinfer_layernorm_available
            and x.dtype == torch.bfloat16
            and self.dtype == torch.float32
        ):
            return layernorm(x, self.weight, self.bias, self.variance_epsilon)
        else:
            return self.forward_native(x)

    def forward_native(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        weight = self.weight if self.elementwise_affine else None
        bias = self.bias if self.use_bias else None
        orig_dtype = x.dtype
        x = x.to(self.dtype)
        return F.layer_norm(
            x,
            (self.hidden_size,),
            weight=weight,
            bias=bias,
            eps=self.variance_epsilon,
        ).to(orig_dtype)

    def forward_hip(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if (
            _has_aiter_layer_norm
            and x.dtype in (torch.bfloat16, torch.float16)
            and x.dtype == self.dtype
        ):
            orig_shape = x.shape
            x = x.reshape(-1, self.hidden_size)
            return layer_norm(x, self.weight, self.bias, self.variance_epsilon).view(
                orig_shape
            )
        else:
            return self.forward_native(x)

    def forward_npu(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_native(x)

    def forward_cpu(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if _is_cpu_amx_available:
            bias_data = self.bias.data if self.use_bias else None
            return torch.ops.sgl_kernel.layernorm_cpu(
                x, self.weight.data, bias_data, self.variance_epsilon
            )
        else:
            return self.forward_native(x)


class GemmaRMSNorm(MultiPlatformOp):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps
        self.register_buffer("gemma_weight", self.weight.data + 1.0, persistent=False)
        # (Chen-0210) Gemma weight = standard_weight + 1. Precompute once.
        # If TRTLLM allreduce fusion ever provides gemma-style norm
        # natively, this can be removed.
        self.weight.weight_loader = self._weight_loader

    def _weight_loader(self, param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        assert param.size() == loaded_weight.size()
        param.data.copy_(loaded_weight)
        self.gemma_weight = param.data + 1.0

    def _forward_impl(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        post_residual_addition: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        needs_reshape = x.dim() != 2 and residual is None
        if needs_reshape:
            original_shape = x.shape
            x = x.contiguous().reshape(-1, original_shape[-1])
        if residual is not None:
            if post_residual_addition is not None:
                residual = residual + post_residual_addition
            gemma_fused_add_rmsnorm(
                x, residual, self.weight.data, self.variance_epsilon
            )
            return x, residual
        out = gemma_rmsnorm(x, self.weight.data, self.variance_epsilon)
        if needs_reshape:
            out = out.reshape(original_shape)
        return out

    def forward_native(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        post_residual_addition: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        orig_dtype = x.dtype
        if residual is not None:
            if post_residual_addition is not None:
                residual = residual + post_residual_addition
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
        post_residual_addition: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        return self._forward_impl(x, residual, post_residual_addition)

    def forward_hip(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        post_residual_addition: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if not _has_vllm_rms_norm:
            return self.forward_native(x, residual, post_residual_addition)

        w = self.gemma_weight
        if _use_aiter:
            # aiter API: rms_norm(input, weight, eps) -> output
            #            fused_add_rms_norm(output, input, residual, residual_out, weight, eps)
            if residual is not None:
                output = torch.empty_like(x)
                residual_out = torch.empty_like(x)
                if post_residual_addition is not None:
                    residual = residual + post_residual_addition
                fused_add_rms_norm(
                    output, x, residual, residual_out, w, self.variance_epsilon
                )
                return output, residual_out
            return rms_norm(x, w, self.variance_epsilon)
        else:
            # vllm API: rms_norm(out, input, weight, eps) -> None (in-place)
            #           fused_add_rms_norm(out, input, residual_out, residual, weight, eps)
            if not x.is_contiguous():
                x = x.contiguous()
            if residual is not None:
                out = torch.empty_like(x)
                residual_out = torch.empty_like(x)
                if post_residual_addition is not None:
                    residual = residual + post_residual_addition
                fused_add_rms_norm(
                    out, x, residual_out, residual, w, self.variance_epsilon
                )
                return out, residual_out
            out = torch.empty_like(x)
            rms_norm(out, x, w, self.variance_epsilon)
            return out

    def forward_cpu(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        post_residual_addition: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if _is_cpu_amx_available:
            if residual is not None:
                if post_residual_addition is not None:
                    residual = residual + post_residual_addition
                torch.ops.sgl_kernel.gemma_fused_add_rmsnorm_cpu(
                    x, residual, self.weight.data, self.variance_epsilon
                )
                return x, residual
            return torch.ops.sgl_kernel.gemma_rmsnorm_cpu(
                x, self.weight.data, self.variance_epsilon
            )
        return self.forward_native(x, residual, post_residual_addition)

    def forward_npu(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        post_residual_addition: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if envs.SGLANG_NPU_FORWARD_NATIVE_GEMMA_RMS_NORM.get():
            return self.forward_native(x, residual)
        if residual is not None:
            if post_residual_addition is not None:
                residual = residual + post_residual_addition
            norm_out, residual = add_gemma_rms_norm(
                x, self.weight, residual, self.variance_epsilon
            )
            return norm_out, residual

        x, _ = torch_npu.npu_gemma_rms_norm(x, self.weight, self.variance_epsilon)
        return x

    def forward_xpu(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        post_residual_addition: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        return self._forward_impl(x, residual, post_residual_addition)

    def forward_with_allreduce_fusion(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        post_residual_addition: Optional[torch.Tensor] = None,
        use_attn_tp_group: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward with allreduce fusion; uses 1 + weight for fused kernels."""
        return _forward_with_allreduce_fusion(
            self,
            x,
            residual,
            post_residual_addition,
            self.gemma_weight,
            use_attn_tp_group=True,
        )


class Gemma3RMSNorm(MultiPlatformOp):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))
        # Re-dispatch

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward_native(self, x):
        output = self._norm(x.float())
        # Llama does x.to(float16) * w whilst Gemma3 is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

    def forward_cpu(self, x):
        if _is_cpu_amx_available and x.stride(-1) == 1:
            return torch.ops.sgl_kernel.gemma3_rmsnorm_cpu(x, self.weight, self.eps)
        return self.forward_native(x)

    def forward_cuda(self, x):
        return self.forward_native(x)

    def forward_npu(self, x):
        output, _ = torch_npu.npu_gemma_rms_norm(x, self.weight, self.eps)
        return output

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


class Gemma4RMSNorm(MultiPlatformOp):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        scale_shift: float = 0.0,
        with_scale: bool = True,
    ):
        super().__init__()
        self.with_scale = with_scale

        if self.with_scale:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_buffer("weight", torch.ones(dim), persistent=False)

        self.eps = eps
        self.scale_shift = scale_shift

    def __repr__(self):
        dim = self.weight.shape[0]
        return (
            f"{self.__class__.__name__}(dim={dim}, eps={self.eps}, "
            f"with_scale={self.with_scale}, scale_shift={self.scale_shift})"
        )

    def _norm(self, x):
        mean_squared = x.pow(2).mean(-1, keepdim=True) + self.eps
        return x * torch.pow(mean_squared, -0.5)

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        normed_output = self._norm(x.float())
        if self.with_scale:
            normed_output = normed_output * (self.weight.float() + self.scale_shift)
        return normed_output.type_as(x)

    def forward_cpu(self, x: torch.Tensor) -> torch.Tensor:
        if _is_cpu_amx_available:
            return torch.ops.sgl_kernel.gemma4_rmsnorm_cpu(
                x, self.weight.data, self.eps, self.scale_shift, self.with_scale
            )
        return self.forward_native(x)

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return x
        needs_reshape = x.dim() != 2
        if needs_reshape:
            original_shape = x.shape
            x = x.contiguous().reshape(-1, original_shape[-1])
        if self.with_scale and self.scale_shift == 1.0:
            # gemma_rmsnorm: norm(x) * (1 + weight)
            out = gemma_rmsnorm(x, self.weight.data, self.eps)
        else:
            # rmsnorm: norm(x) * weight
            # with_scale=False → weight is ones → norm(x) * 1 = norm(x)
            # scale_shift=0.0 → standard RMSNorm without +1 shift
            out = rmsnorm(x, self.weight.data, self.eps)

        if needs_reshape:
            out = out.reshape(original_shape)
        return out

    def forward_hip(self, x: torch.Tensor) -> torch.Tensor:
        # sgl_kernel's gemma_rmsnorm is not available on ROCm;
        # delegate to the pure-PyTorch implementation.
        return self.forward_native(x)


class RMSNormWithoutScale(MultiPlatformOp):
    def __init__(self, hidden_size: int, eps=1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward_native(self, x):
        orig_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x.to(orig_dtype)

    def forward_cuda(self, x):
        return self.forward_native(x)

    def extra_repr(self):
        return f"{self.hidden_size}, eps={self.eps}"
