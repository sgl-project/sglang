# Copyright 2026 SGLang Team
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
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from sglang.kernels.ops.gemm.bf16_fp32 import (
    linear_bf16_fp32,
    mark_hpc_bf16xfp32_gemm_enabled,
)
from sglang.srt.environ import envs
from sglang.srt.layers.amx_utils import PackWeightMethod
from sglang.srt.layers.utils.common import set_weight_attrs
from sglang.srt.runtime_context import get_context, get_exec
from sglang.srt.utils import (
    cpu_has_amx_support,
    get_bool_env_var,
    get_device_sm,
    is_cpu,
    is_cuda,
    is_gfx95_supported,
    is_hip,
    is_npu,
    use_intel_amx_backend,
)

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_cpu = is_cpu()
_is_npu = is_npu()
_is_cpu_amx_available = cpu_has_amx_support()
_device_sm = get_device_sm()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
_is_gfx95_supported = _is_hip and is_gfx95_supported()

if _is_cuda:
    from sglang.kernels.ops.gemm.tiny_gemm import can_use_tiny_gemm, tiny_gemm_bf16
else:
    can_use_tiny_gemm = None
    tiny_gemm_bf16 = None

if _use_aiter:
    from sglang.srt.layers.rocm_linear_utils import aiter_dsv3_router_gemm
else:
    aiter_dsv3_router_gemm = None

if _is_gfx95_supported and not _use_aiter:
    from sglang.kernels.ops.gemm.router_gemv import router_gemv, router_gemv_supported
else:
    router_gemv = None
    router_gemv_supported = None

# Minimum token count from which the HPC-Ops bf16xfp32 GEMM beats cuBLAS,
# measured on H200 per (hidden_size, num_experts) shape.
ROUTER_HPC_GEMM_MIN_M: dict[tuple[int, int], int] = {
    (4096, 256): 8,
    (6144, 768): 64,
    (3072, 384): 128,
}


def tiny_router_gemm_max_tokens(
    *, num_experts: int, hidden_size: int, weight_dtype: torch.dtype
) -> int:
    """Rows up to which the tiny GEMM beats cuBLAS for the router, or -1."""
    if not _is_cuda or _device_sm < 90 or weight_dtype != torch.bfloat16:
        return -1
    if can_use_tiny_gemm is None or not can_use_tiny_gemm(num_experts, hidden_size):
        return -1
    return 16


def _is_deterministic_inference() -> bool:
    return (
        get_context().is_config_namespace_published("exec")
        and get_exec().deterministic.enable_deterministic_inference
    )


class RouterGate(nn.Module):
    """Shared MoE router projection that produces FP32 router logits."""

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        *,
        fp32_compute: bool = True,
        params_dtype: torch.dtype | None = None,
        bias: bool = False,
        has_correction_bias: bool = False,
        correction_bias_name: str = "e_score_correction_bias",
        correction_bias_dtype: torch.dtype = torch.float32,
        correction_bias_shape: tuple[int, ...] | None = None,
        zero_init_correction_bias: bool = False,
        correction_bias_requires_grad: bool = True,
        hpc_kernel_min_m: int | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.prefix = prefix

        weight_dtype = (
            (torch.float32 if fp32_compute else torch.get_default_dtype())
            if params_dtype is None
            else params_dtype
        )
        self.params_dtype = weight_dtype
        self.fp32_compute = weight_dtype == torch.float32

        self.weight = nn.Parameter(
            torch.empty((num_experts, hidden_size), dtype=weight_dtype)
        )
        set_weight_attrs(
            self.weight,
            {
                "input_dim": 1,
                "output_dim": 0,
                "weight_loader": self.weight_loader,
            },
        )

        if bias:
            self.bias = nn.Parameter(torch.empty((num_experts,), dtype=weight_dtype))
            set_weight_attrs(
                self.bias,
                {
                    "output_dim": 0,
                    "weight_loader": self.weight_loader,
                },
            )
        else:
            self.register_parameter("bias", None)

        if has_correction_bias:
            shape = (
                correction_bias_shape
                if correction_bias_shape is not None
                else (num_experts,)
            )
            init_fn = torch.zeros if zero_init_correction_bias else torch.empty
            corr_param = nn.Parameter(
                init_fn(shape, dtype=correction_bias_dtype),
                requires_grad=correction_bias_requires_grad,
            )
            self.register_parameter(correction_bias_name, corr_param)
        else:
            self.register_parameter(correction_bias_name, None)

        self.quant_method = (
            PackWeightMethod(weight_names=["weight"])
            if _is_cpu and _is_cpu_amx_available and not self.fp32_compute
            else None
        )

        resolved_hpc_min_m = (
            ROUTER_HPC_GEMM_MIN_M.get((hidden_size, num_experts))
            if hpc_kernel_min_m is None
            else hpc_kernel_min_m
        )
        self.hpc_kernel_min_m = (
            resolved_hpc_min_m
            if (
                self.fp32_compute
                and self.bias is None
                and envs.SGLANG_OPT_BF16_FP32_GEMM_ALGO.get() == "hpc"
            )
            else None
        )
        if self.hpc_kernel_min_m is not None:
            mark_hpc_bf16xfp32_gemm_enabled()

        self.tiny_router_gemm_max_tokens = (
            tiny_router_gemm_max_tokens(
                num_experts=num_experts,
                hidden_size=hidden_size,
                weight_dtype=self.weight.dtype,
            )
            if self.bias is None
            else -1
        )

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)
        assert param.size() == loaded_weight.size(), (
            f"{param.shape=} {param.dtype=} {loaded_weight.shape=} {loaded_weight.dtype=}"
        )
        param.data.copy_(loaded_weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        x = (
            hidden_states.reshape(-1, orig_shape[-1])
            if hidden_states.dim() > 2
            else hidden_states
        )
        logits = self._forward_2d(x)
        if hidden_states.dim() > 2:
            return logits.view(*orig_shape[:-1], self.num_experts)
        return logits

    def _forward_2d(self, x: torch.Tensor) -> torch.Tensor:
        deterministic = _is_deterministic_inference()

        if self.weight.dtype == torch.float32:
            if self.hpc_kernel_min_m is not None and not deterministic:
                return linear_bf16_fp32(
                    x, self.weight, hpc_kernel_min_m=self.hpc_kernel_min_m
                )
            bias = (
                self.bias.float()
                if self.bias is not None and self.bias.dtype != torch.float32
                else self.bias
            )
            return F.linear(x.float(), self.weight, bias)

        if use_intel_amx_backend(self):
            return torch.ops.sgl_kernel.weight_packed_linear(
                x,
                self.weight,
                self.bias,
                True,  # is_vnni
            ).float()

        if deterministic:
            if (
                (_is_cuda or _is_hip)
                and x.is_cuda
                and x.dtype == torch.bfloat16
                and self.weight.dtype == torch.bfloat16
            ):
                logits = torch.mm(x, self.weight.t(), out_dtype=torch.float32)
                if self.bias is not None:
                    logits = logits + self.bias.float()
                return logits
            bias = self.bias.float() if self.bias is not None else None
            return F.linear(x.float(), self.weight.float(), bias)

        if (
            0 < x.shape[0] <= self.tiny_router_gemm_max_tokens
            and x.is_cuda
            and x.dtype == torch.bfloat16
        ):
            return tiny_gemm_bf16(x, self.weight, out_dtype=torch.float32)

        if _use_aiter and x.is_cuda:
            logits = aiter_dsv3_router_gemm(x, self.weight).float()
            if self.bias is not None:
                logits = logits + self.bias.float()
            return logits

        if (
            router_gemv is not None
            and self.bias is None
            and router_gemv_supported(x, self.weight)
        ):
            return router_gemv(x, self.weight)

        if _is_npu:
            return F.linear(x.to(self.weight.dtype), self.weight, self.bias).float()

        if (
            _is_cuda
            and x.is_cuda
            and x.dtype == torch.bfloat16
            and self.weight.dtype == torch.bfloat16
        ):
            logits = linear_bf16_fp32(x, self.weight)
            if self.bias is not None:
                logits = logits + self.bias.float()
            return logits

        bias = self.bias.float() if self.bias is not None else None
        return F.linear(x.float(), self.weight.float(), bias)
