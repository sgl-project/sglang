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
"""Fused operators for activation layers."""

import logging
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig

from sglang.kernels.fused_op import BaseFusedOp
from sglang.kernels.ops.activation.xielu import xielu_coeffs, xielu_reference
from sglang.srt.distributed import (
    divide,
)
from sglang.srt.environ import envs
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    Phase,
    check_cuda_graph_backend,
)
from sglang.srt.runtime_context import get_exec, get_parallel, publish_role
from sglang.srt.utils import (
    cpu_has_amx_support,
    get_bool_env_var,
    is_cpu,
    is_cuda,
    is_hip,
    is_musa,
    is_npu,
    is_xpu,
    set_weight_attrs,
)
from sglang.utils import resolve_obj_by_qualname

_is_cuda = is_cuda()
_is_musa = is_musa()
_is_npu = is_npu()
_is_cpu_amx_available = cpu_has_amx_support()
_is_cpu = is_cpu()
_is_hip = is_hip()
_is_xpu = is_xpu()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

if _is_cuda:
    from sgl_kernel import gelu_and_mul as _sgl_gelu_and_mul
    from sgl_kernel import gelu_tanh_and_mul as _sgl_gelu_tanh_and_mul
    from sgl_kernel import silu_and_mul as _sgl_silu_and_mul

    from sglang.kernels.ops.activation.activation import (
        gelu_and_mul as _jit_gelu_and_mul,
    )
    from sglang.kernels.ops.activation.activation import (
        gelu_tanh_and_mul as _jit_gelu_tanh_and_mul,
    )
    from sglang.kernels.ops.activation.activation import (
        relu2,
    )
    from sglang.kernels.ops.activation.activation import (
        silu_and_mul as _jit_silu_and_mul,
    )
    from sglang.kernels.ops.activation.xielu import xielu

    # The jit act-and-mul kernel requires the per-rank hidden size to be a
    # multiple of the vector width (kMaxVecBytes/dtype: 32B on SM100+, else 16B --
    # RuntimeCheck "hidden size must be divisible by vector size" in
    # kernels/jit/csrc/elementwise/activation.cuh). Route unaligned shapes to the
    # sgl_kernel implementation (e.g. DeepSeek-V2-Lite dense 10944/8 = 1368 at
    # tp8, 1368 % 16 != 0).
    _jit_act_max_vec_bytes: Optional[int] = None

    def _jit_act_supported(out: torch.Tensor) -> bool:
        global _jit_act_max_vec_bytes
        if _jit_act_max_vec_bytes is None:
            major, _ = torch.cuda.get_device_capability()
            _jit_act_max_vec_bytes = 32 if major >= 10 else 16
        return out.shape[-1] % (_jit_act_max_vec_bytes // out.dtype.itemsize) == 0

    def _act_and_mul(jit_fn, sgl_fn, input: torch.Tensor, out=None) -> torch.Tensor:
        if out is None:
            out = input.new_empty(*input.shape[:-1], input.shape[-1] // 2)
        (jit_fn if _jit_act_supported(out) else sgl_fn)(input, out)
        return out

    def silu_and_mul(input: torch.Tensor, out=None) -> torch.Tensor:
        return _act_and_mul(_jit_silu_and_mul, _sgl_silu_and_mul, input, out)

    def gelu_and_mul(input: torch.Tensor, out=None) -> torch.Tensor:
        return _act_and_mul(_jit_gelu_and_mul, _sgl_gelu_and_mul, input, out)

    def gelu_tanh_and_mul(input: torch.Tensor, out=None) -> torch.Tensor:
        return _act_and_mul(_jit_gelu_tanh_and_mul, _sgl_gelu_tanh_and_mul, input, out)

elif _is_xpu:
    from sgl_kernel import gelu_and_mul, gelu_tanh_and_mul, silu_and_mul
elif _is_hip:
    from sgl_kernel import gelu_and_mul, gelu_quick, gelu_tanh_and_mul, silu_and_mul
elif _is_musa:
    from sglang.srt.utils.patch_torch import register_fake_if_exists

    @register_fake_if_exists("aten::_fused_swiglu_forward")
    def _(x):
        d = x.shape[-1] // 2
        output_shape = x.shape[:-1] + (d,)
        return torch.empty(output_shape, dtype=x.dtype, device=x.device)


if _use_aiter:
    from aiter import silu_and_mul as _aiter_silu_and_mul

if is_npu():
    import torch_npu

logger = logging.getLogger(__name__)


class SiluAndMul(BaseFusedOp):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if (
            publish_role() is not None
            and get_exec().deterministic.rl_on_policy_target is not None
        ):
            self._forward_method = self.forward_native
        elif _use_aiter and envs.SGLANG_OPT_USE_AITER_SILU_MUL.get():
            self._forward_method = self.forward_aiter

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        return F.silu(x[..., :d]) * x[..., d:]

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        output_shape = x.shape[:-1] + (d,)
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        silu_and_mul(x, out)
        return out

    def forward_aiter(self, x: torch.Tensor, limit: float = 0.0) -> torch.Tensor:
        d = x.shape[-1] // 2
        output_shape = x.shape[:-1] + (d,)
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        _aiter_silu_and_mul(out, x, limit)
        return out

    def forward_cpu(self, x: torch.Tensor) -> torch.Tensor:
        if _is_cpu_amx_available:
            out = torch.ops.sgl_kernel.silu_and_mul_cpu(x)
            return out
        else:
            return self.forward_native(x)

    def forward_npu(self, x: torch.Tensor) -> torch.Tensor:
        out = torch_npu.npu_swiglu(x)
        return out

    def forward_xpu(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        output_shape = x.shape[:-1] + (d,)
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        silu_and_mul(x, out)
        return out

    def forward_musa(self, x: torch.Tensor) -> torch.Tensor:
        if check_cuda_graph_backend(Phase.PREFILL, Backend.TC_PIECEWISE):
            return self.forward_native(x)

        if not hasattr(self, "_musa_swish_glu"):
            # XXX (MUSA): nn.SwishGLU seems to have better performance than silu_and_mul on MUSA, we can switch to it for now. We can consider implementing a silu_and_mul kernel for MUSA in the future if needed.
            self._musa_swish_glu = nn.SwishGLU()
        return self._musa_swish_glu(x)


class SituAndMul(BaseFusedOp):
    """SituGLU activation used by Kimi K3.

    Computes beta * tanh(gate / beta) * sigmoid(gate) * up.
    When linear_beta is set, up is softly clipped:
        up = linear_beta * tanh(up / linear_beta).
    """

    def __init__(self, beta: float = 1.0, linear_beta: float | None = None):
        super().__init__()
        self.beta = float(beta)
        self.linear_beta = None if linear_beta is None else float(linear_beta)

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        gate = x[..., :d].float()
        up = x[..., d:].float()
        gate = self.beta * torch.tanh(gate / self.beta) * torch.sigmoid(gate)
        if self.linear_beta is not None:
            up = self.linear_beta * torch.tanh(up / self.linear_beta)
        return (gate * up).to(x.dtype)

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        from sglang.kernels.ops.activation import situ_and_mul

        return situ_and_mul(x, None, self.beta, self.linear_beta)

    def forward_npu(self, x: torch.Tensor) -> torch.Tensor:
        from sgl_kernel_npu.activation.situ import situ_and_mul

        return situ_and_mul(x)

    def forward_cpu(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_native(x)


class GeluAndMul(BaseFusedOp):
    def __init__(self, approximate="tanh"):
        super().__init__()
        self.approximate = approximate

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        output_shape = x.shape[:-1] + (d,)
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        if self.approximate == "tanh":
            gelu_tanh_and_mul(x, out)
        elif self.approximate == "none":
            gelu_and_mul(x, out)
        else:
            raise RuntimeError("GeluAndMul only support tanh or none")
        return out

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        return F.gelu(x[..., :d], approximate=self.approximate) * x[..., d:]

    def forward_cpu(self, x: torch.Tensor) -> torch.Tensor:
        if _is_cpu_amx_available and self.approximate == "tanh":
            return torch.ops.sgl_kernel.gelu_tanh_and_mul_cpu(x)
        elif _is_cpu_amx_available and self.approximate == "none":
            return torch.ops.sgl_kernel.gelu_and_mul_cpu(x)
        else:
            return self.forward_native(x)

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    def forward_xpu(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    def forward_npu(self, x: torch.Tensor) -> torch.Tensor:
        if envs.SGLANG_NPU_FORWARD_NATIVE_GELUTANH.get():
            return self.forward_native(x)
        y_npu, gelu_npu = torch_npu.npu_geglu(
            x,
            dim=-1,
            approximate=1 if self.approximate == "tanh" else 0,
            activate_left=True,
        )
        return y_npu


class NewGELU(BaseFusedOp):
    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        c = math.sqrt(2.0 / math.pi)
        return 0.5 * x * (1.0 + torch.tanh(c * (x + 0.044715 * torch.pow(x, 3.0))))

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Implement the CUDA kernel for NewGELU in sgl-kernel
        return self.forward_native(x)


class ReLU2(BaseFusedOp):
    """
    Applies the squared Rectified Linear Unit function.
    y = max(0, x)^2
    """

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(x)
        return x * x

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        return relu2(x)

    def forward_hip(self, x: torch.Tensor) -> torch.Tensor:
        # The JIT relu2 kernel is CUDA-only and not imported on ROCm.
        return self.forward_native(x)


class QuickGELU(BaseFusedOp):
    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_native(x)

    def forward_hip(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        gelu_quick(x, out)
        return out

    def forward_npu(self, x: torch.Tensor) -> torch.Tensor:
        return torch_npu.npu_fast_gelu(x)


class XIELU(BaseFusedOp):
    """xIELU activation (https://arxiv.org/abs/2411.13010), used by Apertus.

    ``alpha_p`` and ``alpha_n`` are stored in the checkpoint's pre-softplus
    form. The effective coefficients ``softplus(alpha_p)`` and
    ``beta + softplus(alpha_n)`` are constant at inference time, so they are
    derived once in fp32 into the non-persistent ``_coeffs`` buffer
    (:meth:`update_coefficients`, after loading weights), which the CUDA kernel
    reads on device. Both paths evaluate the expression in fp32 and round once
    to the input dtype.
    """

    def __init__(
        self,
        alpha_p_init: float = 0.8,
        alpha_n_init: float = 0.8,
        beta: float = 0.5,
        eps: float = -1e-6,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.alpha_p = nn.Parameter(
            torch.log(torch.exp(torch.tensor(alpha_p_init, dtype=dtype)) - 1).unsqueeze(
                0
            )
        )
        self.alpha_n = nn.Parameter(
            torch.log(
                torch.exp(torch.tensor(alpha_n_init - beta, dtype=dtype)) - 1
            ).unsqueeze(0)
        )
        self.register_buffer("beta", torch.tensor(beta, dtype=dtype))
        self.register_buffer("eps", torch.tensor(eps, dtype=dtype))
        self.register_buffer(
            "_coeffs", torch.empty(4, dtype=torch.float32), persistent=False
        )
        self.update_coefficients()

    @torch.no_grad()
    def update_coefficients(self) -> None:
        """Derive the effective coefficients in place; call after loading weights."""
        self._coeffs.copy_(
            xielu_coeffs(self.alpha_p, self.alpha_n, self.beta, self.eps)
        )

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        coeffs = xielu_coeffs(self.alpha_p, self.alpha_n, self.beta, self.eps)
        return xielu_reference(x, coeffs)

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype not in (torch.float16, torch.bfloat16) or not x.is_contiguous():
            return self.forward_native(x)
        return xielu(x, self._coeffs)

    def forward_hip(self, x: torch.Tensor) -> torch.Tensor:
        # The JIT xielu kernel is CUDA-only and not imported on ROCm.
        return self.forward_native(x)


class ScaledActivation(nn.Module):
    """An activation function with post-scale parameters.

    This is used for some quantization methods like AWQ.
    """

    def __init__(
        self,
        act_module: nn.Module,
        intermediate_size: int,
        input_is_parallel: bool = True,
        params_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.act = act_module
        self.input_is_parallel = input_is_parallel
        if input_is_parallel:
            self.tp_rank = get_parallel().tp_rank
            tp_size = get_parallel().tp_size
            intermediate_size_per_partition = divide(intermediate_size, tp_size)
        else:
            intermediate_size_per_partition = intermediate_size
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.scales = nn.Parameter(
            torch.empty(intermediate_size_per_partition, dtype=params_dtype)
        )
        set_weight_attrs(self.scales, {"weight_loader": self.weight_loader})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x) / self.scales

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        if self.input_is_parallel:
            shard_size = param_data.shape[0]
            start_idx = self.tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)


_ACTIVATION_REGISTRY = {
    "gelu": nn.GELU(),
    "gelu_pytorch_tanh": nn.GELU(approximate="tanh"),
    "gelu_new": NewGELU(),
    "relu2": ReLU2(),
    "xielu": XIELU(),
}


def get_act_fn(
    act_fn_name: str,
    quant_config: Optional[QuantizationConfig] = None,
    intermediate_size: Optional[int] = None,
    input_is_parallel: bool = True,
    params_dtype: Optional[torch.dtype] = None,
) -> nn.Module:
    """Get an activation function by name."""
    act_fn_name = act_fn_name.lower()
    if act_fn_name not in _ACTIVATION_REGISTRY:
        raise ValueError(f"Activation function {act_fn_name!r} is not supported.")

    act_fn = _ACTIVATION_REGISTRY[act_fn_name]
    if quant_config is not None and act_fn_name in quant_config.get_scaled_act_names():
        if intermediate_size is None:
            raise ValueError(
                "intermediate_size must be specified for scaled activation functions."
            )
        return ScaledActivation(
            act_fn, intermediate_size, input_is_parallel, params_dtype
        )
    return act_fn


def get_cross_encoder_activation_function(config: PretrainedConfig):
    if (
        hasattr(config, "sbert_ce_default_activation_function")
        and config.sbert_ce_default_activation_function is not None
    ):
        function_name = config.sbert_ce_default_activation_function
        assert function_name.startswith("torch.nn.modules."), (
            "Loading of activation functions is restricted to "
            "torch.nn.modules for security reasons"
        )
        return resolve_obj_by_qualname(function_name)()
    else:
        # adapt bge-reranker
        return nn.Identity()
