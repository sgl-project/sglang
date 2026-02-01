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
"""
Fused Activation Layers - Migrated to Platform Abstraction
============================================================

This module demonstrates **RFC Feature #3: Platform-specific Op / Kernel Resolution**
applied to real model layer code.

MIGRATION GUIDE - Before vs After:
----------------------------------

BEFORE (old pattern - scattered platform checks):
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Module-level platform detection
    _is_cuda = is_cuda()
    _is_npu = is_npu()
    _is_hpu = is_hpu()
    _is_xpu = is_xpu()
    _is_hip = is_hip()
    _is_cpu_amx_available = ...

    # Conditional kernel imports
    if _is_cuda or _is_xpu:
        from sgl_kernel import silu_and_mul, gelu_and_mul, gelu_tanh_and_mul

    # Scattered if-else in each layer
    class SiluAndMul(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if _is_cuda:
                silu_and_mul(x, out)
            elif _is_npu:
                import torch_npu
                return torch_npu.npu_swiglu(x)
            elif _is_cpu_amx_available:
                return torch.ops.sgl_kernel.silu_and_mul_cpu(x)
            else:
                # native fallback
                return F.silu(x[..., :d]) * x[..., d:]

AFTER (new pattern - clean platform abstraction):
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Simple, type-safe imports
    from sglang.srt.ops import silu_and_mul, gelu_and_mul, gelu_tanh_and_mul
    from sglang.srt.platforms import current_platform

    # Use MultiPlatformOp base class for method dispatch
    class SiluAndMul(MultiPlatformOp):
        def forward_native(self, x):
            return F.silu(x[..., :d]) * x[..., d:]

        def forward_cuda(self, x):
            silu_and_mul(x, out)  # Platform-agnostic op!
            return out

        def forward_npu(self, x):
            return current_platform.modules.npu_swiglu(x)

        def forward_cpu(self, x):
            if current_platform.has_amx:
                return torch.ops.sgl_kernel.silu_and_mul_cpu(x)
            return self.forward_native(x)

BENEFITS:
---------
1. **No scattered if-else** - Platform dispatch is automatic via MultiPlatformOp
2. **Clean imports** - `from sglang.srt.ops import silu_and_mul` works everywhere
3. **Full IDE support** - Ops have docstrings, type hints, autocompletion
4. **Type safety** - Platform functions accessed via `current_platform.modules`
5. **Easy to add platforms** - Just add `forward_<platform>()` method
6. **Easy to add ops** - Register in Platform._init_ops(), define in ops/

KEY PATTERNS USED HERE:
-----------------------
1. `from sglang.srt.ops import ...` - Platform-agnostic op functions
2. `current_platform.modules.<func>` - Platform-specific functions (npu_swiglu, etc.)
3. `current_platform.has_amx` - Platform properties (CPU-specific check)
4. `MultiPlatformOp` base class - Automatic method dispatch by device type
"""

import logging
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig

from sglang.srt.distributed import (
    divide,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig

# MultiPlatformOp provides automatic forward_<device>() dispatch
from sglang.srt.layers.utils import MultiPlatformOp

# =============================================================================
# NEW PATTERN: Clean imports from sglang.srt.ops
# =============================================================================
# These ops automatically dispatch to the correct platform kernel:
# - CUDA → sgl_kernel.silu_and_mul
# - ROCm → sgl_kernel.silu_and_mul (HIP version)
# - NPU  → torch_npu.npu_swiglu (wrapped)
# - CPU  → native PyTorch fallback (or AMX kernel if available)
from sglang.srt.ops import gelu_and_mul, gelu_tanh_and_mul, silu_and_mul

# Access platform-specific properties and modules
from sglang.srt.platforms import current_platform
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import set_weight_attrs
from sglang.utils import resolve_obj_by_qualname

logger = logging.getLogger(__name__)


# =============================================================================
# EXAMPLE: SiluAndMul - MultiPlatformOp with Platform-Agnostic Ops
# =============================================================================
# This class demonstrates the new pattern:
# - Inherits from MultiPlatformOp for automatic device dispatch
# - Uses silu_and_mul from sglang.srt.ops (platform-agnostic)
# - Accesses platform-specific functions via current_platform.modules
# - Checks platform properties via current_platform.has_amx


class SiluAndMul(MultiPlatformOp):
    """Fused SiLU activation with element-wise multiply (SwiGLU).

    Used in LLaMA, Mistral, DeepSeek and other SwiGLU-based models.

    MultiPlatformOp automatically calls the right forward_<device>() method:
    - forward_cuda() on NVIDIA CUDA
    - forward_hip() on AMD ROCm
    - forward_npu() on Huawei Ascend
    - forward_cpu() on CPU
    - forward_native() as fallback
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if get_global_server_args().rl_on_policy_target is not None:
            self._forward_method = self.forward_native

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        """Pure PyTorch fallback - works on any device."""
        d = x.shape[-1] // 2
        return F.silu(x[..., :d]) * x[..., d:]

    def _forward_with_op(self, x: torch.Tensor) -> torch.Tensor:
        """Forward using the platform-dispatched op.

        Note: silu_and_mul() is imported from sglang.srt.ops.
        It automatically uses the correct kernel for the current platform.
        """
        d = x.shape[-1] // 2
        output_shape = x.shape[:-1] + (d,)
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        silu_and_mul(x, out)  # <-- Platform-agnostic op!
        return out

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        """CUDA: Use platform-dispatched silu_and_mul (sgl_kernel)."""
        return self._forward_with_op(x)

    def forward_hip(self, x: torch.Tensor) -> torch.Tensor:
        """ROCm: Use platform-dispatched silu_and_mul (sgl_kernel HIP)."""
        return self._forward_with_op(x)

    def forward_musa(self, x: torch.Tensor) -> torch.Tensor:
        """MUSA: Uses nn.SwishGLU which is optimized for Moore Threads."""
        return nn.SwishGLU()(x)

    def forward_cpu(self, x: torch.Tensor) -> torch.Tensor:
        """CPU: Check for AMX support via current_platform.has_amx."""
        if current_platform.has_amx:
            # Intel AMX kernel available
            return torch.ops.sgl_kernel.silu_and_mul_cpu(x)
        # No AMX - fall back to native PyTorch
        return self.forward_native(x)

    def forward_npu(self, x: torch.Tensor) -> torch.Tensor:
        """NPU: Access torch_npu.npu_swiglu via current_platform.modules."""
        # current_platform.modules provides type-safe access to vendor functions
        return current_platform.modules.npu_swiglu(x)

    def forward_xpu(self, x: torch.Tensor) -> torch.Tensor:
        """XPU: Use platform-dispatched silu_and_mul (sgl_kernel)."""
        return self._forward_with_op(x)


# =============================================================================
# EXAMPLE: GeluAndMul - Similar pattern for GELU activation
# =============================================================================


class GeluAndMul(MultiPlatformOp):
    """Fused GELU activation with element-wise multiply (GEGLU).

    Used in models with GEGLU feed-forward networks.
    Supports both exact GELU and tanh approximation.
    """

    def __init__(self, approximate="tanh"):
        super().__init__()
        self.approximate = approximate

    def _forward_with_op(self, x: torch.Tensor) -> torch.Tensor:
        """Forward using the platform-dispatched ops."""
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
        # CPU platform handles AMX detection internally
        if current_platform.has_amx:
            if self.approximate == "tanh":
                return torch.ops.sgl_kernel.gelu_tanh_and_mul_cpu(x)
            elif self.approximate == "none":
                return torch.ops.sgl_kernel.gelu_and_mul_cpu(x)
        return self.forward_native(x)

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_with_op(x)

    def forward_hip(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_with_op(x)

    def forward_xpu(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_with_op(x)

    def forward_npu(self, x: torch.Tensor) -> torch.Tensor:
        # Access torch_npu.npu_geglu through platform modules
        y_npu, _ = current_platform.modules.npu_geglu(
            x,
            dim=-1,
            approximate=1 if self.approximate == "tanh" else 0,
            activate_left=True,
        )
        return y_npu


class NewGELU(MultiPlatformOp):
    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        c = math.sqrt(2.0 / math.pi)
        return 0.5 * x * (1.0 + torch.tanh(c * (x + 0.044715 * torch.pow(x, 3.0))))

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Implement the CUDA kernel for NewGELU in sgl-kernel
        return self.forward_native(x)


class ReLU2(nn.Module):
    """
    Applies the squared Rectified Linear Unit function.
    y = max(0, x)^2
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(x)
        return x * x


class QuickGELU(MultiPlatformOp):
    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_native(x)

    def forward_hip(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        # Access gelu_quick through platform modules
        current_platform.modules.gelu_quick(x, out)
        return out

    def forward_npu(self, x: torch.Tensor) -> torch.Tensor:
        # Access torch_npu.npu_fast_gelu through platform modules
        return current_platform.modules.npu_fast_gelu(x)


class XIELU(MultiPlatformOp):
    """
    Applies the xIELU activation function introduced in https://arxiv.org/abs/2411.13010
    If the user has installed the nickjbrowning/XIELU, we import xIELU CUDA
    Otherwise, we emit a single warning and use xIELU Python
    """

    def __init__(
        self,
        alpha_p_init: float = 0.8,
        alpha_n_init: float = 0.8,
        beta: float = 0.5,
        eps: float = -1e-6,
        dtype: torch.dtype = torch.bfloat16,
        with_vector_loads: bool = False,
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
        self.with_vector_loads = with_vector_loads
        # Temporary until xIELU CUDA fully implemented
        self._beta_scalar = float(self.beta.detach().cpu().float().item())
        self._eps_scalar = float(self.eps.detach().cpu().float().item())

        self._xielu_cuda_obj = None
        try:
            import xielu.ops  # noqa: F401

            self._xielu_cuda_obj = torch.classes.xielu.XIELU()
            msg = "Using experimental xIELU CUDA."
            try:
                from torch._dynamo import allow_in_graph

                self._xielu_cuda_fn = allow_in_graph(self._xielu_cuda)
                msg += " Enabled torch._dynamo for xIELU CUDA."
            except Exception as err:
                msg += (
                    f" Could not enable torch._dynamo for xIELU ({err}) - "
                    "this may result in slower performance."
                )
                self._xielu_cuda_fn = self._xielu_cuda
            logger.warning_once(msg)
        except Exception as err:
            pass
            # logger.warning_once(
            #     "CUDA-fused xIELU not available (%s) –"
            #     " falling back to a Python version.\n"
            #     "For CUDA xIELU (experimental), `pip install git+https://github.com/nickjbrowning/XIELU`",
            #     str(err),
            # )

    def _xielu_python(self, x: torch.Tensor) -> torch.Tensor:
        alpha_p = nn.functional.softplus(self.alpha_p)
        alpha_n = self.beta + nn.functional.softplus(self.alpha_n)
        return torch.where(
            x > 0,
            alpha_p * x * x + self.beta * x,
            (torch.expm1(torch.min(x, self.eps)) - x) * alpha_n + self.beta * x,
        )

    def _xielu_cuda(self, x: torch.Tensor) -> torch.Tensor:
        """Firewall function to prevent torch.compile from seeing .item()"""
        assert self._xielu_cuda_obj is not None, "XIELU CUDA object must not be None"
        original_shape = x.shape
        # CUDA kernel expects 3D tensors, reshape if needed
        while x.dim() < 3:
            x = x.unsqueeze(0)
        if x.dim() > 3:
            x = x.view(-1, 1, x.size(-1))
        if original_shape != x.shape:
            logger.warning_once(
                "Warning: xIELU input tensor expects 3 dimensions"
                " but got (shape: %s). Reshaping to (shape: %s).\n"
                "Note: For SGLang this may be expected if sending"
                "[B*S,D] instead of [B,S,D].",
                original_shape,
                x.shape,
            )
        result = self._xielu_cuda_obj.forward(
            x,
            self.alpha_p,
            self.alpha_n,
            # Temporary until xIELU CUDA fully implemented -> self.{beta,eps}.item()
            self._beta_scalar,
            self._eps_scalar,
            self.with_vector_loads,
        )
        return result.view(original_shape)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self._xielu_cuda_obj is not None and input.is_cuda:
            if not torch._dynamo.is_compiling():
                return self._xielu_cuda_fn(input)
            else:
                logger.warning_once(
                    "torch._dynamo is compiling, using Python version of xIELU."
                )
        return self._xielu_python(input)


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
            tp_size = get_tensor_model_parallel_world_size()
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
            tp_rank = get_tensor_model_parallel_rank()
            shard_size = param_data.shape[0]
            start_idx = tp_rank * shard_size
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
                "intermediate_size must be specified for scaled "
                "activation functions."
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
