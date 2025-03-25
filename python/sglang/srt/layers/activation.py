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

from sglang.srt.utils import is_cuda_available

_is_cuda = is_cuda_available()

if _is_cuda:
    from sgl_kernel import gelu_and_mul, gelu_tanh_and_mul, silu_and_mul

import triton
import triton.language as tl

from sglang.srt.custom_op import CustomOp
from sglang.srt.distributed import (
    divide,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.utils import libentry, set_weight_attrs

logger = logging.getLogger(__name__)


@triton.jit
def silu_and_mul_kernel(out_ptr, x_ptr, d: tl.constexpr, d_padded: tl.constexpr):
    token_idx = tl.program_id(0)
    idx = tl.arange(0, d_padded)

    # Load A and B from x
    mask = idx < d  # Mask for values within the original range of d
    x_a = tl.load(x_ptr + token_idx * 2 * d + idx, mask=mask, other=0.0)
    x_b = tl.load(x_ptr + token_idx * 2 * d + d + idx, mask=mask, other=0.0)

    # SiLU activation on A and element-wise multiplication with B
    x_a_fp32 = x_a.to(tl.float32)
    x_silu = x_a_fp32 / (1.0 + tl.exp(-x_a_fp32))
    result = x_silu * x_b

    # Write result to output
    tl.store(out_ptr + token_idx * d + idx, result, mask=mask)


class _SiluAndMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, out, x):
        d = x.shape[-1] // 2
        d_padded = 1 << (d - 1).bit_length()  # Calculate d_padded in Python
        num_tokens = x.shape[0]

        # Launch the Triton kernel
        silu_and_mul_kernel[(num_tokens,)](out, x, d=d, d_padded=d_padded)


def _silu_and_mul(out, x):
    _SiluAndMul.apply(out, x)


class SiluAndMul(CustomOp):
    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        return F.silu(x[..., :d]) * x[..., d:]

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        output_shape = x.shape[:-1] + (d,)
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        silu_and_mul(x, out)
        return out

    def forward_triton(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        output_shape = x.shape[:-1] + (d,)
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        _silu_and_mul(out, x)
        return out


class GeluAndMul(CustomOp):
    def __init__(self, approximate="tanh"):
        super().__init__()
        self.approximate = approximate

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        return F.gelu(x[..., :d], approximate=self.approximate) * x[..., d:]

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
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


class NewGELU(CustomOp):
    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        c = math.sqrt(2.0 / math.pi)
        return 0.5 * x * (1.0 + torch.tanh(c * (x + 0.044715 * torch.pow(x, 3.0))))

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Implement the CUDA kernel for NewGELU in sgl-kernel
        return self.forward_native(x)


class QuickGELU(CustomOp):
    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        # TODO(zhyncs): Implement the CUDA kernel for QuickGELU in sgl-kernel
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


if not _is_cuda:
    logger.info(
        "sgl-kernel is not available on Non-NV platforms. Fallback to other kernel libraries."
    )
    from vllm.model_executor.layers.activation import GeluAndMul, SiluAndMul
