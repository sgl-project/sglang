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

from typing import Optional

import torch
from vllm import _custom_ops as ops

from sglang.srt.layers.quantization.base_config import QuantizationConfig


def dequantize_awq(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    quant_config: Optional[QuantizationConfig],
) -> torch.Tensor:
    return ops.awq_dequantize(
        qweight,
        scales,
        qzeros,
        0,
        0,
        0,
    ).T


"""
Adapted from https://github.com/AutoGPTQ/AutoGPTQ/blob/c0f8edb5812c467d7b1bfb90962f6418b40acb02/auto_gptq/nn_modules/qlinear/qlinear_cuda_old.py#L295

qweight - [K // 32 * bits , N             ], int32
scales -  [K // group_size, N             ], float16
qzeros  - [K // group_size, N // 32 * bits], int32
output  - [N,               K             ], float16
"""


def dequantize_gptq(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    quant_config: Optional[QuantizationConfig],
) -> torch.Tensor:
    bits = 32 * qzeros.size(1) // qweight.size(1)
    if bits not in [2, 4, 8]:
        raise NotImplementedError("Only 2,4,8 bits are supported.")
    group_size = 32 * qweight.size(0) // qzeros.size(0) // bits

    wf = (
        torch.tensor(list(range(0, 32, bits)), dtype=torch.int32)
        .unsqueeze(0)
        .to(qweight.device)
    )

    zeros = torch.bitwise_right_shift(
        torch.unsqueeze(qzeros, 2).expand(-1, -1, 32 // bits), wf.unsqueeze(0)
    ).to(torch.int8)
    zeros = torch.bitwise_and(zeros, (2**bits) - 1)
    zeros = zeros + 1

    zeros = zeros.reshape(-1, 1, zeros.shape[1] * zeros.shape[2])

    scales = scales.reshape(-1, 1, scales.shape[-1])

    weight = torch.bitwise_right_shift(
        torch.unsqueeze(qweight, 1).expand(-1, 32 // bits, -1), wf.unsqueeze(-1)
    ).to(torch.int8)
    weight = torch.bitwise_and(weight, (2**bits) - 1)
    weight = weight.reshape(-1, group_size, weight.shape[2])
    weight = scales * (weight - zeros)
    weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])
    weight = weight.T
    return weight


dequant_method_dict = {"awq": dequantize_awq, "gptq": dequantize_gptq}


def get_dequantize_method(quant_config: Optional[QuantizationConfig] = None):
    dequant_method = dequantize_awq
    if quant_config is not None:
        quant_name = quant_config.get_name()
        for name in dequant_method_dict:
            if name in quant_name:
                return dequant_method_dict[name]
    return dequant_method


def dequantize(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: Optional[torch.Tensor] = None,
    quant_config: Optional[QuantizationConfig] = None,
) -> torch.Tensor:
    dequant_method = get_dequantize_method(quant_config)
    return dequant_method(qweight, scales, qzeros, quant_config)
