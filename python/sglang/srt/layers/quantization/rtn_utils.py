# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright © 2025, Oracle and/or its affiliates.

import logging

import torch
from torch.nn.parameter import Parameter

logger = logging.getLogger(__name__)


def rtn_quantize(
    tensor: torch.Tensor, num_bits: int, group_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a tensor using per-group static scaling factor.

    Args:
        tensor: The input tensor.
        num_bits: Target precision for the result (supported values are
                  8 or 4).
        group_size: Quantization granularity.
                    If equal to -1, each row in the input tensor is treated
                    as one group.
    """
    batch_present = len(tensor.shape) == 3
    if not batch_present:
        tensor = tensor.unsqueeze(0)

    q_range = 2**num_bits
    num_groups = (
        tensor.shape[1] * tensor.shape[2] // group_size
        if group_size != -1
        else tensor.shape[1]
    )
    # Calculate a scaling factor per input group.
    input_flat = tensor.reshape(tensor.shape[0], num_groups, -1)
    input_min = torch.min(input_flat, dim=2, keepdim=True)[0]
    input_max = torch.max(input_flat, dim=2, keepdim=True)[0]
    input_max_abs = torch.max(input_min.abs(), input_max.abs())
    scale = input_max_abs * 2.0 / (q_range - 1)
    # Scale each input group, truncate and round to the nearest integer.
    scaled_input = input_flat / scale
    scaled_input = scaled_input.round()

    scaled_input += q_range // 2
    scaled_input = scaled_input.clamp(0, q_range - 1)

    scale = scale.reshape(tensor.shape[0], tensor.shape[1], -1).contiguous()
    inputs_q = scaled_input.reshape(tensor.shape).to(torch.uint8)
    inputs_q = inputs_q.contiguous()

    if num_bits == 4:
        # Pack two 4-bit values into each byte.
        inputs_q = (inputs_q[:, :, 1::2] << 4) | (inputs_q[:, :, ::2] & 0xF)
        inputs_q = inputs_q.reshape(
            tensor.shape[0], tensor.shape[1] // 2, tensor.shape[2]
        )
        inputs_q = inputs_q.contiguous()

    if not batch_present:
        inputs_q = inputs_q.squeeze(0)
        scale = scale.squeeze(0)

    return inputs_q, scale


def rtn_dequantize(tensor: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize a tensor using per-group static scaling factors.

    Args:
        tensor: The input tensor.
        scale: The tensor with per-group scale factors.
    """
    batch_present = len(tensor.shape) == 3
    if not batch_present:
        tensor = tensor.unsqueeze(0)
        scale = scale.unsqueeze(0)

    num_groups = scale.size(1) * scale.size(2)
    batch, input_dim, output_dim = tensor.shape

    num_bits = 8 if input_dim == scale.size(1) else 4
    q_range = 2**num_bits
    if num_bits == 4:
        input_dim *= 2

    data = torch.empty(
        (batch, input_dim, output_dim), dtype=scale.dtype, device=tensor.device
    )

    if num_bits == 8:
        data.copy_(tensor)
        data -= q_range // 2
    else:
        # Unpack two 4-bit values from each byte.
        tensor = tensor.reshape(batch, input_dim, output_dim // 2)
        for i in range(2):
            data[:, :, i::2] = ((tensor << 4 * (1 - i)) >> 4).to(
                torch.int8
            ) - q_range // 2
    # Scale each input group with its scaling factor.
    scale = scale.reshape(batch, num_groups, -1)
    data = data.reshape(batch, num_groups, -1)
    data = torch.mul(data, scale)

    input_deq = data.reshape((batch, input_dim, output_dim)).contiguous()
    if not batch_present:
        input_deq = input_deq.squeeze(0)

    return input_deq


def fix_weights(layer: torch.nn.Module, param_name: str, reshape: bool = False):
    """torch.compile does not know how to deal with a Parameter subclass
    (aka RTNParameter). As we don't really need RTNParameters for the
    forward pass, we replace them with equivalent instances of Parameters.
    """
    old_weight = getattr(layer, param_name)
    data = old_weight.data.data

    delattr(layer, param_name)

    if reshape:
        data = data.reshape(old_weight.shape[0], old_weight.shape[1] * 2, -1)
    new_weight = Parameter(data=data, requires_grad=False)
    layer.register_parameter(param_name, new_weight)
