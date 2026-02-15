# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright © 2025, Oracle and/or its affiliates.

import logging

import numpy as np
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
    if scale.dtype.is_floating_point:
        scale = scale.clamp_min(torch.finfo(scale.dtype).tiny)
    # Scale each input group, round to the nearest integer, shift
    # the range and truncate.
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
    data = old_weight.data
    if hasattr(data, "as_subclass"):
        data = data.as_subclass(torch.Tensor)

    delattr(layer, param_name)

    if reshape:
        data = data.reshape(old_weight.shape[0], old_weight.shape[1] * 2, -1)
    new_weight = Parameter(data=data, requires_grad=False)
    layer.register_parameter(param_name, new_weight)


def _get_perms():
    perm = []
    for i in range(32):
        perm1 = []
        col = i // 4
        for block in [0, 1]:
            for row in [
                2 * (i % 4),
                2 * (i % 4) + 1,
                2 * (i % 4 + 4),
                2 * (i % 4 + 4) + 1,
            ]:
                perm1.append(16 * row + col + 8 * block)
        for j in range(4):
            perm.extend([p + 256 * j for p in perm1])

    perm_arr = np.array(perm)
    interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    perm_arr = perm_arr.reshape((-1, 8))[:, interleave].ravel()
    perm_tensor = torch.from_numpy(perm_arr)
    scale_perm = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return perm_tensor, scale_perm, scale_perm_single


_perm, _scale_perm, _scale_perm_single = _get_perms()


def pack_for_marlin(weight, scale, qbits):
    batch = weight.shape[0]

    n = weight.size(1)
    k = weight.size(2)
    groupsize = k // scale.size(2)

    tile = 16
    s = scale.permute(0, 2, 1)
    w = weight.permute(0, 2, 1)
    if groupsize != k:
        w = w.reshape((batch, -1, groupsize, n))
        w = w.permute(0, 2, 1, 3)
        w = w.reshape((batch, groupsize, -1))
        s = s.reshape((batch, 1, -1))

    if groupsize != k:
        w = w.reshape((batch, groupsize, -1, n))
        w = w.permute(0, 2, 1, 3)
        w = w.reshape((batch, k, n)).contiguous()
        s = s.reshape((batch, -1, len(_scale_perm)))[:, :, _scale_perm]
    else:
        s = s.reshape((batch, -1, len(_scale_perm_single)))[:, :, _scale_perm_single]
    s = s.reshape((batch, -1, n)).contiguous()
    w = w.reshape((batch, k // tile, tile, n // tile, tile))
    w = w.permute((0, 1, 3, 2, 4))
    w = w.reshape((batch, k // tile, n * tile))
    res = w
    res = res.reshape((batch, -1, _perm.numel()))[:, :, _perm].reshape(res.shape)
    if qbits == 4:
        q = torch.zeros(
            (batch, res.shape[1], res.shape[2] // 2), dtype=torch.int8, device=w.device
        )
        for i in range(2):
            q |= res[:, :, i::2] << 4 * i
        q = q.reshape(batch, -1, n).contiguous()
    else:
        q = res.clone()
        q[:, :, 2::8] = res[:, :, 4::8]
        q[:, :, 3::8] = res[:, :, 5::8]
        q[:, :, 4::8] = res[:, :, 2::8]
        q[:, :, 5::8] = res[:, :, 3::8]
        q = q.reshape(batch, -1, n).to(torch.int8).contiguous()

    return q, s


def repack_8bit_into_32bit(input):
    output = torch.zeros(
        (input.shape[0], input.shape[1], input.shape[2] // 4),
        dtype=torch.int32,
        device=input.device,
    )
    for i in range(4):
        output |= (input[:, :, i::4] & 0xFF).to(torch.int32) << 8 * i

    return output


def repack_weights(qweight, scale, weight_bits):
    batch_present = len(qweight.shape) == 3
    if not batch_present:
        qweight = qweight.unsqueeze(0)
        scale = scale.unsqueeze(0)

    if weight_bits == 4:
        qweight_unpacked = torch.empty(
            (qweight.shape[0], qweight.shape[1] * 2, qweight.shape[2]),
            dtype=torch.uint8,
            device=qweight.device,
        )
        for i in range(2):
            qweight_unpacked[:, :, i::2] = ((qweight << 4 * (1 - i)) >> 4).reshape(
                qweight.shape[0], qweight.shape[1] * 2, qweight.shape[2] // 2
            )
    else:
        qweight_unpacked = qweight

    qweight_packed, scale_packed = pack_for_marlin(qweight_unpacked, scale, weight_bits)
    qweight_repacked = repack_8bit_into_32bit(qweight_packed.to(torch.uint8))
    qweight_reshaped = qweight_repacked.reshape(
        qweight.shape[0], qweight.shape[2] // 16, -1
    )
    if not batch_present:
        qweight_reshaped = qweight_reshaped.squeeze(0)
        scale_packed = scale_packed.squeeze(0)

    return qweight_reshaped, scale_packed
