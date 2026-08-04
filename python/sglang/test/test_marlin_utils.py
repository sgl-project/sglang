"""
Adapted from
https://github.com/vllm-project/vllm/blob/020f58abcdea65302225663130d08fd8f4dd755a/vllm/model_executor/layers/quantization/utils/marlin_utils_test.py
"""

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utility functions used for tests and benchmarks"""

from typing import Optional

import numpy as np
import torch
from sgl_kernel.scalar_type import ScalarType

from sglang.srt.layers.quantization.marlin_utils import (
    GPTQ_MARLIN_TILE,
    marlin_permute_scales,
    marlin_zero_points,
)
from sglang.srt.layers.quantization.utils import (
    get_pack_factor,
    gptq_quantize_weights,
    quantize_weights,
    sort_weights,
)


class MarlinWorkspace:

    def __init__(self, out_features, min_thread_n, max_parallel):
        assert (
            out_features % min_thread_n == 0
        ), "out_features = {} is undivisible by min_thread_n = {}".format(
            out_features, min_thread_n
        )

        max_workspace_size = (out_features // min_thread_n) * max_parallel

        self.scratch = torch.zeros(max_workspace_size, dtype=torch.int, device="cuda")


def marlin_permute_weights(q_w, size_k, size_n, perm, tile=GPTQ_MARLIN_TILE):
    assert q_w.shape == (size_k, size_n)
    assert size_k % tile == 0, f"size_k = {size_k}, tile = {tile}"
    assert size_n % tile == 0, f"size_k = {size_n}, tile = {tile}"

    # Permute weights to 16x64 marlin tiles
    q_w = q_w.reshape((size_k // tile, tile, size_n // tile, tile))
    q_w = q_w.permute((0, 2, 1, 3))
    q_w = q_w.reshape((size_k // tile, size_n * tile))

    q_w = q_w.reshape((-1, perm.numel()))[:, perm].reshape(q_w.shape)

    return q_w


def marlin_weights(q_w, size_k, size_n, num_bits, perm):
    # Permute
    q_w = marlin_permute_weights(q_w, size_k, size_n, perm)

    # Pack
    pack_factor = get_pack_factor(num_bits)
    orig_device = q_w.device

    q_w = q_w.cpu().numpy().astype(np.uint32)

    q_packed = np.zeros((q_w.shape[0], q_w.shape[1] // pack_factor), dtype=np.uint32)
    for i in range(pack_factor):
        q_packed |= q_w[:, i::pack_factor] << num_bits * i

    q_packed = torch.from_numpy(q_packed.astype(np.int32)).to(orig_device)

    return q_packed


def get_weight_perm(num_bits: int):
    perm_list: list[int] = []
    for i in range(32):
        perm1: list[int] = []
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
            perm_list.extend([p + 256 * j for p in perm1])

    perm = np.array(perm_list)

    if num_bits == 4:
        interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    elif num_bits == 8:
        interleave = np.array([0, 2, 1, 3])
    else:
        raise Exception("num_bits must be 4 or 8, got {}".format(num_bits))

    perm = perm.reshape((-1, len(interleave)))[:, interleave].ravel()
    perm = torch.from_numpy(perm)
    return perm


def marlin_quantize(
    w: torch.Tensor,
    quant_type: ScalarType,
    group_size: int,
    act_order: bool,
    test_perm: Optional[torch.Tensor] = None,
):
    size_k, size_n = w.shape
    num_bits = quant_type.size_bits

    # Normalize group_size
    if group_size == -1:
        group_size = size_k
    assert group_size <= size_k

    # Quantize (and apply act_order if provided)
    w_ref, q_w, s, g_idx, rand_perm = gptq_quantize_weights(
        w, quant_type, group_size, act_order, test_perm
    )

    # For act_order, sort the "weights" and "g_idx" so that group ids are
    # increasing
    sort_indices = torch.empty(0, dtype=torch.int, device=w.device)
    if act_order:
        q_w, g_idx, sort_indices = sort_weights(q_w, g_idx)

    # Reformat to marlin
    weight_perm = get_weight_perm(num_bits)
    marlin_q_w = marlin_weights(q_w, size_k, size_n, num_bits, weight_perm)
    marlin_s = marlin_permute_scales(s, size_k, size_n, group_size)

    # Create result
    res_list = [w_ref, marlin_q_w, marlin_s, g_idx, sort_indices, rand_perm]
    for i in range(len(res_list)):
        res_list[i] = res_list[i].to(w.device)

    return res_list


def awq_marlin_quantize(w: torch.Tensor, quant_type: ScalarType, group_size: int):
    size_k, size_n = w.shape

    # Normalize group_size
    if group_size == -1:
        group_size = size_k
    assert group_size <= size_k

    # Detect num groups
    assert size_k % group_size == 0
    num_groups = size_k // group_size

    # Quantize with zp
    w_ref, q_w, s, zp = quantize_weights(w, quant_type, group_size, zero_points=True)

    # Reformat to marlin
    weight_perm = get_weight_perm(quant_type.size_bits)
    marlin_q_w = marlin_weights(q_w, size_k, size_n, quant_type.size_bits, weight_perm)
    marlin_s = marlin_permute_scales(s, size_k, size_n, group_size)
    marlin_zp = marlin_zero_points(zp, num_groups, size_n, quant_type.size_bits)

    # Create result
    res_list = [w_ref, marlin_q_w, marlin_s, marlin_zp]
    for i in range(len(res_list)):
        res_list[i] = res_list[i].to(w.device)

    return res_list


def make_nvfp4_weight_and_ref(
    size_n: int,
    size_k: int,
    dtype: torch.dtype,
    group_size: int = 16,
    device: str = "cuda",
):
    """Build a random NVFP4-quantized weight and its FP dequantized reference.

    Returns:
        fp4_weight: (size_n, size_k // 2) uint8, two packed FP4 (E2M1) values per byte
        scales: (size_n, size_k // group_size) FP8 E4M3 per-group scales
        global_scale: scalar in `dtype`, the FP16/BF16 outer scale
        weight_ref: (size_n, size_k) tensor in `dtype` = dequantized weight
    """
    fp4_weight = torch.randint(
        0, 256, (size_n, size_k // 2), dtype=torch.uint8, device=device
    )
    scale_source = torch.randn((size_n, size_k), dtype=dtype, device=device)
    # /6 = FP4 (E2M1) max; /448 = FP8 (E4M3) max — sets each level to its dtype's full range.
    scales = scale_source.view(size_n, -1, group_size).abs().max(-1)[0] / 6
    global_scale = scales.max() / 448
    scales = (scales / global_scale).to(torch.float8_e4m3fn)

    def _unpack(byte_view: torch.Tensor) -> torch.Tensor:
        # Convert 4-bit E2M1 nibble (in upper bits of a uint8) to FP8 E4M3 bit pattern.
        unpacked = (byte_view & 0b10000000) | ((byte_view & 0b01110000) >> 2)
        return unpacked.view(torch.float8_e4m3fn).to(dtype) * (2**6)

    part_low = _unpack(fp4_weight)
    part_high = _unpack(fp4_weight << 4)

    weight_ref = torch.cat([part_high.unsqueeze(2), part_low.unsqueeze(2)], 2).view(
        size_n, size_k
    )
    weight_ref = (
        weight_ref
        * global_scale.to(dtype)
        * scales.repeat_interleave(group_size, 1).to(dtype)
    )

    return fp4_weight, scales, global_scale, weight_ref
