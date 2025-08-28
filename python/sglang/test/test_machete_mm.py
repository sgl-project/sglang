# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the machete kernel.

Run `pytest tests/test_machete_mm.py`.
"""

import pytest
import torch
import math
from dataclasses import dataclass, fields
from typing import Optional

from sgl_kernel import ScalarType, scalar_types, machete_supported_schedules, machete_mm
from sglang.srt.layers.quantization.utils import machete_quantize_and_pack

@dataclass
class Tensors:
    w_ref: torch.Tensor
    a_ref: torch.Tensor
    a: torch.Tensor
    w_q: torch.Tensor
    w_g_s: Optional[torch.Tensor]
    w_g_zp: Optional[torch.Tensor]
    w_ch_s: Optional[torch.Tensor]
    w_tok_s: Optional[torch.Tensor]
    mm_group_cnt: int = 1


@dataclass
class TypeConfig:
    act_type: torch.dtype
    weight_type: ScalarType
    output_type: Optional[torch.dtype]
    group_scale_type: Optional[torch.dtype]
    group_zero_type: Optional[torch.dtype]
    channel_scale_type: Optional[torch.dtype]
    token_scale_type: Optional[torch.dtype]


MNK_SHAPES = [
    (1, 128, 128),
    (1, 7168, 2048),
    (64, 4096, 4096),
    (64, 8192, 28672),
    (1024, 4096, 8192),
    (1024, 8192, 4096),
]

GROUP_SIZES_TO_TEST: list[Optional[int]] = [128]

TEST_TYPES = [
    *(TypeConfig(act_type=torch.float8_e4m3fn,
                 weight_type=scalar_types.uint4b8,
                 output_type=a_type,
                 group_scale_type=a_type,
                 group_zero_type=None,
                 channel_scale_type=s_type,
                 token_scale_type=s_type)
      for s_type in [None, torch.float]
      for a_type in [torch.float16, torch.bfloat16])
]


def group_size_valid(shape: tuple[int, int, int],
                     group_size: Optional[int]) -> bool:
    return group_size is None or group_size == -1 or shape[2] % group_size == 0

def rand_data(shape, dtype=torch.float16, scale=1, offset=0):
    if dtype.is_floating_point:
        return (scale * torch.rand(shape, device="cuda") - offset).to(dtype)
    else:
        return torch.randint(-8, 7, shape, dtype=dtype, device="cuda")

def maybe_convert_zeropoints(zps: Optional[torch.Tensor], s: torch.Tensor):
    return zps if zps is None else -1 * s * (zps.to(s.dtype))

def create_gemm_data(shape: tuple[int, int, int],
                        types: TypeConfig,
                        group_size: Optional[int],
                        mm_group_cnt: int = 1,
                        subset_stride_factor: Optional[int] = None) -> Tensors:
    m, n, k = shape
    factor = subset_stride_factor or 1

    print("create_data, shape:", shape, "types:", types, "group_size:", group_size)

    a = rand_data((m * factor, k * factor), types.act_type, scale=3, offset=2)
    w = rand_data((k * factor, n * factor), types.act_type, scale=3, offset=1)

    if factor > 1:
        a = a[0:m, 0:k]
        w = w[0:k, 0:n]

    if types.group_scale_type is not None:
        w = w.to(types.group_scale_type)
    if w.dtype.itemsize == 1:
        w = w.to(torch.float16)

    w_ref, w_q_packed, w_s, w_zp = machete_quantize_and_pack(
        a.dtype, w, types.weight_type, types.group_scale_type, group_size,
        types.group_zero_type is not None)
        
    if mm_group_cnt > 1:
        # 创建 grouped weight
        w_tmp = torch.cat([w.unsqueeze(1) for i in range(mm_group_cnt)], dim=1).contiguous()
        w = w_tmp.reshape([w.shape[0], -1])

        _, w_q_packed, w_s, w_zp = machete_quantize_and_pack(
            a.dtype, w, types.weight_type, types.group_scale_type, group_size,
            types.group_zero_type is not None)


    if not a.dtype.is_floating_point:
        aiinfo = torch.iinfo(a.dtype)
        w_ref = w_ref.round().clamp(aiinfo.min, aiinfo.max)

    a_ref = a.to(torch.float32)
    w_ref = w_ref.to(torch.float32)

    w_ch_s = None if types.channel_scale_type is None else rand_data((n,), types.channel_scale_type)
    w_tok_s = None if types.token_scale_type is None else rand_data((m,), types.token_scale_type)

    if mm_group_cnt > 1:
        w_ch_s = w_ch_s.repeat(mm_group_cnt) if w_ch_s is not None else None

    return Tensors(w_ref=w_ref,
                a_ref=a_ref,
                a=a,
                w_q=w_q_packed,
                w_g_s=w_s,
                w_g_zp=maybe_convert_zeropoints(w_zp, w_s),
                w_ch_s=w_ch_s,
                w_tok_s=w_tok_s,
                mm_group_cnt=mm_group_cnt)

def machete_mm_test_helper(types: TypeConfig,
                           tensors: Tensors,
                           group_size: Optional[int] = None,
                           schedule: Optional[str] = None):
    output_ref = torch.matmul(tensors.a_ref, tensors.w_ref)
    output_ref_type = output_ref.dtype
    if tensors.w_ch_s is not None:
        w_ch_s = tensors.w_ch_s
        if tensors.mm_group_cnt > 1:
            w_ch_s = w_ch_s.reshape([tensors.mm_group_cnt, -1])[0]
        output_ref = (output_ref.to(w_ch_s.dtype) * w_ch_s.unsqueeze(0)).to(output_ref_type)
    if tensors.w_tok_s is not None:
        output_ref = (output_ref.to(tensors.w_tok_s.dtype) *
                      tensors.w_tok_s.unsqueeze(1)).to(output_ref_type)

    if schedule is None:
        schedule = "256x16_2x1x1_TmaMI__TmaCoop_PersistentScheduler"

    if tensors.mm_group_cnt > 1:
        # group gemm
        group_layout = torch.zeros([tensors.a.shape[0]], dtype=torch.int32).to(tensors.a.device)
        BLOCK_SIZE_M = int(schedule.split("_")[0].split("x")[1])
        valid_len = torch.tensor([BLOCK_SIZE_M * group_layout.shape[0]], dtype=torch.int32, device=tensors.a.device)
        output = machete_mm(
                a=tensors.a,
                b_q=tensors.w_q.reshape([tensors.mm_group_cnt, tensors.w_q.shape[0], -1]), # [G,K,N]
                b_type=types.weight_type,
                b_group_scales=tensors.w_g_s,
                b_group_zeros=tensors.w_g_zp,
                b_group_size=group_size,
                b_channel_scales=tensors.w_ch_s,
                a_token_scales=tensors.w_tok_s,
                out_type=types.output_type,
                schedule=schedule,
                group_layout=group_layout,
                group_stride=1,
                valid_len=valid_len
            )
    else:
        # normal single gemm
        output = machete_mm(
            a=tensors.a,
            b_q=tensors.w_q,
            b_type=types.weight_type,
            b_group_scales=tensors.w_g_s,
            b_group_zeros=tensors.w_g_zp,
            b_group_size=group_size,
            b_channel_scales=tensors.w_ch_s,
            a_token_scales=tensors.w_tok_s,
            out_type=types.output_type,
            schedule=schedule,
            group_layout=None,
            group_stride=1,
        )

    # Relax atol as our reduction dim becomes larger (more rounding error)
    # Relax atol when we have zeropoints since the way machete applies
    #  zeropoints (after scales) causes noise around 0
    atol = 1 if tensors.w_g_zp is not None else min(5e-2 * math.sqrt(tensors.a.shape[1]), 1)
    rtol = 1e-1 if tensors.a.element_size() >= 2 else 2e-1
    torch.testing.assert_close(output, output_ref.to(output.dtype), rtol=rtol, atol=atol)


@pytest.mark.parametrize("shape", MNK_SHAPES, ids=lambda x: "x".join(str(v) for v in x))
@pytest.mark.parametrize("types", TEST_TYPES)
@pytest.mark.parametrize("mm_group_cnt", [1, 4])
def test_machete_all_schedules(shape, types: TypeConfig, mm_group_cnt):

    group_sizes: list[Optional[int]] = []
    if types.group_scale_type is None:
        group_sizes = [None]
    else:
        group_sizes = GROUP_SIZES_TO_TEST

    for group_size in group_sizes:
        if not group_size_valid(shape, group_size):
            continue

        tensors = create_gemm_data(shape, types, group_size, mm_group_cnt)

        for schedule in machete_supported_schedules(
                types.act_type,
                types.weight_type,
                group_scales_type=types.group_scale_type,
                group_zeros_type=types.group_zero_type,
                out_type=types.output_type):

            print(f"="*100)
            print(f"MNK = {shape}")
            print(f"Testing schedule {schedule}")
            machete_mm_test_helper(types, tensors, group_size, schedule)
