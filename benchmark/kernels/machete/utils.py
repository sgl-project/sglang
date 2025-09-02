import os
import json
import numpy
import torch
from typing import Dict, List, Tuple, Optional,TypedDict
from dataclasses import dataclass, fields
from sgl_kernel import ScalarType
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
    w_ref2: torch.Tensor = None
    w_q2: torch.Tensor = None
    w_g_s2: Optional[torch.Tensor] = None
    w_g_zp2: Optional[torch.Tensor] = None
    w_ch_s2: Optional[torch.Tensor] = None


@dataclass
class TypeConfig:
    act_type: torch.dtype
    weight_type: ScalarType
    output_type: Optional[torch.dtype]
    group_scale_type: Optional[torch.dtype]
    group_zero_type: Optional[torch.dtype]
    channel_scale_type: Optional[torch.dtype]
    token_scale_type: Optional[torch.dtype]

class BenchmarkConfig(TypedDict):
    BLOCK_SIZE_M: int
    BLOCK_SIZE_N: int
    CLUSTER_SIZE: int
    SCHE: str


def get_configs_compute_bound() -> List[Dict[str, int]]:
    configs: List[BenchmarkConfig] = []

    for block_m in [128, 256]:
        for block_n in [16, 32, 64, 128, 256]:
            for cluster in [1, 2]:
                for SCHE in ["streamK", "PersistentScheduler"]:
                    configs.append(
                        {"BLOCK_SIZE_M": block_m,
                        "BLOCK_SIZE_N": block_n,
                        "CLUSTER_SIZE": cluster,
                        "SCHE": SCHE
                        }
                    )
    return configs


def get_schedule_name(config):
    schedule = "{}x{}_{}x1x1_TmaMI__TmaCoop_{}".format(
        config["BLOCK_SIZE_M"], 
        config["BLOCK_SIZE_N"],
        config["CLUSTER_SIZE"],
        config["SCHE"])
    return schedule


def sort_config(config: BenchmarkConfig) -> BenchmarkConfig:
    return {
        "BLOCK_SIZE_M": config["BLOCK_SIZE_M"],
        "BLOCK_SIZE_N": config["BLOCK_SIZE_N"],
        "CLUSTER_SIZE": config["CLUSTER_SIZE"],
        "SCHE": config["SCHE"],
    }

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


def create_moe_data(num_experts: int,
                    shape: tuple[int, int, int],
                    types: TypeConfig,
                    group_size: Optional[int] = 128,
                    tp_size: int = 8,
                    subset_stride_factor: Optional[int] = None) -> Tensors:
    m, n, k = shape  # m, intermediate_size, hidden_size
    mm_group_cnt = num_experts
    
    ddtype = types.output_type
    w1 = rand_data((k, mm_group_cnt, 2*n//tp_size), ddtype, scale=3, offset=1).reshape([k, -1])
    w2 = rand_data([n//tp_size, mm_group_cnt, k], ddtype, scale=3, offset=1).reshape([n//tp_size, -1])

    w_ref, w_q_packed, w_s, w_zp = machete_quantize_and_pack(
        torch.float8_e4m3fn, w1, types.weight_type, types.group_scale_type, group_size,
        types.group_zero_type is not None)

    w_ref2, w_q_packed2, w_s2, w_zp2 = machete_quantize_and_pack(
        torch.float8_e4m3fn, w2, types.weight_type, types.group_scale_type, group_size,
        types.group_zero_type is not None)

    w_ch_s = None if types.channel_scale_type is None else\
        rand_data((n,), types.channel_scale_type)
    w_tok_s = None if types.token_scale_type is None else\
        rand_data((m,), types.token_scale_type)

    w_ch_s2 = None if w_ch_s is None else torch.cat([w_ch_s for i in range(mm_group_cnt)])

    w_q_packed = w_q_packed.reshape([num_experts, -1, 2*n//tp_size])
    w_q_packed2 = w_q_packed2.reshape([num_experts, -1, k])

    return Tensors(w_ref=None,
                   a_ref=None,
                   a=None,
                   w_q=w_q_packed.reshape([num_experts, -1, 2*n//tp_size]),
                   w_g_s=w_s,
                   w_g_zp=maybe_convert_zeropoints(w_zp, w_s),
                   w_ch_s=w_ch_s,
                   w_tok_s=w_tok_s,
                   mm_group_cnt=mm_group_cnt,
                   w_ref2=None,
                   w_q2=w_q_packed2.reshape([num_experts, -1, k]),
                   w_g_s2=w_s2,
                   w_g_zp2=maybe_convert_zeropoints(w_zp2, w_s2),
                   w_ch_s2=w_ch_s2)
