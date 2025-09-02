import pytest
import torch
from typing import Optional
from dataclasses import dataclass, fields

from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_experts_machete_impl
from sglang.srt.layers.quantization.utils import quantize_weights
from sgl_kernel import machete_prepack_B, ScalarType, scalar_types

from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.moe.topk import select_experts

NUM_EXPERTS = [8, 64]
TOP_KS = [2, 6]

@dataclass
class MacheteTypes:
    act_type: torch.dtype
    weight_type: ScalarType
    output_type: Optional[torch.dtype]
    group_scale_type: Optional[torch.dtype]
    group_zero_type: Optional[torch.dtype]
    channel_scale_type: Optional[torch.dtype]
    token_scale_type: Optional[torch.dtype]

def process_qweight(qweight, types):
    G, out_dim, in_dim = qweight.shape
    qweight = qweight.reshape([-1, in_dim]).view(torch.int32).T
    qweight_processed = machete_prepack_B(qweight, types.act_type, types.weight_type, types.group_scale_type)
    return qweight_processed.reshape([G, -1, out_dim])

def process_weights_after_loading(qweight, scales, maybe_zeros, types):
    qweight = process_qweight(qweight, types)
    scales = scales.reshape([-1, scales.shape[-1]]).permute([1,0]).contiguous().to(types.group_scale_type)
    if maybe_zeros is not None:
        pass
    return qweight, scales, maybe_zeros

def torch_moe(a, w1, w2, score, topk):
    B, D = a.shape
    a = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
    out = torch.zeros(B * topk, w2.shape[1], dtype=a.dtype, device=a.device)
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk)
    topk_weight = topk_weight.view(-1)
    topk_ids = topk_ids.view(-1)
    for i in range(w1.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            out[mask] = SiluAndMul()(a[mask] @ w1[i].transpose(0, 1)) @ w2[i].transpose(
                0, 1
            )
    return (
        out.view(B, -1, w2.shape[1]) * topk_weight.view(B, -1, 1).to(out.dtype)
    ).sum(dim=1)


# modify from https://github.com/vllm-project/vllm/blob/main/tests/kernels/test_moe.py
@pytest.mark.parametrize("m", [1, 32, 222])
@pytest.mark.parametrize("n", [256])
@pytest.mark.parametrize("k", [128])
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("group_size", [128])
@pytest.mark.parametrize("has_zp", [False])
@pytest.mark.parametrize("weight_bits", [4])
def test_fused_moe(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    dtype: torch.dtype,
    group_size: int,
    has_zp: bool,
    weight_bits: int,
):
    with torch.cuda.device("cuda:0"):
        types = MacheteTypes(act_type=torch.float8_e4m3fn,
                    weight_type=scalar_types.uint4b8,
                    output_type=dtype,
                    group_scale_type=dtype,
                    group_zero_type=None if has_zp else dtype,
                    channel_scale_type=None,
                    token_scale_type=None)

        print(m, n, k, e, topk, dtype, group_size, has_zp, weight_bits)
        a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
        w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 10
        w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 10
        score = torch.randn((m, e), device="cuda", dtype=dtype)

        if weight_bits == 4:
            pack_factor = 2
            quant_type = scalar_types.uint4b8

        w1_ref = w1.clone()
        w2_ref = w2.clone()
        w1_qweight = torch.empty(
            (e, 2 * n, k // pack_factor), device="cuda", dtype=torch.uint8
        )
        w2_qweight = torch.empty((e, k, n // pack_factor), device="cuda", dtype=torch.uint8)
        w1_scales = torch.empty((e, 2 * n, k // group_size), device="cuda", dtype=dtype)
        w2_scales = torch.empty((e, k, n // group_size), device="cuda", dtype=dtype)
        w1_qzeros = torch.empty(
            (e, 2 * n // pack_factor, k // group_size), device="cuda", dtype=torch.uint8
        )
        w2_qzeros = torch.empty(
            (e, k // pack_factor, n // group_size), device="cuda", dtype=torch.uint8
        )

        for i in range(e * 2):
            expert_id = i % e
            if i // e == 0:
                w, w_ref, w_qweight, w_scales, w_qzeros = (
                    w1,
                    w1_ref,
                    w1_qweight,
                    w1_scales,
                    w1_qzeros,
                )
            else:
                w, w_ref, w_qweight, w_scales, w_qzeros = (
                    w2,
                    w2_ref,
                    w2_qweight,
                    w2_scales,
                    w2_qzeros,
                )
            weight, qweight, scales, qzeros = quantize_weights(
                w[expert_id].T, quant_type, group_size, has_zp, False
            )

            weight = weight.T
            qweight = qweight.T.contiguous().to(torch.uint8)
            scales = scales.T
            if has_zp:
                qzeros = qzeros.T.contiguous().to(torch.uint8)
            if weight_bits == 4:
                qweight = qweight[:, 1::2] * 16 + qweight[:, ::2]
                if has_zp:
                    qzeros = qzeros[1::2, :] * 16 + qzeros[::2, :]

            w_ref[expert_id] = weight
            w_qweight[expert_id] = qweight
            w_scales[expert_id] = scales
            if has_zp:
                w_qzeros[expert_id] = qzeros
        
        w1_qweight, w1_scales, w1_qzeros = process_weights_after_loading(w1_qweight, w1_scales, w1_qzeros if has_zp else None, types)
        w2_qweight, w2_scales, w2_qzeros = process_weights_after_loading(w2_qweight, w2_scales, w2_qzeros if has_zp else None, types)

        topk_output = select_experts(
            hidden_states=a,
            router_logits=score,
            top_k=topk,
        )

        machete_output = fused_experts_machete_impl(
            a,
            w1_qweight,
            w2_qweight,
            topk_weights=topk_output.topk_weights,
            topk_ids=topk_output.topk_ids,
            w1_scale=w1_scales,
            w2_scale=w2_scales,
            w1_zp=w1_qzeros if has_zp else None,
            w2_zp=w2_qzeros if has_zp else None,
        )
        torch_output = torch_moe(a, w1_ref, w2_ref, score, topk)
        torch.testing.assert_close(machete_output, torch_output, atol=1, rtol=0)