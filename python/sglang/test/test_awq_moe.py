import types
from typing import Optional

import pytest
import torch

from sglang.srt.layers.moe.topk import select_experts
from sglang.srt.layers.quantization.awq import AWQMarlinConfig, AWQMoEMethod


def pack_int4_to_int32(w_int4: torch.Tensor) -> torch.Tensor:
    """Pack the last dim (nibble-wise) to int32."""
    w_int8 = w_int4.to(torch.int8)
    assert w_int8.shape[-1] % 8 == 0
    w_int32 = w_int8.reshape(-1, 8)

    packed = torch.zeros(w_int32.shape[0], dtype=torch.int32, device=w_int4.device)
    for i in range(8):
        packed |= (w_int32[:, i] & 0x0F) << (4 * i)

    return packed.reshape(w_int8.shape[:-1] + (-1,))


class DummyLayer(torch.nn.Module):
    """A minimal nn.Module stub that can carry quantized weight tensors."""

    pass


def ref_awq_marlin_moe(
    x: torch.Tensor,
    w13_qweight: torch.Tensor,
    w2_qweight: torch.Tensor,
    w13_scales: torch.Tensor,
    w2_scales: torch.Tensor,
    w13_qzeros: torch.Tensor,
    w2_qzeros: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    group_size: int,
    pack_factor: int = 8,  # 4-bit → 32-bit => 8 nibbles
):
    num_experts = w13_qweight.shape[0]
    hidden_size = x.shape[1]
    intermediate_size = w2_qweight.shape[1]
    results = torch.zeros_like(x)
    dtype = x.dtype

    zeros_int4_w13 = torch.empty(
        (num_experts, w13_qzeros.shape[1], intermediate_size * 2),
        dtype=torch.int8,
        device=x.device,
    )
    zeros_int4_w2 = torch.empty(
        (num_experts, w2_qzeros.shape[1], hidden_size),
        dtype=torch.int8,
        device=x.device,
    )
    for i in range(pack_factor):
        zeros_int4_w13[:, :, i::pack_factor] = w13_qzeros >> (i * 4)
        zeros_int4_w2[:, :, i::pack_factor] = w2_qzeros >> (i * 4)
    zeros_int4_w13 = (zeros_int4_w13 & 0x0F).to(dtype)
    zeros_int4_w2 = (zeros_int4_w2 & 0x0F).to(dtype)

    w_unpacked = torch.empty(
        num_experts,
        hidden_size,
        intermediate_size * 2,
        dtype=torch.int8,
        device=x.device,
    )
    for i in range(pack_factor):
        w_unpacked[:, :, i::pack_factor] = w13_qweight >> (i * 4)
    w_unpacked = (w_unpacked & 0x0F).permute(0, 2, 1).to(dtype)  # E,2*inter,K
    s = w13_scales.repeat_interleave(group_size, dim=1).permute(0, 2, 1)
    z = zeros_int4_w13.repeat_interleave(group_size, dim=1).permute(0, 2, 1)
    w13_dequant = (w_unpacked - z) * s  # E,2*inter,K

    w_unpacked = torch.empty(
        num_experts, intermediate_size, hidden_size, dtype=torch.int8, device=x.device
    )
    for i in range(pack_factor):
        w_unpacked[:, :, i::pack_factor] = w2_qweight >> (i * 4)
    w_unpacked = (w_unpacked & 0x0F).permute(0, 2, 1).to(dtype)  # E,K,inter
    s = w2_scales.repeat_interleave(group_size, dim=1).permute(0, 2, 1)
    z = zeros_int4_w2.repeat_interleave(group_size, dim=1).permute(0, 2, 1)
    w2_dequant = (w_unpacked - z) * s  # E,K,inter

    for i in range(x.shape[0]):
        token_input = x[i, :]
        for k_idx in range(topk_ids.shape[1]):
            expert_id = topk_ids[i, k_idx].item()
            router_weight = topk_weights[i, k_idx]
            if expert_id >= num_experts:
                continue
            w13 = w13_dequant[expert_id]
            w2 = w2_dequant[expert_id]
            gate_up = torch.matmul(token_input, w13.T)
            gate, up = gate_up.chunk(2, dim=-1)
            hidden = torch.nn.functional.silu(gate) * up
            down = torch.matmul(hidden, w2.T)
            results[i, :] += down * router_weight
    return results.to(dtype)


@pytest.mark.parametrize("M", [16])  # num_tokens
@pytest.mark.parametrize("K", [1024])  # hidden_size
@pytest.mark.parametrize("N", [2816])  # intermediate_size
@pytest.mark.parametrize("E", [8])  # num_experts
@pytest.mark.parametrize("topk", [2])
@pytest.mark.parametrize("group_size", [128])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_awq_marlin_moe(M, K, N, E, topk, group_size, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA device not available")

    torch.manual_seed(0)

    device = "cuda"
    quant_config = AWQMarlinConfig(weight_bits=4, group_size=group_size)
    awq_moe_method = AWQMoEMethod(quant_config)
    pack_factor = quant_config.pack_factor  # 8

    layer = DummyLayer().to(device)
    x = torch.randn(M, K, dtype=dtype, device=device)
    router_logits = torch.randn(M, E, dtype=dtype, device=device)

    w13_q_unpacked = torch.randint(
        0, 16, (E, K, N * 2), dtype=torch.int8, device=device
    )
    layer.w13_qweight = pack_int4_to_int32(w13_q_unpacked).requires_grad_(False)
    layer.w13_scales = torch.randn(
        E, K // group_size, N * 2, dtype=dtype, device=device
    ).requires_grad_(False)
    w13_z_unpacked = torch.randint(
        0, 16, (E, K // group_size, N * 2), dtype=torch.int8, device=device
    )
    layer.w13_qzeros = pack_int4_to_int32(w13_z_unpacked).requires_grad_(False)

    w2_q_unpacked = torch.randint(0, 16, (E, N, K), dtype=torch.int8, device=device)
    layer.w2_qweight = pack_int4_to_int32(w2_q_unpacked).requires_grad_(False)
    layer.w2_scales = torch.randn(
        E, N // group_size, K, dtype=dtype, device=device
    ).requires_grad_(False)
    w2_z_unpacked = torch.randint(
        0, 16, (E, N // group_size, K), dtype=torch.int8, device=device
    )
    layer.w2_qzeros = pack_int4_to_int32(w2_z_unpacked).requires_grad_(False)

    topk_weights, topk_ids = select_experts(
        hidden_states=x, router_logits=router_logits, top_k=topk
    )

    ref_output = ref_awq_marlin_moe(
        x,
        layer.w13_qweight,
        layer.w2_qweight,
        layer.w13_scales,
        layer.w2_scales,
        layer.w13_qzeros,
        layer.w2_qzeros,
        topk_weights,
        topk_ids,
        group_size,
        pack_factor,
    )

    awq_moe_method.create_weights(layer, E, K, N, dtype)
    layer.intermediate_size_per_partition = N
    awq_moe_method.process_weights_after_loading(layer)

    output = awq_moe_method.apply(layer, x, router_logits, topk, renormalize=False)

    torch.testing.assert_close(output, ref_output, rtol=1e-2, atol=1e-2)
