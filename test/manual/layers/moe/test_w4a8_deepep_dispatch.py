import pytest
import torch

from sglang.kernels.ops.moe.ep_moe_kernels import (
    deepep_permute_fp8_to_per_tensor_quant,
    fp8_per_token_to_per_tensor_quant_triton,
)
from sglang.kernels.ops.quantization.fp8_kernel import (
    sglang_per_token_group_quant_fp8,
)
from sglang.srt.layers.moe.token_dispatcher.deepep import (
    DeepEPNormalDispatchOutput,
)
from sglang.srt.layers.quantization.w4afp8 import W4AFp8MoEMethod


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("column_major_scales", [False, True])
def test_deepep_normal_fp8_reorder_and_requantize(column_major_scales):
    torch.manual_seed(0)
    num_tokens, hidden_size, topk = 3, 1024, 2
    source = torch.randn(
        (num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16
    )
    source_fp8, source_scale = sglang_per_token_group_quant_fp8(
        source,
        128,
        column_major_scales=column_major_scales,
    )

    # One invalid route exercises the same src2dst contract as DeepEP.
    src2dst = torch.tensor(
        [[2, -1], [0, 3], [1, 4]], device="cuda", dtype=torch.int64
    )
    output_scale = torch.tensor([0.02], device="cuda", dtype=torch.float32)
    output = torch.zeros(
        (5, hidden_size), device="cuda", dtype=torch.float8_e4m3fn
    )

    deepep_permute_fp8_to_per_tensor_quant(
        input=source_fp8,
        input_scale=source_scale,
        gateup_input=output,
        src2dst=src2dst,
        output_scale=output_scale,
        topk=topk,
    )

    source_dequant = source_fp8.float() * source_scale.float().repeat_interleave(
        128, dim=1
    )
    expected_rows = (source_dequant / output_scale).to(torch.float8_e4m3fn)
    expected = torch.zeros_like(output)
    for token_id in range(num_tokens):
        for route_id in range(topk):
            dst = src2dst[token_id, route_id].item()
            if dst >= 0:
                expected[dst] = expected_rows[token_id]

    torch.testing.assert_close(output.float(), expected.float(), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_deepep_low_latency_fp8_requantize():
    torch.manual_seed(1)
    num_experts, max_tokens, hidden_size = 2, 5, 1024
    source = torch.randn(
        (num_experts, max_tokens, hidden_size),
        device="cuda",
        dtype=torch.bfloat16,
    )
    source_fp8, source_scale = sglang_per_token_group_quant_fp8(source, 128)
    masked_m = torch.tensor([5, 3], device="cuda", dtype=torch.int32)
    output_scale = torch.tensor([0.02], device="cuda", dtype=torch.float32)
    output = torch.zeros_like(source_fp8)

    fp8_per_token_to_per_tensor_quant_triton(
        x=source_fp8,
        x_scale=source_scale,
        masked_m=masked_m,
        output_scale=output_scale,
        output=output,
    )

    source_dequant = source_fp8.float() * source_scale.float().repeat_interleave(
        128, dim=2
    )
    expected = (source_dequant / output_scale).to(torch.float8_e4m3fn)
    for expert_id, valid_tokens in enumerate(masked_m.tolist()):
        torch.testing.assert_close(
            output[expert_id, :valid_tokens].float(),
            expected[expert_id, :valid_tokens].float(),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            output[expert_id, valid_tokens:].float(),
            torch.zeros_like(output[expert_id, valid_tokens:].float()),
            rtol=0,
            atol=0,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_deepep_normal_empty_fp8_dispatch_returns_bf16_for_combine():
    hidden_size, topk = 1024, 2
    dispatch_output = DeepEPNormalDispatchOutput(
        hidden_states=torch.empty(
            (0, hidden_size), device="cuda", dtype=torch.float8_e4m3fn
        ),
        hidden_states_scale=torch.empty((0, hidden_size // 128), device="cuda"),
        topk_ids=torch.empty((0, topk), device="cuda", dtype=torch.int64),
        topk_weights=torch.empty((0, topk), device="cuda", dtype=torch.float32),
        num_recv_tokens_per_expert=[],
    )

    method = object.__new__(W4AFp8MoEMethod)
    output = method.apply_deepep_normal(layer=None, dispatch_output=dispatch_output)

    assert output.shape == (0, hidden_size)
    assert output.dtype == torch.bfloat16
