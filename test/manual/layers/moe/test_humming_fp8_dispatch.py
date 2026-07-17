import pytest
import torch

from sglang.jit_kernel.dsv4.moe import silu_and_mul_masked_post_quant
from sglang.kernels.ops.moe.ep_moe_kernels import moe_permute_with_scale

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="Humming FP8 dispatch requires SM90",
)


def _reference(gate_up, swiglu_limit, group_size=128):
    intermediate = gate_up.shape[-1] // 2
    gate = gate_up[..., :intermediate].float()
    up = gate_up[..., intermediate:].float()
    if swiglu_limit is not None:
        gate = gate.clamp_max(swiglu_limit)
        up = up.clamp(-swiglu_limit, swiglu_limit)
    activation = gate * torch.sigmoid(gate) * up
    grouped = activation.view(*activation.shape[:-1], -1, group_size)
    scale = grouped.abs().amax(dim=-1).clamp_min(1e-10) / 448.0
    output = (grouped / scale[..., None]).to(torch.float8_e4m3fn)
    return output.flatten(-2), scale


def test_grouped_contiguous_permute_scale():
    torch.manual_seed(0)
    inputs = torch.randn(5, 256, device="cuda", dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    input_scale = torch.arange(10, device="cuda", dtype=torch.float32).view(5, 2)
    topk_ids = torch.tensor(
        [[2, 0], [1, -1], [0, 2], [1, 0], [-1, 2]],
        device="cuda",
        dtype=torch.int32,
    )

    outputs, output_scale, src2dst, expert_offsets = moe_permute_with_scale(
        inputs=inputs,
        input_scale=input_scale,
        topk_ids=topk_ids,
        num_experts=3,
        is_ep=True,
    )

    for token in range(topk_ids.size(0)):
        for slot in range(topk_ids.size(1)):
            index = token * topk_ids.size(1) + slot
            dst = int(src2dst[index])
            if topk_ids[token, slot] < 0:
                assert dst < 0
                continue
            assert torch.equal(outputs[dst], inputs[token])
            assert torch.equal(output_scale[dst], input_scale[token])

    assert torch.equal(
        expert_offsets, torch.tensor([0, 3, 5, 8], device="cuda", dtype=torch.int32)
    )


@pytest.mark.parametrize("swiglu_limit", [None, 10.0])
def test_grouped_masked_w2_quant(swiglu_limit):
    torch.manual_seed(0)
    num_experts, max_tokens, intermediate = 8, 64, 3072
    gate_up = (
        torch.randn(
            num_experts,
            max_tokens,
            2 * intermediate,
            device="cuda",
            dtype=torch.bfloat16,
        )
        * 3
    )
    masked_m = torch.randint(
        0, max_tokens + 1, (num_experts,), dtype=torch.int32, device="cuda"
    )
    output = torch.empty(
        num_experts,
        max_tokens,
        intermediate,
        device="cuda",
        dtype=torch.float8_e4m3fn,
    )
    output.view(torch.uint8).fill_(0xAA)
    scale = torch.full(
        (num_experts, max_tokens, intermediate // 128),
        float("nan"),
        device="cuda",
    )

    silu_and_mul_masked_post_quant(
        gate_up,
        output,
        scale,
        128,
        masked_m,
        scale_ue8m0=False,
        topk=8,
        swiglu_limit=swiglu_limit,
    )
    ref_output, ref_scale = _reference(gate_up, swiglu_limit)

    for expert in range(num_experts):
        valid = int(masked_m[expert])
        if valid < max_tokens:
            assert (output[expert, valid:].view(torch.uint8) == 0xAA).all()
            assert scale[expert, valid:].isnan().all()
        if valid == 0:
            continue

        actual_scale = scale[expert, :valid]
        expected_scale = ref_scale[expert, :valid]
        relative_error = (
            actual_scale - expected_scale
        ).abs() / expected_scale.abs().clamp_min(1e-30)
        assert relative_error.max() < 1e-3

        actual = output[expert, :valid].view(torch.uint8).to(torch.int16)
        expected = ref_output[expert, :valid].view(torch.uint8).to(torch.int16)
        assert ((actual - expected).abs() <= 1).float().mean() >= 0.9999
