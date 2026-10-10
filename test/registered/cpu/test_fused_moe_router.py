import pytest
import sgl_kernel  # noqa: F401
import torch

from sglang.kernels.ops.moe.router import fused_moe_router_shim
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-test-cpu-intel")


def _reference_router(hidden_states, gating_output, topk, softcap, correction_bias=None):
    logits = hidden_states.float() @ gating_output.float().t()
    if softcap != 0:
        logits = torch.tanh(logits / softcap) * softcap
    if correction_bias is not None:
        logits = logits + correction_bias.float()
    return torch.topk(torch.softmax(logits, dim=-1, dtype=torch.float32), topk, dim=-1)


def _make_case(tokens, hidden_dim, num_experts, dtype, correction_bias):
    # Keep values small enough to avoid saturating the softcap while adding a
    # deterministic expert trend that makes top-k ids robust to accumulation-order
    # differences between torch matmul and the AVX512 kernel.
    hidden_states = torch.randn((tokens, hidden_dim), dtype=dtype) / 16
    gating_output = torch.randn((num_experts, hidden_dim), dtype=torch.float32) / 16
    expert_trend = torch.linspace(-0.03, 0.03, num_experts, dtype=torch.float32)
    gating_output[:, 0] += expert_trend
    bias = None
    if correction_bias:
        bias = torch.linspace(-0.05, 0.05, num_experts, dtype=torch.float32)
    return hidden_states.contiguous(), gating_output.contiguous(), bias


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "tokens,hidden_dim,num_experts,topk,softcap,correction_bias",
    [
        (1, 17, 8, 1, 0.0, False),
        (7, 31, 8, 2, 30.0, False),
        (33, 64, 16, 2, 30.0, True),
        (128, 257, 8, 2, 30.0, False),
        (16, 63, 32, 4, 0.0, True),
    ],
)
def test_fused_moe_router_cpu_matches_reference(
    dtype, tokens, hidden_dim, num_experts, topk, softcap, correction_bias
):
    torch.manual_seed(tokens * 1000 + hidden_dim + num_experts + topk)
    hidden_states, gating_output, bias = _make_case(
        tokens, hidden_dim, num_experts, dtype, correction_bias
    )

    ref_weights, ref_ids = _reference_router(
        hidden_states, gating_output, topk, softcap, bias
    )
    out_weights, out_ids = torch.ops.sgl_kernel.fused_moe_router_cpu(
        hidden_states, gating_output, topk, softcap, bias
    )

    torch.testing.assert_close(out_ids, ref_ids.to(torch.int32), rtol=0, atol=0)
    torch.testing.assert_close(out_weights, ref_weights, rtol=5e-4, atol=5e-4)


def test_fused_moe_router_shim_dispatches_cpu_kernel():
    torch.manual_seed(2026)
    hidden_states, gating_output, bias = _make_case(
        tokens=11,
        hidden_dim=48,
        num_experts=8,
        dtype=torch.bfloat16,
        correction_bias=True,
    )

    ref_weights, ref_ids = _reference_router(hidden_states, gating_output, 2, 30.0, bias)
    out_weights, out_ids = fused_moe_router_shim(
        30.0,
        hidden_states,
        gating_output,
        2,
        False,
        correction_bias=bias,
    )

    torch.testing.assert_close(out_ids, ref_ids.to(torch.int32), rtol=0, atol=0)
    torch.testing.assert_close(out_weights, ref_weights, rtol=5e-4, atol=5e-4)
