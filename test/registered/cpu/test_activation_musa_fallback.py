import pytest
import torch

from sglang.kernels.ops.activation import activation as activation_ops
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-test-cpu-intel")
register_cpu_ci(est_time=1, suite="base-b-test-cpu-arm64")

DTYPE_TOLERANCE = {
    torch.float16: 1e-3,
    torch.bfloat16: 1.6e-2,
    torch.float32: 1e-5,
}


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_musa_activation_fallback(monkeypatch, dtype):
    monkeypatch.setattr(activation_ops, "is_musa", lambda: True)
    input = torch.randn(2, 4, 128, dtype=dtype)
    hidden_size = input.shape[-1] // 2
    gate = input[..., :hidden_size].float()
    up = input[..., hidden_size:].float()
    expected = (torch.nn.functional.silu(gate) * up).to(dtype)
    module = activation_ops.activation_module(dtype, fast_math=False)
    assert isinstance(module, activation_ops._TorchActivationModule)
    out = torch.empty_like(input[..., :hidden_size])
    module.run_activation(input, out, "silu")

    torch.testing.assert_close(
        out,
        expected,
        atol=DTYPE_TOLERANCE[dtype],
        rtol=DTYPE_TOLERANCE[dtype],
    )


def test_musa_activation_fallback_with_rounding_and_clamp(monkeypatch):
    monkeypatch.setattr(activation_ops, "is_musa", lambda: True)
    input = torch.randn(2, 4, 128, dtype=torch.float16) * 2
    hidden_size = input.shape[-1] // 2
    clamp_limit = 1.5
    gate = input[..., :hidden_size].float().clamp_max(clamp_limit)
    up = input[..., hidden_size:].float().clamp(min=-clamp_limit, max=clamp_limit)
    activated = torch.nn.functional.silu(gate).to(torch.float16).float()
    expected = (activated * up).to(torch.float16)
    module = activation_ops.activation_module(torch.float16, fast_math=False)
    out = torch.empty_like(input[..., :hidden_size])
    module.run_activation_with_rounding(input, out, "silu", clamp_limit=clamp_limit)

    torch.testing.assert_close(
        out,
        expected,
        atol=DTYPE_TOLERANCE[torch.float16],
        rtol=DTYPE_TOLERANCE[torch.float16],
    )


def test_musa_unary_activation_fallback(monkeypatch):
    monkeypatch.setattr(activation_ops, "is_musa", lambda: True)
    input = torch.randn(2, 4, 128, dtype=torch.float16)
    expected = torch.relu(input).square()
    module = activation_ops.activation_module(torch.float16, fast_math=False)
    out = torch.empty_like(input)
    module.run_unary_activation(input, out, "relu2")

    torch.testing.assert_close(
        out,
        expected,
        atol=DTYPE_TOLERANCE[torch.float16],
        rtol=DTYPE_TOLERANCE[torch.float16],
    )
