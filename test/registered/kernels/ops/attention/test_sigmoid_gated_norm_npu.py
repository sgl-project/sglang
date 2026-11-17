"""Sigmoid gating at the NPU normalization boundary."""

import sys

import pytest
import torch

from sglang.kernels.ops.attention.fla import layernorm_gated
from sglang.test.ci.ci_register import register_cpu_ci, register_npu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_npu_ci(est_time=10, suite="base-b-test-1-npu-a3")


@pytest.mark.parametrize("device", ["cpu", "npu"])
@pytest.mark.parametrize(
    "tokens,strided,norm_before_gate",
    [
        pytest.param(1, False, True, id="qwen3.8-flash-next-decode-t1-attn-tp8"),
        pytest.param(4, True, True, id="qwen3.8-flash-next-verify-t4-attn-tp8"),
        pytest.param(4, False, False, id="gate-before-norm"),
    ],
)
def test_sigmoid_gated_norm(monkeypatch, device, tokens, strided, norm_before_gate):
    if device == "npu" and not layernorm_gated._is_npu:
        pytest.skip("NPU is required")
    if device == "cpu":
        # Only replace the external kernel; run the actual sigmoid adapter.
        def ungated_norm(x, weight, bias, eps, **kwargs):
            y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
            return (y * weight.float()).to(x.dtype), None, None

        monkeypatch.setattr(layernorm_gated, "_is_npu", True)
        monkeypatch.setattr(layernorm_gated, "_layer_norm_fwd", ungated_norm)

    torch.manual_seed(42)
    # TP8 has 48 / 8 = 6 local value heads, each with 128 channels.
    x = torch.randn(tokens * 6, 128, dtype=torch.bfloat16, device=device)
    weight = torch.randn(128, dtype=x.dtype, device=device)
    z = torch.linspace(-8, 8, x.numel(), device=device).reshape_as(x).to(x.dtype)
    if strided:
        storage = torch.empty(tokens, 6, 256, dtype=x.dtype, device=device)
        storage[..., :128].copy_(z.reshape(tokens, 6, 128))
        z = storage[..., :128]

    gate = z.cpu().float().reshape_as(x).sigmoid()
    norm_input = x.cpu().float() if norm_before_gate else x.cpu().float() * gate
    expected = norm_input * torch.rsqrt(
        norm_input.square().mean(-1, keepdim=True) + 1e-6
    )
    expected *= weight.cpu().float()
    if norm_before_gate:
        expected *= gate

    actual = layernorm_gated.layernorm_fn(
        x,
        weight,
        None,
        z=z,
        eps=1e-6,
        is_rms_norm=True,
        activation="sigmoid",
        norm_before_gate=norm_before_gate,
    )
    assert actual.shape == x.shape and actual.dtype == x.dtype
    torch.testing.assert_close(actual.cpu().float(), expected, atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
