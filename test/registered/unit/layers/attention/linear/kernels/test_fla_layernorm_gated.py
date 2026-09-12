import pytest
import torch
import torch.nn.functional as F

from sglang.kernels.ops.attention.fla.layernorm_gated import RMSNorm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


@pytest.mark.parametrize("activation", ["swish", "silu", "sigmoid"])
@pytest.mark.parametrize("norm_before_gate", [False, True])
def test_forward_native_gate(activation: str, norm_before_gate: bool) -> None:
    torch.manual_seed(0)
    norm = RMSNorm(
        16,
        dtype=torch.float32,
        activation=activation,
        norm_before_gate=norm_before_gate,
    )
    norm.weight.data.normal_()
    x, z = torch.randn(4, 16), torch.randn(4, 16)

    gate = torch.sigmoid(z) if activation == "sigmoid" else F.silu(z)
    norm_input = x if norm_before_gate else x * gate
    normalized = (
        norm_input
        * torch.rsqrt(norm_input.square().mean(dim=-1, keepdim=True) + norm.eps)
        * norm.weight
    )
    expected = normalized * gate if norm_before_gate else normalized

    actual = norm.forward_native(x, z)
    assert actual.shape == x.shape
    torch.testing.assert_close(actual, expected)


def test_forward_native_accepts_strided_three_dimensional_gate() -> None:
    torch.manual_seed(0)
    norm = RMSNorm(16, dtype=torch.float32)
    norm.weight.data.normal_()
    x = torch.randn(6, 16)
    z = torch.randn(2, 3, 32)[..., :16]
    assert not z.is_contiguous()
    gate = F.silu(z).reshape_as(x)
    expected = (
        x
        * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + norm.eps)
        * norm.weight
        * gate
    )

    torch.testing.assert_close(norm.forward_native(x, z), expected)


def test_forward_native_rejects_broadcast_gate() -> None:
    norm = RMSNorm(16, dtype=torch.float32)
    x = torch.randn(4, 16)
    z = torch.randn(1, 16)

    with pytest.raises(ValueError, match="Unsupported gated RMSNorm shapes"):
        norm.forward_native(x, z)


def test_forward_native_rejects_a_same_size_incompatible_gate() -> None:
    norm = RMSNorm(16, dtype=torch.float32)
    with pytest.raises(ValueError, match="Unsupported gated RMSNorm shapes"):
        norm.forward_native(torch.randn(4, 16), torch.randn(16, 4))
