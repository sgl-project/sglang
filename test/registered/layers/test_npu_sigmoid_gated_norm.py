"""Regression coverage for the NPU sigmoid gated-norm adapter."""

import sys

import pytest
import torch
import torch.nn.functional as F

import sglang.kernels.ops.attention.fla.layernorm_gated as gated_norm
from sglang.kernels.ops.attention.fla.layernorm_gated import layernorm_fn, rms_norm_ref
from sglang.test.ci.ci_register import register_cpu_ci, register_npu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_npu_ci(est_time=10, suite="base-b-test-1-npu-a3")


def layer_norm_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    z: torch.Tensor | None = None,
    eps: float = 1e-6,
    group_size: int | None = None,
    norm_before_gate: bool = True,
    is_rms_norm: bool = False,
) -> torch.Tensor:
    """Reference implementation for both LayerNorm and RMSNorm (supports optional gate + group norm)."""
    if is_rms_norm:
        return rms_norm_ref(
            x,
            weight,
            bias,
            z=z,
            eps=eps,
            group_size=group_size,
            norm_before_gate=norm_before_gate,
            upcast=True,
        )

    dtype = x.dtype
    x_f = x.float()
    w_f = weight.float()
    b_f = bias.float() if bias is not None else None
    z_f = z.float() if z is not None else None

    if z_f is not None and not norm_before_gate:
        x_f = x_f * F.silu(z_f)

    if group_size is None:
        mean = x_f.mean(dim=-1, keepdim=True)
        var = (x_f - mean).square().mean(dim=-1, keepdim=True)
        rstd = torch.rsqrt(var + eps)
        out = (x_f - mean) * rstd * w_f
        if b_f is not None:
            out = out + b_f
    else:
        hidden = x_f.shape[-1]
        assert hidden % group_size == 0
        ng = hidden // group_size
        xg = x_f.view(*x_f.shape[:-1], ng, group_size)
        mean = xg.mean(dim=-1, keepdim=True)
        var = (xg - mean).square().mean(dim=-1, keepdim=True)
        rstd = torch.rsqrt(var + eps)
        xg = (xg - mean) * rstd
        out = xg.reshape(*x_f.shape[:-1], hidden) * w_f
        if b_f is not None:
            out = out + b_f

    if z_f is not None and norm_before_gate:
        out = out * F.silu(z_f)

    return out.to(dtype)


@pytest.mark.parametrize("device", ["cpu", "npu"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("norm_before_gate", [True, False])
@pytest.mark.parametrize("layout", ["2d", "3d_input", "strided_gate", "no_gate"])
@pytest.mark.parametrize("is_rms_norm,group_size", [(True, None), (False, 32)])
def test_npu_sigmoid_gated_norm(
    monkeypatch, device, dtype, norm_before_gate, layout, is_rms_norm, group_size
):
    if device == "npu":
        if not gated_norm._is_npu:
            pytest.skip("NPU is not available")
    else:
        # Exercise adapter dispatch even on CI machines without an NPU.
        def ungated_norm(x, weight, bias, eps, **kwargs):
            assert kwargs["z"] is None
            return layer_norm_ref(x, weight, bias, eps=eps, **kwargs), None, None

        monkeypatch.setattr(gated_norm, "_is_npu", True)
        monkeypatch.setattr(gated_norm, "_layer_norm_fwd", ungated_norm)

    generator = torch.Generator(device="cpu").manual_seed(42)
    x = torch.randn(15, 128, generator=generator).to(dtype)
    weight = torch.randn(128, generator=generator).to(dtype)
    bias = None if is_rms_norm else torch.randn(128, generator=generator).to(dtype)
    # Include zero and saturated positive/negative gates.
    z = torch.linspace(-20, 20, 15 * 128).reshape(15, 128).to(dtype)
    z[:, 0] = 0
    if layout == "3d_input":
        x, z = x.reshape(5, 3, 128), z.reshape(5, 3, 128)
    elif layout == "no_gate":
        z = None

    # Keep the reference in FP32 through norm and gate, without the adapter's
    # intermediate low-precision rounding at the kernel boundary.
    gate = None if z is None else z.float().sigmoid()
    ref_input = x.float()
    if gate is not None and not norm_before_gate:
        ref_input = ref_input * gate
    expected = layer_norm_ref(
        ref_input,
        weight.float(),
        None if bias is None else bias.float(),
        group_size=group_size,
        is_rms_norm=is_rms_norm,
    )
    if gate is not None and norm_before_gate:
        expected = expected * gate

    x, weight = x.to(device), weight.to(device)
    bias = None if bias is None else bias.to(device)
    z = None if z is None else z.to(device)
    if layout == "strided_gate":
        storage = torch.empty((5, 3, 256), device=device, dtype=dtype)
        storage[..., :128].copy_(z.reshape(5, 3, 128))
        z = storage[..., :128]
        assert not z.is_contiguous()
    actual = layernorm_fn(
        x,
        weight,
        bias,
        z=z,
        group_size=group_size,
        norm_before_gate=norm_before_gate,
        is_rms_norm=is_rms_norm,
        activation="sigmoid",
    )
    assert actual.shape == x.shape
    assert actual.dtype == dtype
    tolerance = 2e-2 if dtype == torch.bfloat16 else 3e-3
    torch.testing.assert_close(
        actual.cpu(), expected.to(dtype), atol=tolerance, rtol=tolerance
    )


@pytest.mark.parametrize("is_npu,activation", [(True, "swish"), (False, "sigmoid")])
def test_npu_sigmoid_adapter_preserves_other_dispatch(monkeypatch, is_npu, activation):
    x = torch.ones(2, 128)
    z = torch.zeros_like(x)
    calls = []

    def backend(x, weight, bias, eps, **kwargs):
        calls.append(kwargs)
        return x, None, None

    monkeypatch.setattr(gated_norm, "_is_npu", is_npu)
    monkeypatch.setattr(gated_norm, "_layer_norm_fwd", backend)
    layernorm_fn(x, torch.ones(128), None, z=z, activation=activation)
    assert len(calls) == 1
    assert calls[0]["activation"] == activation
    torch.testing.assert_close(calls[0]["z"], z)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
