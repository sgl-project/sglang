import sys

import pytest
import torch

from sglang.test.ci.ci_register import register_mps_ci

register_mps_ci(est_time=10, suite="stage-a-unit-test-mps")

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="requires Apple MPS"
)

DEVICE = "mps"
EPS = 1e-6
# One output ulp: fp32 differs only by reduction order, half types by rounding.
TOLERANCE = {
    torch.float32: dict(atol=1e-5, rtol=1e-5),
    torch.float16: dict(atol=2e-3, rtol=2e-3),
    torch.bfloat16: dict(atol=1.6e-2, rtol=1.6e-2),
}
# Odd widths leave idle threads in the last SIMD group; 4100 > 1024 makes
# each thread loop over several elements.
SHAPES = [(1, 7), (3, 64), (5, 1000), (2, 3, 2048), (4, 4100)]


def _make_norm(hidden_size, dtype, **kwargs):
    from sglang.srt.layers.layernorm import RMSNorm

    norm = RMSNorm(hidden_size, eps=EPS, **kwargs).to(device=DEVICE, dtype=dtype)
    with torch.no_grad():
        norm.weight.normal_()
    return norm


@pytest.mark.parametrize("dtype", list(TOLERANCE))
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("with_residual", [False, True])
def test_rmsnorm_mps_matches_native(dtype, shape, with_residual):
    from sglang.kernels.ops.layernorm.rmsnorm_mps import rmsnorm_mps

    torch.manual_seed(0)
    norm = _make_norm(shape[-1], dtype)
    x = torch.randn(shape, device=DEVICE, dtype=dtype)
    residual = torch.randn_like(x) if with_residual else None
    inputs = (x.clone(), None if residual is None else residual.clone())

    expected = norm.forward_native(x, residual)
    actual = rmsnorm_mps(x, norm.weight, EPS, residual)

    if with_residual:
        torch.testing.assert_close(actual[0], expected[0], **TOLERANCE[dtype])
        # Both paths round the same fp32 sum, so the residual is exact.
        torch.testing.assert_close(actual[1], expected[1], atol=0, rtol=0)
    else:
        torch.testing.assert_close(actual, expected, **TOLERANCE[dtype])
    torch.testing.assert_close(x, inputs[0], atol=0, rtol=0)
    if residual is not None:
        torch.testing.assert_close(residual, inputs[1], atol=0, rtol=0)


def test_rmsnorm_dispatches_to_mps_kernel(monkeypatch):
    import sglang.kernels.ops.layernorm.rmsnorm_mps as rmsnorm_mps_module

    calls = []
    kernel = rmsnorm_mps_module.rmsnorm_mps

    def counting_kernel(*args, **kwargs):
        calls.append(args)
        return kernel(*args, **kwargs)

    monkeypatch.setattr(rmsnorm_mps_module, "rmsnorm_mps", counting_kernel)
    x = torch.randn(2, 64, device=DEVICE, dtype=torch.bfloat16)

    _make_norm(64, torch.bfloat16)(x)
    assert len(calls) == 1

    # Variants the kernel does not implement must take the native path.
    _make_norm(64, torch.bfloat16, fp32_residual=True)(x, torch.randn_like(x))
    _make_norm(64, torch.bfloat16, var_hidden_size=32)(x)
    _make_norm(64, torch.bfloat16)(x.float())
    assert len(calls) == 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
