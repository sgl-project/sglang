import sys

import pytest
import torch

from sglang.test.ci.ci_register import register_mps_ci

register_mps_ci(est_time=10, suite="stage-a-unit-test-mps")

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="requires Apple MPS"
)

DEVICE = "mps"
# One output ulp: Metal's exp and MPS's silu may round differently.
TOLERANCE = {
    torch.float32: dict(atol=1e-5, rtol=1e-5),
    torch.float16: dict(atol=2e-3, rtol=2e-3),
    torch.bfloat16: dict(atol=1.6e-2, rtol=1.6e-2),
}
# (1, 2) is the smallest valid input; odd d and 3-D shapes check the row split.
SHAPES = [(1, 2), (3, 14), (5, 2000), (2, 3, 4096), (7, 22016)]


@pytest.mark.parametrize("dtype", list(TOLERANCE))
@pytest.mark.parametrize("shape", SHAPES)
def test_silu_and_mul_mps_matches_native(dtype, shape):
    from sglang.kernels.ops.activation.silu_and_mul_mps import silu_and_mul_mps
    from sglang.srt.layers.activation import SiluAndMul

    torch.manual_seed(0)
    x = torch.randn(shape, device=DEVICE, dtype=dtype) * 4
    x_before = x.clone()

    expected = SiluAndMul().forward_native(x)
    actual = silu_and_mul_mps(x)

    torch.testing.assert_close(actual, expected, **TOLERANCE[dtype])
    torch.testing.assert_close(x, x_before, atol=0, rtol=0)


def test_silu_and_mul_mps_handles_strided_input():
    from sglang.kernels.ops.activation.silu_and_mul_mps import silu_and_mul_mps
    from sglang.srt.layers.activation import SiluAndMul

    x = torch.randn(64, 8, device=DEVICE).t()
    torch.testing.assert_close(
        silu_and_mul_mps(x), SiluAndMul().forward_native(x), **TOLERANCE[x.dtype]
    )


def test_silu_and_mul_dispatches_to_mps_kernel(monkeypatch):
    import sglang.kernels.ops.activation.silu_and_mul_mps as silu_and_mul_mps_module
    from sglang.srt.layers.activation import SiluAndMul

    calls = []
    kernel = silu_and_mul_mps_module.silu_and_mul_mps

    def counting_kernel(*args, **kwargs):
        calls.append(args)
        return kernel(*args, **kwargs)

    monkeypatch.setattr(silu_and_mul_mps_module, "silu_and_mul_mps", counting_kernel)

    SiluAndMul()(torch.randn(2, 64, device=DEVICE, dtype=torch.bfloat16))
    assert len(calls) == 1

    # A zero-thread launch is invalid, so an empty batch must take the native path.
    out = SiluAndMul()(torch.randn(0, 64, device=DEVICE, dtype=torch.bfloat16))
    assert out.shape == (0, 32)
    assert len(calls) == 1

    # A CPU tensor also takes the native path.
    x = torch.randn(2, 64, dtype=torch.bfloat16)
    layer = SiluAndMul()
    torch.testing.assert_close(layer(x), layer.forward_native(x))
    assert len(calls) == 1


@pytest.mark.parametrize(
    "make_input",
    [
        pytest.param(lambda: torch.randn(2, 63, device=DEVICE), id="odd_last_dim"),
        pytest.param(
            lambda: torch.ones(2, 64, device=DEVICE, dtype=torch.int32),
            id="unsupported_dtype",
        ),
        pytest.param(lambda: torch.randn(2, 64), id="cpu_tensor"),
        pytest.param(lambda: torch.randn(0, 64, device=DEVICE), id="empty"),
        # A stride-0 view reaches 2**32 elements without allocating them.
        pytest.param(
            lambda: torch.ones(1, 1, device=DEVICE).expand(2**16, 2**16),
            id="numel_2_pow_32",
        ),
    ],
)
def test_silu_and_mul_mps_supported_rejects(make_input):
    from sglang.kernels.ops.activation.silu_and_mul_mps import (
        silu_and_mul_mps_supported,
    )

    assert silu_and_mul_mps_supported(torch.randn(2, 64, device=DEVICE))
    assert not silu_and_mul_mps_supported(make_input())


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
