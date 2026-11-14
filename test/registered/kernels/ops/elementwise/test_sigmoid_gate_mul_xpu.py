import sys
from unittest.mock import patch

import pytest
import torch

from sglang.test.ci.ci_register import register_xpu_ci

register_xpu_ci(est_time=30, suite="stage-b-test-1-gpu-xpu")

pytestmark = pytest.mark.skipif(
    not hasattr(torch, "xpu") or not torch.xpu.is_available(),
    reason="requires XPU",
)

DEVICE = "xpu"


def _make_gate(num_tokens, num_heads, head_dim, dtype, strided):
    if not strided:
        return torch.randn(num_tokens, num_heads, head_dim, dtype=dtype, device=DEVICE)
    # A gate sliced out of a wider fused projection output.
    wide = torch.randn(num_tokens, num_heads, 2 * head_dim, dtype=dtype, device=DEVICE)
    return wide[..., head_dim:]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("num_tokens", [1, 7, 33])
def test_fused_sigmoid_mul_3d_gate(num_tokens, inplace, strided, dtype):
    from sglang.kernels.ops.elementwise.elementwise import fused_sigmoid_mul

    torch.manual_seed(0)
    num_heads, head_dim = 8, 128
    x = torch.randn(num_tokens, num_heads * head_dim, dtype=dtype, device=DEVICE)
    gate = _make_gate(num_tokens, num_heads, head_dim, dtype, strided)
    assert gate.stride(-1) == 1 and gate.is_contiguous() != strided

    ref = x.float() * torch.sigmoid(gate.float().reshape(num_tokens, -1))
    x_ptr = x.data_ptr()
    gate_orig = gate.clone()
    out = fused_sigmoid_mul(x, gate, inplace=inplace)

    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(gate, gate_orig, rtol=0, atol=0)
    if inplace:
        assert out.data_ptr() == x_ptr
        torch.testing.assert_close(x, out, rtol=0, atol=0)
    else:
        assert out.data_ptr() != x_ptr


@pytest.mark.parametrize("inplace", [False, True])
def test_fused_sigmoid_mul_flat_gate(inplace):
    from sglang.kernels.ops.elementwise.elementwise import fused_sigmoid_mul

    torch.manual_seed(0)
    x = torch.randn(5, 4096, dtype=torch.bfloat16, device=DEVICE)
    gate = torch.randn(5, 4096, dtype=torch.bfloat16, device=DEVICE)
    ref = x.float() * torch.sigmoid(gate.float())
    x_ptr = x.data_ptr()
    out = fused_sigmoid_mul(x, gate, inplace=inplace)
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)
    assert (out.data_ptr() == x_ptr) == inplace


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_rounded_sigmoid_keeps_triton_path(dtype):
    from sglang.kernels.ops.elementwise.elementwise import fused_sigmoid_mul

    torch.manual_seed(0)
    x = torch.randn(7, 1024, dtype=dtype, device=DEVICE)
    gate = torch.randn_like(x)
    expected = (x.float() * torch.sigmoid(gate.float()).to(dtype).float()).to(dtype)
    with patch(
        "sgl_kernel.fused_sigmoid_mul",
        side_effect=AssertionError("SYCL does not support rounded sigmoid"),
    ):
        out = fused_sigmoid_mul(x, gate, round_sigmoid=True)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
