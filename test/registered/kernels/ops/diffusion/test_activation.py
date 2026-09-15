"""``diffusion.activation``: activation-function fusions.

All of these are bit-exact by construction -- they are elementwise chains with
no reduction, so reproducing aten's per-op fp32-opmath / round-to-bf16
boundaries is enough and ``torch.equal`` is the assertion.

The cublasLt linear+tanh-GELU epilogue is *not* here: it is not bit-exact and
is therefore quality-gated, so it is tested through its mount protocol in
``test_sites.py``.
"""

import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.kernels.ops.diffusion import (
    can_use_fused_bias_glu,
    can_use_fused_bias_silu,
    can_use_fused_silu_mul,
    fused_bias_glu,
    fused_bias_silu,
    fused_packed_silu_mul_bitexact,
    fused_silu_mul_bitexact,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=6, stage="base-b-kernel-unit", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("channels", [2240, 11200])
def test_sana_bias_silu_is_bit_exact(channels):
    torch.manual_seed(0)
    x = torch.randn(
        (1, channels, 7, 5),
        device="cuda",
        dtype=torch.bfloat16,
    ).to(memory_format=torch.channels_last)
    bias = torch.randn(channels, device="cuda", dtype=torch.bfloat16)

    assert can_use_fused_bias_silu(x, bias)
    actual = fused_bias_silu(x, bias)
    expected = F.silu(x + bias[None, :, None, None])

    assert actual.is_contiguous(memory_format=torch.channels_last)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("channels", [2240, 5600])
def test_sana_bias_glu_is_bit_exact(channels):
    torch.manual_seed(1)
    x = torch.randn(
        (1, 2 * channels, 7, 5),
        device="cuda",
        dtype=torch.bfloat16,
    ).to(memory_format=torch.channels_last)
    bias = torch.randn(2 * channels, device="cuda", dtype=torch.bfloat16)

    assert can_use_fused_bias_glu(x, bias)
    actual = fused_bias_glu(x, bias)
    biased = x + bias[None, :, None, None]
    hidden, gate = torch.chunk(biased, 2, dim=1)
    expected = hidden * F.silu(gate)

    assert actual.is_contiguous(memory_format=torch.channels_last)
    assert torch.equal(actual, expected)


# ---------------------------------------------------------------------------
# silu(a) * b for split-projection SwiGLU
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(1, 4096, 3072), (2, 17, 512)])
def test_silu_mul_is_bit_exact(shape):
    # Separate gate/up GEMMs, so the concatenated ``silu_and_mul`` kernels do
    # not apply without an extra full-width cat -- this kernel replaces the
    # eager ``F.silu(a) * b`` pair instead.
    torch.manual_seed(0)
    a = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    b = torch.randn_like(a)

    assert can_use_fused_silu_mul(a, b)
    assert torch.equal(fused_silu_mul_bitexact(a, b), F.silu(a) * b)


@pytest.mark.parametrize("hidden", [384, 3072])
@pytest.mark.parametrize("strided", [False, True])
def test_packed_silu_mul_is_bit_exact(hidden, strided):
    # The packed form splits one [.., 2 * hidden] projection in-kernel; it must
    # accept the strided view a wider projection slice produces.
    torch.manual_seed(1)
    if strided:
        x = torch.randn(1, 19, 3 * hidden, device="cuda", dtype=torch.bfloat16)
        x = x[..., : 2 * hidden]
    else:
        x = torch.randn(1, 19, 2 * hidden, device="cuda", dtype=torch.bfloat16)

    expected = F.silu(x[..., :hidden]) * x[..., hidden:]
    assert torch.equal(fused_packed_silu_mul_bitexact(x), expected)


def test_silu_mul_rejects_mismatched_operands():
    a = torch.randn(1, 8, 64, device="cuda", dtype=torch.bfloat16)
    assert not can_use_fused_silu_mul(a, a.float())  # mixed dtypes
    assert not can_use_fused_silu_mul(a, a[:, :-1])  # mismatched shapes


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
