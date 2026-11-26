"""Every-backend-vs-native parity for BaseFusedOp ops on real GPU (RFC #29630).

For each reworked fused op, run every backend eligible on this platform and
assert it matches the pure-torch ``forward_native`` reference within dtype
tolerance. New backends are picked up automatically.
"""

import math

import pytest
import torch

from sglang.kernels.spec import KernelBackend
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="extra-a", runner_config="1-gpu-small")

_SKIP = {KernelBackend.TORCH, KernelBackend.TORCH_COMPILE}

_TOL = {
    torch.float16: dict(atol=1e-2, rtol=1e-2),
    torch.bfloat16: dict(atol=2e-2, rtol=2e-2),
}

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _eligible(op):
    return [
        b for b in op.available_backends() if b not in _SKIP and op.backend_eligible(b)
    ]


def _close(got, ref, dtype, msg):
    torch.testing.assert_close(got, ref, **_TOL[dtype], msg=msg)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1, 8192), (128, 8192)])
def test_silu_and_mul(dtype, shape):
    from sglang.kernels.ops.activation import _SILU_AND_MUL as op

    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype, device="cuda")
    ref = op.forward_native(x)
    for b in _eligible(op):
        _close(op.forward(x, backend=b), ref, dtype, f"{op.op} {b.value}")


def _exact(got, ref, msg):
    torch.testing.assert_close(got, ref, atol=0, rtol=0, msg=msg)


@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_beam_attention_gate(dtype, with_bias):
    """Gate + bias, its sigmoid and the product round like eager, bit for bit."""
    from sglang.kernels.ops.elementwise.beam_fused import BeamAttentionGateOp

    op = BeamAttentionGateOp()
    # 256 consecutive values per sign around 7/1024, where sigmoid lands next to
    # 0.5 and a non-eager sigmoid (approximate exp/div, Inductor's) moves FP16.
    center = torch.tensor(7 / 1024, dtype=dtype).view(torch.int16)
    gate = (center + torch.arange(-128, 128, dtype=torch.int16)).view(dtype)
    gate = torch.cat([gate, -gate]).repeat(2, 1).cuda()
    generator = torch.Generator().manual_seed(5)
    # A unit row exposes the sigmoid's rounding; a random row checks the product.
    attention = torch.stack([torch.ones(512), torch.randn(512, generator=generator)])
    bias = torch.randn(4, generator=generator) if with_bias else torch.zeros(4)
    attention, bias = attention.to("cuda", dtype), bias.to("cuda", dtype)
    expected = op.forward_native(attention, gate, bias, 128)
    for b in _eligible(op):
        _exact(
            op(attention, gate, bias, 128, backend=b), expected, f"{op.op} {b.value}"
        )


def _bf16_ulps(got, ref):
    """Largest distance in representable BF16 steps, counted across zero."""

    def ordered(t):
        bits = t.view(torch.int16).int()
        return torch.where(bits < 0, -(bits & 0x7FFF), bits)

    return (ordered(got) - ordered(ref)).abs().max().item()


@pytest.mark.parametrize("with_next", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_beam_sandwich_residual_rmsnorm(dtype, with_next):
    """BF16 stays within one ULP of eager; FP16 cancellation is bounded, not one ULP."""
    from sglang.kernels.ops.layernorm.beam_fused import (
        BeamSandwichResidualRMSNormOp,
        _rmsnorm_native,
    )

    op = BeamSandwichResidualRMSNormOp()
    # The recorded 64-row fixture draws the gate inputs first; keeping those draws
    # keeps its FP16 residual-cancellation points.
    torch.manual_seed(64)
    torch.randn(64, 2 * 5120 + 2048, device="cuda", dtype=dtype)
    torch.randn(40, device="cuda", dtype=dtype)
    x = torch.randn(64, 5120, device="cuda", dtype=dtype)
    r = torch.randn_like(x)
    w1 = torch.randn(5120, device="cuda", dtype=dtype)
    w2 = torch.randn_like(w1)
    args = (x, r, w1, w2 if with_next else None, 1e-6, (0.3, 0.9))
    expected_out, expected_combined = op.forward_native(*args)
    for b in _eligible(op):
        out, combined = op(*args, backend=b)
        msg = f"{op.op} {b.value}"
        if not with_next:
            assert out is None, msg
        if dtype == torch.bfloat16:
            for got, ref in ((combined, expected_combined), (out, expected_out)):
                if ref is None:
                    continue
                torch.testing.assert_close(got, ref, atol=0, rtol=2**-7, msg=msg)
                assert _bf16_ulps(got, ref) <= 1, msg
        else:
            fp16_tol = dict(atol=2**-11, rtol=2**-10, msg=msg)
            torch.testing.assert_close(combined, expected_combined, **fp16_tol)
            if with_next:
                # The next norm reads the combined stream, so bound it on the
                # kernel's own combined output rather than eager's.
                ref_out = _rmsnorm_native(combined, w2, 1e-6)
                torch.testing.assert_close(out, ref_out, **fp16_tol)


def test_beam_residual_cancellation():
    """Each scaled term rounds before the sum, so exactly opposite terms cancel."""
    from sglang.kernels.ops.layernorm.beam_fused import BeamSandwichResidualRMSNormOp

    op = BeamSandwichResidualRMSNormOp()
    branch = torch.ones(1, 5120, device="cuda", dtype=torch.bfloat16)
    post = torch.full((5120,), 5, device="cuda", dtype=branch.dtype)
    scales = (1 / math.sqrt(26), 5 / math.sqrt(26))
    args = (branch, -branch, post, torch.ones_like(post), 1e-6, scales)
    expected = op.forward_native(*args)
    assert torch.count_nonzero(expected[1]).item() == 0
    for b in _eligible(op):
        _exact(op(*args, backend=b), expected, f"{op.op} {b.value}")


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
