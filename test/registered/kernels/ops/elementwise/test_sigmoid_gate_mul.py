import sys

import pytest
import torch

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=4, stage="base-b-kernel-unit", runner_config="1-gpu-small")
register_amd_ci(est_time=4, stage="jit-kernel-unit", runner_config="amd")

DEVICE = "cuda"


def reference_sigmoid_gate_mul(x, gate):
    return x * torch.sigmoid(gate)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize(
    "shape",
    [
        (1, 4096),
        (4, 4096),
        (8, 4096),
        (32, 8192),
        (1, 128),
        (16, 16384),
    ],
)
def test_sigmoid_gate_mul_correctness(shape, dtype):
    from sglang.kernels.ops.elementwise.elementwise import fused_sigmoid_mul

    torch.manual_seed(42)
    x = torch.randn(shape, dtype=dtype, device=DEVICE)
    gate = torch.randn(shape, dtype=dtype, device=DEVICE)

    ref = reference_sigmoid_gate_mul(x, gate)
    out = fused_sigmoid_mul(x, gate)

    rtol = 1e-2 if dtype == torch.bfloat16 else 1e-3
    atol = 2e-2 if dtype == torch.bfloat16 else 1e-3
    torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("shape", [(4, 4096), (1, 128)])
def test_sigmoid_gate_mul_does_not_modify_inputs(shape):
    from sglang.kernels.ops.elementwise.elementwise import fused_sigmoid_mul

    torch.manual_seed(42)
    x = torch.randn(shape, dtype=torch.bfloat16, device=DEVICE)
    gate = torch.randn(shape, dtype=torch.bfloat16, device=DEVICE)
    x_orig = x.clone()
    gate_orig = gate.clone()

    fused_sigmoid_mul(x, gate)

    torch.testing.assert_close(x, x_orig, rtol=0, atol=0)
    torch.testing.assert_close(gate, gate_orig, rtol=0, atol=0)


def test_sigmoid_gate_mul_output_dtype():
    from sglang.kernels.ops.elementwise.elementwise import fused_sigmoid_mul

    for dtype in [torch.bfloat16, torch.float16, torch.float32]:
        x = torch.randn(4, 4096, dtype=dtype, device=DEVICE)
        gate = torch.randn(4, 4096, dtype=dtype, device=DEVICE)
        out = fused_sigmoid_mul(x, gate)
        assert out.dtype == dtype, f"Expected {dtype}, got {out.dtype}"


def test_sigmoid_gate_mul_contiguous_output():
    from sglang.kernels.ops.elementwise.elementwise import fused_sigmoid_mul

    x = torch.randn(4, 4096, dtype=torch.bfloat16, device=DEVICE)
    gate = torch.randn(4, 4096, dtype=torch.bfloat16, device=DEVICE)
    out = fused_sigmoid_mul(x, gate)
    assert out.is_contiguous()


def test_sigmoid_gate_mul_strided_gate_round_sigmoid_is_bit_exact():
    num_tokens = 300
    # The gate is the tail columns of a merged [q_up | gate] GEMM output.
    from sglang.kernels.ops.elementwise.elementwise import fused_sigmoid_mul

    torch.manual_seed(0)
    x = torch.randn(num_tokens, 8192, dtype=torch.bfloat16, device=DEVICE)
    merged = torch.randn(num_tokens, 4096 + 8192, dtype=torch.bfloat16, device=DEVICE)
    gate = merged[:, 4096:]
    ref = x * torch.sigmoid(gate)
    assert torch.equal(fused_sigmoid_mul(x, gate, round_sigmoid=True), ref)
    assert torch.equal(fused_sigmoid_mul(x, gate.contiguous(), round_sigmoid=True), ref)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
