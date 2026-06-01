from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

import itertools

import pytest
import torch

from sglang.srt.layers.elementwise import fused_sigmoid_mul

DTYPES = [torch.float16, torch.bfloat16]
TOKEN_COUNTS = [1, 4, 16, 64, 512, 1024, 2048, 8192]
HIDDEN_DIMS = [2048, 3072, 4096, 6144]
NUM_HEADS = [1, 28]


def _reference(attn_output, gate):
    return attn_output * torch.sigmoid(gate)


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(42)


@pytest.mark.parametrize(
    "num_tokens, hidden_dim, dtype",
    list(itertools.product(TOKEN_COUNTS, HIDDEN_DIMS, DTYPES)),
)
def test_correctness(num_tokens, hidden_dim, dtype):
    rtol, atol = (2e-2, 2e-2) if dtype == torch.bfloat16 else (1e-2, 1e-2)

    attn_output = torch.randn(num_tokens, hidden_dim, dtype=dtype, device="cuda")
    gate = torch.randn(num_tokens, hidden_dim, dtype=dtype, device="cuda")

    ref = _reference(attn_output, gate)
    out = fused_sigmoid_mul(attn_output, gate)

    torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "num_tokens, num_heads, dtype",
    list(itertools.product(TOKEN_COUNTS, NUM_HEADS, DTYPES)),
)
def test_3d_shape(num_tokens, num_heads, dtype):
    """Test with 3D tensors (num_tokens, num_heads, head_dim) as used in attention."""
    rtol, atol = (2e-2, 2e-2) if dtype == torch.bfloat16 else (1e-2, 1e-2)
    head_dim = 128

    attn_output = torch.randn(
        num_tokens, num_heads, head_dim, dtype=dtype, device="cuda"
    )
    gate = torch.randn(num_tokens, num_heads, head_dim, dtype=dtype, device="cuda")

    ref = _reference(attn_output, gate)
    out = fused_sigmoid_mul(attn_output, gate)

    torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", DTYPES)
def test_gate_near_zero(dtype):
    num_tokens, hidden_dim = 16, 2048
    attn_output = torch.randn(num_tokens, hidden_dim, dtype=dtype, device="cuda")
    gate = torch.zeros(num_tokens, hidden_dim, dtype=dtype, device="cuda")

    ref = _reference(attn_output, gate)
    out = fused_sigmoid_mul(attn_output, gate)

    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


def test_returns_new_tensor():
    num_tokens, hidden_dim = 32, 2048
    attn_output = torch.randn(
        num_tokens, hidden_dim, dtype=torch.float16, device="cuda"
    )
    gate = torch.randn(num_tokens, hidden_dim, dtype=torch.float16, device="cuda")

    out = fused_sigmoid_mul(attn_output, gate)

    assert out.data_ptr() != attn_output.data_ptr()
    assert out.data_ptr() != gate.data_ptr()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
