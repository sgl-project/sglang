import itertools

import pytest
import torch

from sglang.kernels.ops.elementwise.elementwise import fused_sigmoid_mul

DTYPES = [torch.float16, torch.bfloat16]
TOKEN_COUNTS = [1, 2, 4, 8, 16, 64, 512, 1024, 2048, 4096, 8192]
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
    """Test with 3D contiguous tensors (num_tokens, num_heads, head_dim)."""
    rtol, atol = (2e-2, 2e-2) if dtype == torch.bfloat16 else (1e-2, 1e-2)
    head_dim = 128

    attn_output = torch.randn(
        num_tokens, num_heads, head_dim, dtype=dtype, device="cuda"
    )
    gate = torch.randn(num_tokens, num_heads, head_dim, dtype=dtype, device="cuda")

    ref = _reference(attn_output, gate)
    out = fused_sigmoid_mul(attn_output, gate)

    torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "num_tokens, num_heads, dtype",
    list(itertools.product(TOKEN_COUNTS, NUM_HEADS, DTYPES)),
)
def test_strided_gate(num_tokens, num_heads, dtype):
    """Test strided gate path: attn_output is 2D, gate is 3D non-contiguous from chunk."""
    rtol, atol = (2e-2, 2e-2) if dtype == torch.bfloat16 else (1e-2, 1e-2)
    head_dim = 128
    hidden_dim = num_heads * head_dim

    # Simulate the real pattern: chunk produces non-contiguous views
    q_gate = torch.randn(
        num_tokens, num_heads, 2 * head_dim, dtype=dtype, device="cuda"
    )
    _, gate = torch.chunk(q_gate, 2, dim=-1)
    # gate is non-contiguous when num_tokens > 1 or num_heads > 1

    attn_output = torch.randn(num_tokens, hidden_dim, dtype=dtype, device="cuda")
    gate_flat = gate.reshape(num_tokens, hidden_dim)

    ref = _reference(attn_output, gate_flat)
    out = fused_sigmoid_mul(attn_output, gate, inplace=False)

    torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "num_tokens, dtype", list(itertools.product(TOKEN_COUNTS, DTYPES))
)
def test_qwen3_5_moe_target_strided_gate(num_tokens, dtype):
    """Qwen3.5 MoE target config: 32 attention heads, head_dim 256."""
    rtol, atol = (2e-2, 2e-2) if dtype == torch.bfloat16 else (1e-2, 1e-2)
    num_heads, head_dim = 32, 256
    hidden_dim = num_heads * head_dim

    q_gate = torch.randn(
        num_tokens, num_heads, 2 * head_dim, dtype=dtype, device="cuda"
    )
    _, gate = torch.chunk(q_gate, 2, dim=-1)
    attn_output = torch.randn(num_tokens, hidden_dim, dtype=dtype, device="cuda")

    ref = _reference(attn_output, gate.reshape(num_tokens, hidden_dim))
    out = fused_sigmoid_mul(attn_output, gate, inplace=False)

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
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
