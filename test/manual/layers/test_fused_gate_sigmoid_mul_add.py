import itertools

import pytest
import torch

from sglang.kernels.ops.elementwise.elementwise import fused_gate_sigmoid_mul_add

DTYPES = [torch.float16, torch.bfloat16]
TOKEN_COUNTS = [1, 2, 4, 8, 16, 64, 512, 1024, 2048, 4096, 8192]
HIDDEN_DIMS = [2048, 3072, 4096, 6144]


def _reference(hidden_states, gate_weight, shared_output, final_hidden_states):
    gate = hidden_states @ gate_weight
    final_hidden_states += torch.sigmoid(gate).unsqueeze(1) * shared_output


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(42)


@pytest.mark.parametrize(
    "num_tokens, hidden_dim, dtype",
    list(itertools.product(TOKEN_COUNTS, HIDDEN_DIMS, DTYPES)),
)
def test_correctness(num_tokens, hidden_dim, dtype):
    rtol, atol = (2e-2, 2e-2) if dtype == torch.bfloat16 else (1e-2, 1e-2)

    hidden_states = torch.randn(num_tokens, hidden_dim, dtype=dtype, device="cuda")
    gate_weight = torch.randn(hidden_dim, dtype=dtype, device="cuda")
    shared_output = torch.randn(num_tokens, hidden_dim, dtype=dtype, device="cuda")
    final_ref = torch.randn(num_tokens, hidden_dim, dtype=dtype, device="cuda")
    final_test = final_ref.clone()

    _reference(hidden_states, gate_weight, shared_output, final_ref)
    fused_gate_sigmoid_mul_add(hidden_states, gate_weight, shared_output, final_test)

    torch.testing.assert_close(final_test, final_ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", DTYPES)
def test_gate_near_zero(dtype):
    num_tokens, hidden_dim = 16, 2048
    hs = torch.randn(num_tokens, hidden_dim, dtype=dtype, device="cuda")
    gw = torch.zeros(hidden_dim, dtype=dtype, device="cuda")
    so = torch.randn(num_tokens, hidden_dim, dtype=dtype, device="cuda")
    f_ref = torch.randn(num_tokens, hidden_dim, dtype=dtype, device="cuda")
    f_test = f_ref.clone()

    _reference(hs, gw, so, f_ref)
    fused_gate_sigmoid_mul_add(hs, gw, so, f_test)

    torch.testing.assert_close(f_test, f_ref, rtol=1e-2, atol=1e-2)


def test_inplace_semantics():
    num_tokens, hidden_dim = 32, 2048
    hs = torch.randn(num_tokens, hidden_dim, dtype=torch.float16, device="cuda")
    gw = torch.randn(hidden_dim, dtype=torch.float16, device="cuda")
    so = torch.randn(num_tokens, hidden_dim, dtype=torch.float16, device="cuda")
    fhs = torch.randn(num_tokens, hidden_dim, dtype=torch.float16, device="cuda")
    original_ptr = fhs.data_ptr()

    fused_gate_sigmoid_mul_add(hs, gw, so, fhs)

    assert fhs.data_ptr() == original_ptr


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
