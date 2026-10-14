import itertools

import pytest
import torch

from sglang.kernels.ops.elementwise.elementwise import (
    fused_gate_sigmoid_mul,
    fused_gate_sigmoid_mul_add,
)
from sglang.kernels.ops.moe.shared_expert_gate import shared_expert_gate
from sglang.srt.utils import get_device

DTYPES = [torch.float16, torch.bfloat16]
TOKEN_COUNTS = [1, 2, 4, 8, 16, 64, 512, 1024, 2048, 4096, 8192]
HIDDEN_DIMS = [2048, 3072, 4096, 6144]
DEVICE = get_device()


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

    hidden_states = torch.randn(num_tokens, hidden_dim, dtype=dtype, device=DEVICE)
    gate_weight = torch.randn(hidden_dim, dtype=dtype, device=DEVICE)
    shared_output = torch.randn(num_tokens, hidden_dim, dtype=dtype, device=DEVICE)
    final_ref = torch.randn(num_tokens, hidden_dim, dtype=dtype, device=DEVICE)
    final_test = final_ref.clone()

    _reference(hidden_states, gate_weight, shared_output, final_ref)
    fused_gate_sigmoid_mul_add(hidden_states, gate_weight, shared_output, final_test)

    torch.testing.assert_close(final_test, final_ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", DTYPES)
def test_gate_near_zero(dtype):
    num_tokens, hidden_dim = 16, 2048
    hs = torch.randn(num_tokens, hidden_dim, dtype=dtype, device=DEVICE)
    gw = torch.zeros(hidden_dim, dtype=dtype, device=DEVICE)
    so = torch.randn(num_tokens, hidden_dim, dtype=dtype, device=DEVICE)
    f_ref = torch.randn(num_tokens, hidden_dim, dtype=dtype, device=DEVICE)
    f_test = f_ref.clone()

    _reference(hs, gw, so, f_ref)
    fused_gate_sigmoid_mul_add(hs, gw, so, f_test)

    torch.testing.assert_close(f_test, f_ref, rtol=1e-2, atol=1e-2)


def test_inplace_semantics():
    num_tokens, hidden_dim = 32, 2048
    hs = torch.randn(num_tokens, hidden_dim, dtype=torch.float16, device=DEVICE)
    gw = torch.randn(hidden_dim, dtype=torch.float16, device=DEVICE)
    so = torch.randn(num_tokens, hidden_dim, dtype=torch.float16, device=DEVICE)
    fhs = torch.randn(num_tokens, hidden_dim, dtype=torch.float16, device=DEVICE)
    original_ptr = fhs.data_ptr()

    fused_gate_sigmoid_mul_add(hs, gw, so, fhs)

    assert fhs.data_ptr() == original_ptr


@pytest.mark.parametrize("weight_scale", [0.0, 0.02, 1.0])
def test_shared_gate_fp32_and_changed_input_graph_replay(weight_scale):
    if torch.version.hip or torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("The standalone shared gate uses CUDA PDL (SM90+).")
    hidden = torch.randn(1, 2560, device=DEVICE, dtype=torch.bfloat16)
    weight = torch.randn(2560, device=DEVICE, dtype=torch.bfloat16) * weight_scale
    shared = torch.randn_like(hidden)
    routed = torch.randn_like(hidden)
    gate = torch.empty(1, device=DEVICE, dtype=torch.float32)
    combined = torch.empty_like(hidden)

    def run():
        shared_expert_gate(hidden, weight, out=gate)
        gated = fused_gate_sigmoid_mul(hidden, weight, shared)
        combined.copy_(routed)
        fused_gate_sigmoid_mul_add(hidden, weight, shared, combined)
        return gated

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gated = run()
    for _ in range(3):
        hidden.normal_()
        shared.normal_()
        routed.normal_()
        graph.replay()
        expected_gate = torch.sigmoid((hidden.float() * weight.float()).sum(-1))
        torch.testing.assert_close(gate, expected_gate, rtol=1e-5, atol=1e-6)
        # Both full-output variants consume the same FP32 gate without
        # rounding the gated shared branch before the final add.
        torch.testing.assert_close(
            gated, (gate[:, None] * shared.float()).to(hidden.dtype), rtol=0, atol=0
        )
        torch.testing.assert_close(
            combined,
            torch.addcmul(routed.float(), gate[:, None], shared.float()).to(
                hidden.dtype
            ),
            rtol=0,
            atol=0,
        )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
