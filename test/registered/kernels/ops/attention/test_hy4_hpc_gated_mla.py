import sys

import pytest
import torch
from torch import nn

from sglang.kernels.jit.utils import get_ci_test_range
from sglang.srt.models.hunyuan_v4 import HYV4Attention
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=60,
    nightly=False,
    disabled=None,
    stage="base-b-kernel-unit",
    runner_config="4-gpu-b200",
)
register_cuda_ci(
    est_time=120,
    nightly=True,
    disabled=None,
    stage="nightly",
    runner_config="4-gpu-b200",
)

# Guards attention-TP shards from silently reverting to the global N=16384 shape.
LOCAL_GATE_WIDTHS = get_ci_test_range(
    full_range=[256, 512, 1024, 2048, 4096, 8192, 16384],
    ci_range=[256, 2048, 16384],
)


def _hpc_gated_mla_available():
    try:
        from hpc.gemm import gated_mla_gemm  # noqa: F401
    except (AttributeError, ImportError):
        return False
    return torch.cuda.is_available() and torch.cuda.get_device_capability() in (
        (10, 0),
        (10, 3),
    )


class TupleLinear(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)

    def forward(self, inputs):
        return nn.functional.linear(inputs, self.weight), None


def _make_attention(weight):
    attention = HYV4Attention.__new__(HYV4Attention)
    nn.Module.__init__(attention)
    attention.linear_gate = TupleLinear(weight)
    attention.local_gate_width = weight.shape[0]
    attention._gate_backend = "hpc"
    attention._gate_fallback_backend = "eager"
    return attention


@pytest.mark.skipif(
    not _hpc_gated_mla_available(),
    reason="requires HPC-Ops gated MLA on SM100 or SM103",
)
@pytest.mark.parametrize("local_gate_width", LOCAL_GATE_WIDTHS)
def test_hy4_gated_mla_attention_tp_eager_graph_parity(local_gate_width):
    torch.manual_seed(local_gate_width)
    hidden_size = 6144
    batch_size = 7
    weight = torch.randn(
        local_gate_width, hidden_size, dtype=torch.bfloat16, device="cuda"
    )
    hidden_states = torch.randn(
        batch_size, hidden_size, dtype=torch.bfloat16, device="cuda"
    )
    attn_out = torch.randn(
        batch_size, local_gate_width, dtype=torch.bfloat16, device="cuda"
    )
    attention = _make_attention(weight)

    def run():
        gate = attention.prepare_attention_output_gate(hidden_states)
        return attention.apply_attention_output_gate(attn_out, gate)

    expected = attn_out * torch.sigmoid(nn.functional.linear(hidden_states, weight))
    eager = run()
    torch.testing.assert_close(eager, expected, rtol=0.08, atol=0.01)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = run()

    graph.replay()
    torch.testing.assert_close(graph_out, eager, rtol=0, atol=0)


@pytest.mark.skipif(
    not _hpc_gated_mla_available(),
    reason="requires HPC-Ops gated MLA on SM100 or SM103",
)
@pytest.mark.parametrize("batch_size", [128, 129])
def test_hy4_gated_mla_dispatch_boundary_graph_parity(batch_size):
    torch.manual_seed(batch_size)
    hidden_size = 6144
    local_gate_width = 256
    weight = torch.randn(
        local_gate_width, hidden_size, dtype=torch.bfloat16, device="cuda"
    )
    hidden_states = torch.randn(
        batch_size, hidden_size, dtype=torch.bfloat16, device="cuda"
    )
    attn_out = torch.randn(
        batch_size, local_gate_width, dtype=torch.bfloat16, device="cuda"
    )
    attention = _make_attention(weight)

    def run():
        gate = attention.prepare_attention_output_gate(hidden_states)
        return attention.apply_attention_output_gate(attn_out, gate)

    eager = run()
    expected = attn_out * torch.sigmoid(nn.functional.linear(hidden_states, weight))
    torch.testing.assert_close(eager, expected, rtol=0.08, atol=0.01)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = run()

    graph.replay()
    torch.testing.assert_close(graph_out, eager, rtol=0, atol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
