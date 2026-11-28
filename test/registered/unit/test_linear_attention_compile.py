"""Recurrent state updates stay opaque while the surrounding model compiles."""

from types import SimpleNamespace
from unittest.mock import patch

import torch
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


def test_compiled_decode_preserves_backend_state_updates():
    layer = RadixLinearAttention(0, 1, 1, 1, 4, 4, 4)
    batch = SimpleNamespace(forward_mode=ForwardMode.DECODE)
    state = torch.zeros(1024, 4)
    graphs = []

    class Backend:
        def forward(self, *, mixed_qkv, **kwargs):
            assert not torch.compiler.is_compiling()
            state[17].add_(1)
            return mixed_qkv + state[17]

    def compile_graph(graph, inputs):
        graphs.append(graph)
        return graph.forward

    with patch(
        "sglang.srt.layers.radix_linear_attention.get_attn_backend",
        return_value=Backend(),
    ):
        forward = torch.compile(
            lambda x: layer(batch, x, x, x) * 2, backend=compile_graph
        )
        assert torch.equal(forward(torch.ones(1, 4)), torch.full((1, 4), 4.0))
        assert torch.equal(forward(torch.ones(1, 4)), torch.full((1, 4), 6.0))
    assert graphs
    assert torch.equal(state[17], torch.full((4,), 2.0))
    assert state.sum().item() == 8
