"""An empty auxiliary collector must not change the compiled model's output."""

from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.models import qwen3
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_qwen3_compiled_output_without_aux_capture():
    class Layer(nn.Module):
        def forward(self, positions, hidden_states, forward_batch, capture_output=None):
            return hidden_states + 1

    model = qwen3.Qwen3Model.__new__(qwen3.Qwen3Model)
    nn.Module.__init__(model)
    model.pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
    model.embed_tokens = nn.Embedding(16, 8)
    model.layers = nn.ModuleList([Layer()])
    model.start_layer, model.end_layer = 0, 1
    model.layers_to_capture = []
    model.norm = nn.Identity()
    batch = SimpleNamespace()
    ids = torch.tensor([1, 2])
    positions = torch.tensor([0, 1])
    torch._dynamo.reset()
    try:
        with (
            patch.object(qwen3.residual_batch, "start", lambda _: None),
            patch.object(qwen3.residual_batch, "final_norm", lambda x, *args, **kw: x),
        ):
            expected = model(ids, positions, batch)
            compiled = torch.compile(model, backend="aot_eager", fullgraph=True)
            actual = compiled(ids, positions, batch)
            assert isinstance(actual, torch.Tensor)
            torch.testing.assert_close(actual, expected)
    finally:
        torch._dynamo.reset()
