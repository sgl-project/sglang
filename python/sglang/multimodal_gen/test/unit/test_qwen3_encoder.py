from types import SimpleNamespace

import torch

from sglang.multimodal_gen.runtime.models.encoders.qwen3 import Qwen3ForCausalLM


class CaptureLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.position_ids = None
        self.attention_lengths = None

    def forward(self, position_ids, hidden_states, residual, attention_lengths):
        self.position_ids = position_ids
        self.attention_lengths = attention_lengths
        if residual is None:
            residual = torch.zeros_like(hidden_states)
        return hidden_states, residual


class IdentityNorm(torch.nn.Module):
    def forward(self, hidden_states, residual):
        if residual is not None:
            hidden_states = hidden_states + residual
        return hidden_states, None


def test_default_position_ids_match_batch_size():
    model = Qwen3ForCausalLM.__new__(Qwen3ForCausalLM)
    torch.nn.Module.__init__(model)
    layer = CaptureLayer()
    model.config = SimpleNamespace(output_hidden_states=False)
    model.layers = torch.nn.ModuleList([layer])
    model.norm = IdentityNorm()

    def get_input_embeddings(input_ids):
        return torch.zeros(input_ids.shape[0], input_ids.shape[1], 8)

    model.get_input_embeddings = get_input_embeddings

    input_ids = torch.zeros(2, 4, dtype=torch.long)
    attention_mask = torch.ones(2, 4, dtype=torch.long)

    model(input_ids=input_ids, attention_mask=attention_mask)

    assert layer.position_ids.shape == input_ids.shape
    assert torch.equal(layer.position_ids[0], torch.arange(4))
    assert torch.equal(layer.position_ids[1], torch.arange(4))
    assert layer.attention_lengths == (4, 4)
