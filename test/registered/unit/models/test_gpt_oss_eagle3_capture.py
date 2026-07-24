import unittest
from types import SimpleNamespace

import torch
from torch import nn

from sglang.srt.models.gpt_oss import GptOssForCausalLM, GptOssModel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _SingleRankPPGroup:
    is_first_rank = True
    is_last_rank = True


class _ZeroEmbedding(nn.Module):
    def forward(self, input_ids):
        return torch.zeros((input_ids.shape[0], 2), dtype=torch.float32)


class _IncrementLayer(nn.Module):
    def forward(self, positions, hidden_states, forward_batch, residual):
        if residual is None:
            residual = torch.zeros_like(hidden_states)
        return hidden_states + 1, residual


class _ResidualNorm(nn.Module):
    def forward(self, hidden_states, residual):
        return hidden_states + residual, None


class TestGptOssEagle3Capture(unittest.TestCase):
    def test_eagle3_layer_ids_are_hidden_state_boundaries(self):
        model = GptOssForCausalLM.__new__(GptOssForCausalLM)
        nn.Module.__init__(model)
        model.pp_group = _SingleRankPPGroup()
        model.config = SimpleNamespace(num_hidden_layers=36)
        model.model = SimpleNamespace(layers_to_capture=[])

        model.set_eagle3_layers_to_capture([24, 30, 36])

        self.assertTrue(model.capture_aux_hidden_states)
        self.assertEqual(model.model.layers_to_capture, [24, 30, 36])

    def test_forward_captures_final_hidden_state_boundary(self):
        model = GptOssModel.__new__(GptOssModel)
        nn.Module.__init__(model)
        model.pp_group = _SingleRankPPGroup()
        model.embed_tokens = _ZeroEmbedding()
        model.layers = nn.ModuleList([_IncrementLayer() for _ in range(3)])
        model.start_layer = 0
        model.end_layer = 3
        model.layers_to_capture = [1, 3]
        model.norm = _ResidualNorm()

        hidden_states, aux_hidden_states = model(
            input_ids=torch.tensor([1]),
            positions=torch.tensor([0]),
            forward_batch=None,
        )

        self.assertEqual(len(aux_hidden_states), 2)
        torch.testing.assert_close(aux_hidden_states[0], torch.ones((1, 2)))
        torch.testing.assert_close(aux_hidden_states[1], torch.full((1, 2), 3.0))
        torch.testing.assert_close(hidden_states, torch.full((1, 2), 3.0))


if __name__ == "__main__":
    unittest.main()
