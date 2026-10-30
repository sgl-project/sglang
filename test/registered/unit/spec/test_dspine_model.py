# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0.

import unittest
from test.registered.unit.spec.test_dspine import bridge_config, make_bridge

import torch
from torch import nn

from sglang.srt.models.dspine import (
    DSpineDecoderLayer,
    DSpineDraftModel,
    DSpineSelector,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class ReferenceNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(8))
        self.variance_epsilon = 1e-6
        self.calls = 0

    def forward(self, hidden, residual=None):
        self.calls += 1
        state = hidden if residual is None else hidden + residual
        result = (
            state
            * torch.rsqrt(state.square().mean(-1, keepdim=True) + self.variance_epsilon)
            * self.weight
        )
        return result if residual is None else (result, state)


class ReferenceAttention(nn.Module):
    def forward(self, *, hidden_states, **kwargs):
        return hidden_states * 0.125


def draft():
    # Isolate device-specific construction, not the production forward/scoring.
    model = DSpineDraftModel.__new__(DSpineDraftModel)
    nn.Module.__init__(model)
    model.config = bridge_config()
    model.block_size = 4
    layers = []
    for _ in range(3):
        layer = DSpineDecoderLayer.__new__(DSpineDecoderLayer)
        nn.Module.__init__(layer)
        layer.input_layernorm = ReferenceNorm()
        layer.post_attention_layernorm = ReferenceNorm()
        layer.self_attn = ReferenceAttention()
        layer.mlp = nn.Linear(8, 8, bias=False)
        layers.append(layer)
    model.layers = nn.ModuleList(layers)
    model.norm = ReferenceNorm()
    model.selected_prefix_bridge = make_bridge()
    model.last_write_stash = None
    model.lm_head = None
    model._refinement_ready = False
    model.candidate_selector = DSpineSelector(model)
    head = nn.Linear(8, 13, bias=False)
    model.prepare_refinement(head)
    return model


class TestDSpineModel(unittest.TestCase):
    def test_fused_next_norm_is_not_applied_twice(self):
        model = draft()
        ids = torch.tensor([2, 12, 12, 12, 7, 12, 12, 12])
        output = model.forward(
            ids, torch.arange(8), None, input_embeds=torch.randn(8, 8)
        )
        self.assertEqual(output.hidden_states.shape, (8, 8))
        self.assertEqual(
            [layer.input_layernorm.calls for layer in model.layers], [1, 0, 0]
        )
        self.assertEqual(model.norm.calls, 0)
        self.assertEqual(model.last_write_stash[0].shape, (8, 8))
        self.assertEqual(model.last_write_stash[1].shape, (8,))
        self.assertEqual(model.last_write_stash[2].shape, (8, 4))

    def test_refinement_uses_current_batch_and_rejects_stale_geometry(self):
        model = draft()
        for batch in (2, 1, 3):
            ids = torch.full((batch * 4,), 12)
            model.forward(
                ids,
                torch.arange(batch * 4),
                None,
                input_embeds=torch.randn(batch * 4, 8),
            )
            candidates = torch.arange(4).expand(batch, 3, 4)
            scores = model.candidate_selector.build_lattice(
                candidate_ids=candidates,
                unary_logits=torch.zeros_like(candidates),
                hidden_states=torch.empty(batch, 3, 8),
                anchor_token_ids=ids.view(batch, 4)[:, 0],
            )
            self.assertEqual(scores.shape, (batch, 3, 4, 4))
            self.assertTrue(torch.isfinite(scores).all())
        with self.assertRaisesRegex(ValueError, "current request blocks"):
            model.candidate_selector.build_lattice(
                candidate_ids=torch.zeros(1, 3, 4, dtype=torch.long),
                unary_logits=None,
                hidden_states=None,
                anchor_token_ids=torch.zeros(1, dtype=torch.long),
            )

    def test_refinement_reload_keeps_graph_buffer_addresses(self):
        model = draft()
        names = ("pred_gate_f32", "prev_f32", "write_f32", "norm_f32", "write_gram_f32")
        pointers = {name: getattr(model, name).data_ptr() for name in names}
        previous = model.write_gram_f32.clone()
        model.selected_prefix_bridge.feature_write_weights.data.add_(0.1)
        model.post_load_weights()
        self.assertFalse(torch.equal(previous, model.write_gram_f32))
        self.assertEqual(
            pointers, {name: getattr(model, name).data_ptr() for name in names}
        )

    def test_shared_sampler_reports_the_realized_conditional_proposal(self):
        model = draft()
        torch.manual_seed(19)
        ids = torch.full((8,), 12)
        model.forward(ids, torch.arange(8), None, input_embeds=torch.randn(8, 8))
        candidates = torch.tensor([1, 3, 5, 7]).expand(2, 3, 4)
        scores = model.candidate_selector.build_lattice(
            candidate_ids=candidates,
            unary_logits=None,
            hidden_states=None,
            anchor_token_ids=ids.view(2, 4)[:, 0],
        )
        temperatures = torch.tensor([0.8, 1.0])
        greedy = torch.tensor([False, True])
        tokens, probabilities = model.candidate_selector.sample_path(
            candidate_ids=candidates,
            scores=scores,
            uniforms=torch.tensor([[0.1, 0.7, 0.4], [0.5, 0.5, 0.5]]),
            temperatures=temperatures,
            greedy_mask=greedy,
        )
        for b in range(2):
            previous = 0
            for step in range(3):
                index = int((candidates[b, step] == tokens[b, step]).nonzero()[0])
                if greedy[b]:
                    expected = torch.nn.functional.one_hot(
                        torch.tensor(index), 4
                    ).float()
                else:
                    expected = torch.softmax(
                        scores[b, step, previous] / temperatures[b], dim=-1
                    )
                torch.testing.assert_close(probabilities[b, step], expected)
                previous = index

    def test_partial_blocks_are_rejected(self):
        model = draft()
        with self.assertRaisesRegex(ValueError, "complete trained request blocks"):
            model.forward(
                torch.zeros(3, dtype=torch.long),
                torch.arange(3),
                None,
                input_embeds=torch.randn(3, 8),
            )


if __name__ == "__main__":
    unittest.main()
