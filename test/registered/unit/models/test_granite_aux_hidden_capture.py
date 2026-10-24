"""Unit tests for aux hidden-state capture in the Granite and GraniteMoe targets.

EAGLE3 / DFLASH / DSPARK drafters read target hidden states at chosen layers.
`set_dflash_layers_to_capture(ids)` captures the state after target layer
`ids[i]`, which SGLang records before layer `ids[i] + 1`. Each capture must be
a snapshot: Granite layers and norms update their inputs in place, so a
captured alias would be silently overwritten by later layers.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from sglang.srt.models.granite import GraniteForCausalLM, GraniteModel
from sglang.srt.models.granitemoe import GraniteMoeForCausalLM, GraniteMoeModel
from sglang.test.test_utils import CustomTestCase

NUM_LAYERS = 4
MULTIPLIER = 2.0


class _ResidualLayer(nn.Module):
    """GraniteDecoderLayer contract: returns (delta, residual), with the stream
    equal to delta + residual. Like fused add-RMSNorm, it adds into the
    residual in place, and at layer 0 the residual aliases the embeddings."""

    def __init__(self, index):
        super().__init__()
        self.index = index

    def forward(self, positions, hidden_states, forward_batch, residual):
        if residual is None:
            residual = hidden_states
        else:
            residual.add_(hidden_states)
        return torch.full_like(hidden_states, float(self.index + 1)), residual


class _ResidualNorm(nn.Module):
    """Fused add-RMSNorm stand-in that clobbers its inputs."""

    def forward(self, hidden_states, residual):
        out = hidden_states + residual
        hidden_states.fill_(-101.0)
        residual.fill_(-103.0)
        return out, residual


class _StreamLayer(nn.Module):
    """GraniteMoeDecoderLayer contract: returns the full stream. It overwrites
    its input afterwards, as routed experts may consume their input buffer."""

    def __init__(self, index):
        super().__init__()
        self.index = index

    def forward(self, positions, hidden_states, forward_batch):
        out = hidden_states + float(self.index + 1)
        hidden_states.fill_(-7.0)
        return out


class _StreamNorm(nn.Module):
    def forward(self, hidden_states):
        out = hidden_states.clone()
        hidden_states.fill_(-9.0)
        return out


def _expected_stream(embeds, k):
    return embeds * MULTIPLIER + k * (k + 1) / 2


def _granite_model(layers_to_capture):
    model = GraniteModel.__new__(GraniteModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(embedding_multiplier=MULTIPLIER)
    model.layers = nn.ModuleList([_ResidualLayer(i) for i in range(NUM_LAYERS)])
    model.norm = _ResidualNorm()
    model.layers_to_capture = list(layers_to_capture)
    return model


def _granitemoe_model(layers_to_capture):
    model = GraniteMoeModel.__new__(GraniteMoeModel)
    nn.Module.__init__(model)
    model.embedding_multiplier = MULTIPLIER
    model.layers = nn.ModuleList([_StreamLayer(i) for i in range(NUM_LAYERS)])
    model.norm = _StreamNorm()
    model.layers_to_capture = list(layers_to_capture)
    return model


def _run_granite(model, embeds):
    return model(None, None, None, input_embeds=embeds.clone())


def _run_granitemoe(model, embeds):
    return model(None, None, None, inputs_embeds=embeds.clone())


class TestGraniteModelAuxCapture(CustomTestCase):
    build = staticmethod(_granite_model)
    run_forward = staticmethod(_run_granite)

    def setUp(self):
        self.embeds = torch.arange(6, dtype=torch.float32).reshape(2, 3)

    def test_no_capture_returns_hidden_states_only(self):
        out = self.run_forward(self.build([]), self.embeds)
        self.assertIsInstance(out, torch.Tensor)
        torch.testing.assert_close(out, _expected_stream(self.embeds, NUM_LAYERS))

    def test_captures_stream_before_each_requested_layer(self):
        layers = [0, 2, NUM_LAYERS]
        out, aux = self.run_forward(self.build(layers), self.embeds)
        self.assertEqual(len(aux), len(layers))
        for captured, k in zip(aux, layers):
            torch.testing.assert_close(captured, _expected_stream(self.embeds, k))
        torch.testing.assert_close(out, _expected_stream(self.embeds, NUM_LAYERS))

    def test_captures_do_not_alias_live_buffers(self):
        _, aux = self.run_forward(self.build(range(NUM_LAYERS + 1)), self.embeds)
        for k, captured in enumerate(aux):
            torch.testing.assert_close(captured, _expected_stream(self.embeds, k))


class TestGraniteMoeModelAuxCapture(TestGraniteModelAuxCapture):
    build = staticmethod(_granitemoe_model)
    run_forward = staticmethod(_run_granitemoe)


def _causal_lm(cls, num_layers=40):
    lm = cls.__new__(cls)
    nn.Module.__init__(lm)
    lm.config = SimpleNamespace(num_hidden_layers=num_layers)
    lm.model = SimpleNamespace(layers_to_capture=[], embed_tokens=object())
    lm.capture_aux_hidden_states = False
    return lm


class TestGraniteForCausalLMCaptureHooks(CustomTestCase):
    cls = GraniteForCausalLM

    def test_dflash_layer_ids_map_to_the_next_layer(self):
        lm = _causal_lm(self.cls)
        lm.set_dflash_layers_to_capture([1, 14, 29, 44, 59])
        self.assertTrue(lm.capture_aux_hidden_states)
        self.assertEqual(lm.model.layers_to_capture, [2, 15, 30, 45, 60])

    def test_dflash_requires_explicit_layer_ids(self):
        with self.assertRaises(ValueError):
            _causal_lm(self.cls).set_dflash_layers_to_capture(None)

    def test_eagle3_default_and_explicit_layers(self):
        lm = _causal_lm(self.cls, num_layers=40)
        lm.set_eagle3_layers_to_capture()
        self.assertTrue(lm.capture_aux_hidden_states)
        self.assertEqual(lm.model.layers_to_capture, [2, 20, 37])
        lm.set_eagle3_layers_to_capture([1, 5])
        self.assertEqual(lm.model.layers_to_capture, [2, 6])

    def test_input_embeddings_are_the_embedding_module(self):
        lm = _causal_lm(self.cls)
        self.assertIs(lm.get_input_embeddings(), lm.model.embed_tokens)

    def test_forward_passes_aux_hidden_states_to_logits_processor(self):
        lm = _causal_lm(self.cls)
        hidden, aux = torch.zeros(2, 3), [torch.ones(2, 3)]
        lm.model = lambda *args: (hidden, aux)
        lm.lm_head = object()
        calls = []
        lm.logits_processor = lambda *args: calls.append(args) or "logits"
        lm.capture_aux_hidden_states = True
        self.assertEqual(lm.forward(None, None, None), "logits")
        self.assertIs(calls[0][1], hidden)
        self.assertIs(calls[0][4], aux)


class TestGraniteMoeForCausalLMCaptureHooks(TestGraniteForCausalLMCaptureHooks):
    cls = GraniteMoeForCausalLM


if __name__ == "__main__":
    unittest.main()
