"""EAGLE3 aux hidden-state capture through the MiniCPM target forward."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.srt.models.minicpm import MiniCPMModel, MiniCPMSALAForCausalLM
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_NUM_LAYERS = 8
_HIDDEN = 4


def _bare_causal_lm(*, capture: bool) -> MiniCPMSALAForCausalLM:
    lm = MiniCPMSALAForCausalLM.__new__(MiniCPMSALAForCausalLM)
    lm.config = SimpleNamespace(
        scale_emb=1.0,
        tie_word_embeddings=True,
        num_hidden_layers=_NUM_LAYERS,
    )
    lm.model = Mock()
    lm.scale_width = 4.0
    lm.logits_processor = Mock(return_value="logits")
    lm.capture_aux_hidden_states = capture
    return lm


class _RecordingLayer:
    def __init__(self, delta: float):
        self.delta = delta

    def __call__(self, positions, hidden_states, forward_batch, residual):
        return hidden_states + 2 * self.delta, None


def _bare_model(layers_to_capture) -> MiniCPMModel:
    model = MiniCPMModel.__new__(MiniCPMModel)
    model.config = SimpleNamespace(scale_emb=1.0)
    model.embed_tokens = lambda input_ids: torch.zeros(
        input_ids.shape[0], _HIDDEN, dtype=torch.float32
    )
    model.layers = [_RecordingLayer(delta=float(i + 1)) for i in range(_NUM_LAYERS)]
    model.norm = lambda hidden_states: hidden_states
    model.layers_to_capture = layers_to_capture
    return model


class TestEagle3LayerMapping(CustomTestCase):
    def test_draft_layer_ids_shift_by_one(self):
        lm = _bare_causal_lm(capture=False)
        lm.set_eagle3_layers_to_capture([1, _NUM_LAYERS // 2, _NUM_LAYERS - 4])

        self.assertTrue(lm.capture_aux_hidden_states)
        self.assertEqual(
            lm.model.layers_to_capture,
            [2, _NUM_LAYERS // 2 + 1, _NUM_LAYERS - 3],
        )

    def test_default_layers_follow_eagle3_convention(self):
        lm = _bare_causal_lm(capture=False)
        lm.set_eagle3_layers_to_capture(None)

        self.assertEqual(
            lm.model.layers_to_capture,
            [2, _NUM_LAYERS // 2, _NUM_LAYERS - 3],
        )


class TestAuxCaptureForward(CustomTestCase):
    def test_capture_records_complete_layer_outputs(self):
        model = _bare_model(layers_to_capture=[1, 2])
        input_ids = torch.zeros(3, dtype=torch.int64)

        hidden_states, aux = model.forward(
            input_ids, positions=None, forward_batch=None
        )

        # Each layer adds two branches, so layer i's input is 2 * sum(1..i).
        self.assertEqual(len(aux), 2)
        self.assertTrue(torch.equal(aux[0], torch.full((3, _HIDDEN), 2.0)))
        self.assertTrue(torch.equal(aux[1], torch.full((3, _HIDDEN), 6.0)))
        # The final layer output is 2 * sum(1..8) = 72.
        self.assertTrue(torch.equal(hidden_states, torch.full((3, _HIDDEN), 72.0)))

    def test_no_capture_returns_plain_tensor(self):
        # Runners unpack a tuple only when capture_aux_hidden_states is set.
        model = _bare_model(layers_to_capture=[])

        result = model.forward(
            torch.zeros(3, dtype=torch.int64), positions=None, forward_batch=None
        )

        self.assertIsInstance(result, torch.Tensor)

    def test_last_layer_capture_precedes_final_normalization(self):
        """Selecting the last target layer must retain its full residual output."""
        for layer_ids, expected in (([7], [72.0]), ([0, 7], [2.0, 72.0])):
            with self.subTest(layer_ids=layer_ids):
                lm = _bare_causal_lm(capture=False)
                torch.nn.Module.__init__(lm)
                lm.model = _bare_model(layers_to_capture=[])
                lm.model.norm = lambda hidden_states: hidden_states / 2
                lm.set_eagle3_layers_to_capture(layer_ids)

                hidden_states, aux = lm.model.forward(
                    torch.zeros(3, dtype=torch.int64),
                    positions=None,
                    forward_batch=None,
                )

                self.assertEqual(len(aux), len(expected))
                for actual, value in zip(aux, expected, strict=True):
                    self.assertTrue(
                        torch.equal(actual, torch.full((3, _HIDDEN), value))
                    )
                self.assertTrue(
                    torch.equal(hidden_states, torch.full((3, _HIDDEN), 36.0))
                )

    def test_aux_reaches_logits_processor_unscaled(self):
        # scale_width divides only the logits input, not the aux hidden states.
        lm = _bare_causal_lm(capture=True)
        hidden = torch.full((3, _HIDDEN), 8.0)
        aux = [torch.full((3, _HIDDEN), 5.0)]
        lm.model.return_value = (hidden, aux)
        input_ids = torch.zeros(3, dtype=torch.int64)

        lm.forward(input_ids, positions=None, forward_batch="batch")

        ids, scaled_hidden, lm_head, batch, passed_aux = (
            lm.logits_processor.call_args.args
        )
        self.assertTrue(torch.equal(scaled_hidden, torch.full((3, _HIDDEN), 2.0)))
        self.assertIs(passed_aux, aux)
        self.assertIs(lm_head, lm.model.embed_tokens)


if __name__ == "__main__":
    unittest.main()
