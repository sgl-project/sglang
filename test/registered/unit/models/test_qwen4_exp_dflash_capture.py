"""Regression for DFLASH aux-hidden capture on Qwen4-Exp (Qwen3.8-Flash-Next).

DFLASH used to get no target features on this model: the decoder layers dropped
the capture list and the second return slot carried the MTP hyper-connection
streams. Drafters for it are trained on target layer k's completed output,
averaged over the hyper-connection streams, with no +1 shift.
"""

import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models import qwen4_exp
from sglang.srt.models.qwen4_exp import (
    Qwen4ExpForConditionalGeneration,
    Qwen4ExpVLModel,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

HC_COUNT, HIDDEN_SIZE, NUM_LAYERS, NUM_TOKENS = 4, 3, 6, 2


class _StreamLayer(nn.Module):
    """Adds a distinct offset to every stream, so a wrong layer or reduction shows."""

    def __init__(self, layer_id: int):
        super().__init__()
        self.ple = None
        self.attn_hyper_connection = SimpleNamespace(hc_norm=None)
        self.offset = torch.arange(HC_COUNT * HIDDEN_SIZE, dtype=torch.float32) ** (
            layer_id + 1
        )

    @contextmanager
    def hc_norms(self, normalized_input, next_norm):
        yield SimpleNamespace(output=None)

    def forward(self, positions, hidden_states, forward_batch, **kwargs):
        return hidden_states + self.offset


def _build_target() -> Qwen4ExpForConditionalGeneration:
    language_model = Qwen4ExpVLModel.__new__(Qwen4ExpVLModel)
    nn.Module.__init__(language_model)
    language_model.pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
    language_model.layers = nn.ModuleList(_StreamLayer(i) for i in range(NUM_LAYERS))
    language_model._start_layer, language_model._end_layer = 0, NUM_LAYERS
    language_model.has_ple = False
    language_model.hc_count, language_model.hidden_size = HC_COUNT, HIDDEN_SIZE
    language_model.dflash_layer_ids = ()
    language_model.last_hc_hidden_states = None
    language_model.hyper_connection_mixer = SimpleNamespace(
        hc_norm=None,
        mix=lambda x, normalized_input=None: (x[..., :HIDDEN_SIZE], None),
    )

    target = Qwen4ExpForConditionalGeneration.__new__(Qwen4ExpForConditionalGeneration)
    nn.Module.__init__(target)
    target.pp_group = SimpleNamespace(is_last_rank=True)
    target.model = language_model
    target.capture_aux_hidden_states = False
    return target


def _run_language_model(target, embeds):
    forward_batch = SimpleNamespace(
        forward_mode=ForwardMode.DECODE, residual_stream=None
    )
    with mock.patch.object(qwen4_exp, "_shards_rows_over_cp", return_value=False):
        return target.model(
            input_ids=torch.zeros(NUM_TOKENS, dtype=torch.long),
            positions=torch.zeros(NUM_TOKENS, dtype=torch.long),
            forward_batch=forward_batch,
            input_embeds=embeds,
        )


class TestQwen4ExpDflashCapture(CustomTestCase):
    def setUp(self):
        self.embeds = torch.randn(NUM_TOKENS, HC_COUNT * HIDDEN_SIZE)

    def _streams_after_layer(self, target, layer_id):
        return self.embeds + sum(
            target.model.layers[i].offset for i in range(layer_id + 1)
        )

    def test_captures_completed_layer_stream_mean_in_aux_slot(self):
        target = _build_target()
        target.set_dflash_layers_to_capture([1, 4])

        output = _run_language_model(target, self.embeds)

        self.assertTrue(target.capture_aux_hidden_states)
        self.assertIsInstance(output, tuple)
        _, aux_hidden_states = output
        self.assertEqual(len(aux_hidden_states), 2)
        for captured, layer_id in zip(aux_hidden_states, [1, 4]):
            streams = self._streams_after_layer(target, layer_id)
            torch.testing.assert_close(
                captured, streams.unflatten(-1, (HC_COUNT, HIDDEN_SIZE)).mean(dim=-2)
            )
        # The outer forward would write a stashed stream over the packed aux states.
        self.assertIsNone(target.model.last_hc_hidden_states)

    def test_mtp_streams_kept_without_dflash(self):
        target = _build_target()

        hidden_states = _run_language_model(target, self.embeds)

        self.assertIsInstance(hidden_states, torch.Tensor)
        torch.testing.assert_close(
            target.model.last_hc_hidden_states,
            self._streams_after_layer(target, NUM_LAYERS - 1),
        )


if __name__ == "__main__":
    unittest.main()
