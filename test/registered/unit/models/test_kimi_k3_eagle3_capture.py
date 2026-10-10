"""EAGLE3 aux hidden capture for Kimi-K3 (kimi-k3-eagle3.1-mla contract).

The draft checkpoint is trained against one-based completed-layer ids
([2, 46, 90] on the 93-layer target) tapping the plain prefix stream, not the
AttnRes mixture DSPARK captures. These cases pin the id convention and the
bank-path dispatch that the checkpoint contract depends on.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from torch import nn

import sglang.srt.models.kimi_k3 as kimi_k3_mod
from sglang.srt.models.kimi_k3 import KimiK3LinearForCausalLM, KimiK3LinearModel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_H = 64


class _FakePPGroup:
    def __init__(self, world_size=1, is_last_rank=True):
        self.world_size = world_size
        self.is_first_rank = True
        self.is_last_rank = is_last_rank


def _bare_model(*, dspark_ids=None, eagle3_ids=None):
    model = object.__new__(KimiK3LinearModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(hidden_size=_H, num_hidden_layers=93)
    model.pp_group = _FakePPGroup()
    model.dspark_layers_to_capture = dspark_ids
    model.eagle3_layers_to_capture = eagle3_ids
    return model


def _bare_lm():
    lm = object.__new__(KimiK3LinearForCausalLM)
    nn.Module.__init__(lm)
    lm.config = SimpleNamespace(num_hidden_layers=93, hidden_size=_H)
    lm.pp_group = _FakePPGroup()
    lm.model = SimpleNamespace(
        eagle3_layers_to_capture=None,
        dspark_layers_to_capture=None,
        carries_bank_slices=False,
        packs_aux_hidden_states=True,
        capture_layer_ids=(1, 45, 89),
    )
    lm.capture_aux_hidden_states = False
    return lm


class TestKimiK3Eagle3Capture(CustomTestCase):
    def test_eagle3_ids_map_to_zero_based_completed_layers(self):
        # one-based [2, 46, 90] -> stream after zero-based layers 1, 45, 89.
        self.assertEqual(
            _bare_model(eagle3_ids=(2, 46, 90)).capture_layer_ids, (1, 45, 89)
        )

    def test_bank_path_eagle3_takes_plain_prefix_stream(self):
        # The AttnRes mixture is a different feature stream from the prefix;
        # eagle3 must snapshot the stream, not reuse the DSPARK aggregate.
        model = _bare_model(eagle3_ids=(2,))
        hidden, batch, snap = object(), object(), object()
        with (
            patch.object(
                kimi_k3_mod.residual_batch, "snapshot", return_value=snap
            ) as snapshot,
            patch.object(KimiK3LinearModel, "_dspark_capture_stream") as dspark_capture,
        ):
            out = model._capture_bank_stream(1, hidden, batch)
        self.assertIs(out, snap)
        snapshot.assert_called_once_with(hidden, batch)
        dspark_capture.assert_not_called()

    def test_bank_path_dspark_still_takes_aggregate(self):
        model = _bare_model(dspark_ids=[1])
        hidden, batch, mixed = object(), object(), object()
        with (
            patch.object(kimi_k3_mod.residual_batch, "snapshot") as snapshot,
            patch.object(
                KimiK3LinearModel, "_dspark_capture_stream", return_value=mixed
            ) as dspark_capture,
        ):
            out = model._capture_bank_stream(1, hidden, batch)
        self.assertIs(out, mixed)
        dspark_capture.assert_called_once_with(1, hidden, batch)
        snapshot.assert_not_called()

    def test_default_ids_for_93_layers_match_checkpoint(self):
        lm = _bare_lm()
        lm.set_eagle3_layers_to_capture(None)
        # [2, 46, 90]: the one-based ids kimi-k3-eagle3.1-mla declares.
        self.assertEqual(lm.model.eagle3_layers_to_capture, (2, 46, 90))
        self.assertTrue(lm.capture_aux_hidden_states)

    def test_checkpoint_ids_stored_without_the_plus_one_shift(self):
        # deepseek_v2 bumps ids starting at 1; this contract is one-based already.
        lm = _bare_lm()
        lm.set_eagle3_layers_to_capture([2, 46, 90])
        self.assertEqual(lm.model.eagle3_layers_to_capture, (2, 46, 90))

    def test_aux_width_counts_eagle3_captures(self):
        # Width must follow capture_layer_ids, not dspark_layers_to_capture (None here).
        self.assertEqual(_bare_lm().get_aux_hidden_states_width(), 3 * _H)


if __name__ == "__main__":
    unittest.main()
