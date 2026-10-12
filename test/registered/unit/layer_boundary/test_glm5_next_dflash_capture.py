"""Regression for DFLASH aux-hidden capture on mHC models.

GLM-5.3-Flash runs with mhc=True. Its MHC boundary folds the residual
into the widened hidden state and returns residual=None, so CUDA-graph
capture used to crash on `hidden_states + residual`. DFLASH also has to
contract that widened state back to the draft hidden size; skipping the
contract is a silent shape/quality bug the crash-guard alone would miss.
"""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.layers.layer_boundary.residual.stream import ResidualStream
from sglang.srt.models.glm5_next import Glm5NextModel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestGlm5NextDflashCapture(CustomTestCase):
    def test_cp_bcg_gathers_each_aux_feature_before_draft_materialization(self):
        from sglang.srt.layers.cp.bcg import execute_prefill_cp_bcg

        # This rank owns tokens 0,4; one additional row is capture padding.
        local = torch.tensor([[0.0], [4.0], [-1.0]])
        auxiliary = [local + 10, local + 20]
        global_tokens = torch.arange(8, dtype=torch.float32).view(-1, 1)

        def gather(value, _batch, _stream):
            self.assertEqual(value.shape[0], 2)
            return global_tokens + value[0, 0]

        def logits(_ids, hidden, _head, _batch, aux):
            torch.testing.assert_close(hidden, global_tokens)
            for actual, offset in zip(aux, (10, 20)):
                torch.testing.assert_close(actual, global_tokens + offset)
            return torch.cat(aux, dim=-1)

        runner = SimpleNamespace(
            prefill_cp_bcg_input=SimpleNamespace(live_local_tokens=2),
            model_runner=SimpleNamespace(
                model=SimpleNamespace(
                    capture_aux_hidden_states=True,
                    pp_group=SimpleNamespace(is_last_rank=True),
                    logits_processor=logits,
                    lm_head=None,
                )
            ),
            backend=SimpleNamespace(replay=lambda *_a, **_kw: (local, auxiliary)),
            _prefill_forward_context=lambda *_a, **_kw: nullcontext(),
        )
        batch = SimpleNamespace(input_ids=torch.arange(8))
        with (
            patch(
                "sglang.srt.layers.cp.bcg.cp_gather_after_forward", side_effect=gather
            ),
            patch("torch.cuda.current_stream", return_value=None),
        ):
            actual = execute_prefill_cp_bcg(runner, batch, batch, 8, 8)
        self.assertEqual(actual.shape, (8, 2))

    def test_dflash_contracts_mhc_hidden_state_without_residual(self):
        model = Glm5NextModel.__new__(Glm5NextModel)
        nn.Module.__init__(model)
        model.config = SimpleNamespace(mhc=True, hc_mult=4)
        model.dflash_capture = True

        hidden_states = torch.arange(24, dtype=torch.float32).reshape(2, 12)

        captured = ResidualStream().snapshot(hidden_states)
        actual = model._prepare_aux_hidden_state(captured)
        expected = hidden_states.unflatten(-1, (4, -1)).mean(dim=-2)

        torch.testing.assert_close(actual, expected)
        self.assertEqual(tuple(actual.shape), (2, 3))


if __name__ == "__main__":
    unittest.main()
