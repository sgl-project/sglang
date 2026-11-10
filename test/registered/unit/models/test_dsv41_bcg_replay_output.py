"""BCG replay must not prefer a CSA2 eager-break tensor over the LPO."""

from __future__ import annotations

import unittest

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend import (
    select_bcg_replay_output,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDsv41BcgReplayOutput(CustomTestCase):
    def test_prefers_live_lpo(self):
        live = LogitsProcessorOutput(
            next_token_logits=torch.zeros(1, 4), hidden_states=None
        )
        stored = LogitsProcessorOutput(
            next_token_logits=None, hidden_states=torch.ones(2, 3)
        )
        self.assertIs(select_bcg_replay_output(live, stored), live)

    def test_falls_back_to_stored_lpo_when_live_is_break_tensor(self):
        live = torch.zeros(2, 8)
        stored = LogitsProcessorOutput(
            next_token_logits=None, hidden_states=torch.ones(2, 8)
        )
        self.assertIs(select_bcg_replay_output(live, stored), stored)


if __name__ == "__main__":
    unittest.main()
