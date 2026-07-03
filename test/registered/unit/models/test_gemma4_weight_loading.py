"""Unit tests for Gemma4 weight loading -- CPU only, no GPU required.

Verifies that the unloaded-parameter diagnostics fire on a full checkpoint
load but are skipped during partial (online) weight updates.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.models.gemma4_causal import Gemma4ForCausalLM
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestGemma4WeightLoading(unittest.TestCase):
    def _make_minimal_causal_model(self):
        model = object.__new__(Gemma4ForCausalLM)
        model.config = SimpleNamespace(attention_k_eq_v=False)
        model.named_parameters = lambda: iter([("missing.weight", torch.empty(1))])
        model.named_buffers = lambda: iter([])
        model.named_modules = lambda: iter([])
        return model

    def test_full_load_logs_unloaded_parameters(self):
        model = self._make_minimal_causal_model()

        with self.assertLogs("sglang.srt.models.gemma4_causal", level="INFO") as logs:
            Gemma4ForCausalLM.load_weights(model, [], is_full_load=True)

        output = "\n".join(logs.output)
        self.assertIn("Some weights are not initialized from checkpoints", output)
        self.assertIn("missing.weight", output)

    def test_partial_load_skips_unloaded_parameter_logs(self):
        model = self._make_minimal_causal_model()

        with self.assertNoLogs("sglang.srt.models.gemma4_causal", level="INFO"):
            Gemma4ForCausalLM.load_weights(model, [], is_full_load=False)


if __name__ == "__main__":
    unittest.main()
