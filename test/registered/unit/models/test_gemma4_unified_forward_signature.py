"""Gemma4 Unified exposes the forward parameters the runners detect by name."""

import inspect
import unittest

from sglang.srt.models.gemma4_unified import Gemma4UnifiedForConditionalGeneration
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestGemma4UnifiedForwardSignature(CustomTestCase):
    def test_forward_names_pp_proxy_tensors_and_input_embeds(self):
        # ModelRunner rejects --pp-size > 1 unless forward names
        # pp_proxy_tensors, and the CUDA graph runners only pass
        # input_embeds / pp_proxy_tensors to a forward that names them.
        parameters = inspect.signature(
            Gemma4UnifiedForConditionalGeneration.forward
        ).parameters
        self.assertIn("pp_proxy_tensors", parameters)
        self.assertIn("input_embeds", parameters)


if __name__ == "__main__":
    unittest.main()
