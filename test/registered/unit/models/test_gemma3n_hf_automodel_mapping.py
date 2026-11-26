"""Importing SGLang's Gemma3n model leaves HF's AutoModel mapping alone."""

import unittest

from transformers import AutoModel, Gemma3nTextConfig
from transformers.models.gemma3n.modeling_gemma3n import Gemma3nTextModel

import sglang.srt.models.gemma3n_causal  # noqa: F401
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestGemma3nKeepsHfAutoModelMapping(CustomTestCase):
    def test_text_config_still_resolves_to_the_hf_text_model(self):
        """HF's Gemma3nModel builds its text tower with AutoModel.from_config;
        a process that imported SGLang's Gemma3n must still get HF's class
        there, not SGLang's entry class with its different forward."""
        self.assertIs(AutoModel._model_mapping[Gemma3nTextConfig], Gemma3nTextModel)


if __name__ == "__main__":
    unittest.main()
