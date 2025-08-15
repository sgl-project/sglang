"""
Test classifier-free guidance functionality.

Usage:
python3 -m unittest test_cfg.TestClassifierFreeGuidance
or
python3 test_cfg.py
"""

import unittest

from transformers import AutoTokenizer

import sglang as sgl
from sglang.test.test_utils import DEFAULT_MODEL_NAME_FOR_TEST, CustomTestCase


class TestClassifierFreeGuidance(CustomTestCase):
    """Test cases for classifier-free guidance functionality."""

    @classmethod
    def setUpClass(cls):
        cls.model_path = DEFAULT_MODEL_NAME_FOR_TEST
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_path)
        cls.engine = sgl.Engine(
            model_path=cls.model_path,
            random_seed=42,
        )

    def setUp(self):
        """Set up test environment for each test."""
        self.base_prompt = "Today's weather is hot and,"
        self.bad_cfg_prompt = "The weather is very bad,"
        self.good_cfg_prompt = "The weather is very good,"

        self.sampling_params = {
            "temperature": 0,
            "max_new_tokens": 8,
        }

        self.top_logprobs_num = 3
        self.atol = 1e-4

    def _gen_logprobs(self, cfg_params):
        num_cfg_params = len(cfg_params)

        outputs = self.engine.generate(
            prompt=[self.base_prompt] * num_cfg_params,
            sampling_params=self.sampling_params,
            return_logprob=True,
            top_logprobs_num=self.top_logprobs_num,
            cfg_params=cfg_params,
        )
        self.assertEqual(len(outputs), num_cfg_params)

        return outputs

    def _compare_logprobs_pair(self, output_0, output_1, is_equal):
        for a_out, b_out in zip(
            output_0["meta_info"]["output_top_logprobs"],
            output_1["meta_info"]["output_top_logprobs"],
        ):
            for a_tok, b_tok in zip(a_out, b_out):
                if is_equal:
                    self.assertAlmostEqual(a_tok[0], b_tok[0], delta=self.atol)
                else:
                    self.assertNotEqual(a_tok[0], b_tok[0])

    def test_no_cfg_weight(self):
        """Test that passing in cfg_weight = 0 results in identical logprobs."""
        cfg_prompts = [self.bad_cfg_prompt, self.good_cfg_prompt]
        cfg_params = [
            {
                "cfg_text": text,
                "cfg_weight": 00,
            }
            for text in cfg_prompts
        ]

        outputs = self._gen_logprobs(cfg_params)
        self._compare_logprobs_pair(outputs[0], outputs[1], True)

    def test_unique_cfg_text(self):
        """Test that passing in same cfg_weight > 0 and unique cfg_text results in different logprobs."""
        cfg_prompts = [self.bad_cfg_prompt, self.good_cfg_prompt]
        cfg_params = [
            {
                "cfg_text": text,
                "cfg_weight": 0.5,
            }
            for text in cfg_prompts
        ]

        outputs = self._gen_logprobs(cfg_params)
        self._compare_logprobs_pair(outputs[0], outputs[1], False)

    def test_unique_cfg_params(self):
        """Test that passing in different CFG weights with the same prompts results in different logprobs."""
        cfg_prompts = [self.bad_cfg_prompt] * 2
        cfg_weights = [0.3, 0.7]
        cfg_params = [
            {
                "cfg_text": text,
                "cfg_weight": cfg_weight,
            }
            for text in cfg_prompts
            for cfg_weight in cfg_weights
        ]

        outputs = self._gen_logprobs(cfg_params)
        for out1, out2 in [(outputs[0], outputs[1]), (outputs[2], outputs[3])]:
            self._compare_logprobs_pair(out1, out2, False)

    @classmethod
    def tearDownClass(cls):
        cls.engine.shutdown()


if __name__ == "__main__":
    unittest.main()
