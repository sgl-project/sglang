import json
import random
import re
import unittest
from concurrent.futures import ThreadPoolExecutor

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=82, suite="stage-b-test-small-1-gpu")
register_amd_ci(est_time=82, suite="stage-b-test-small-1-gpu-amd")


class TestPenalty(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def run_decode(self, sampling_params):
        """Helper method for basic decode tests."""
        return_logprob = True
        top_logprobs_num = 5
        return_text = True
        n = 1

        response = requests.post(
            self.base_url + "/generate",
            json={
                # prompt that is supposed to generate < 32 tokens
                "text": "<|start_header_id|>user<|end_header_id|>\n\nWhat is the answer for 1 + 1 = ?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                "sampling_params": {
                    "max_new_tokens": 48,
                    "n": n,
                    **sampling_params,
                },
                "return_logprob": return_logprob,
                "top_logprobs_num": top_logprobs_num,
                "return_text_in_logprobs": return_text,
                "logprob_start_len": 0,
            },
        )
        self.assertEqual(response.status_code, 200)
        print(json.dumps(response.json()))
        print("=" * 100)

    def run_generate_with_prompt(
        self, prompt, sampling_params, max_tokens=100, seed=None
    ):
        """Helper method to generate text with a specific prompt and parameters."""
        sampling_params = sampling_params.copy()
        sampling_params.setdefault("temperature", 0.05)
        sampling_params.setdefault("top_p", 1.0)
        if seed is not None:
            sampling_params["seed"] = seed

        response = requests.post(
            self.base_url + "/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                **sampling_params,
            },
        )
        self.assertEqual(response.status_code, 200)
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        return content

    def _get_vocab_diversity(self, text):
        """Calculate vocabulary diversity as unique_words / total_words.

        Higher values mean more diverse (less repetitive) text.
        """
        words = re.findall(r"\b\w+\b", text.lower())
        if not words:
            return 1.0
        return len(set(words)) / len(words)

    def _test_penalty_effect(
        self,
        prompt,
        baseline_params,
        penalty_params,
        expected_reduction=True,
        max_tokens=150,
    ):
        """Generic test for penalty effects using vocabulary diversity.

        Measures unique_words/total_words ratio instead of counting a specific
        word, because penalties affect ALL token probabilities — the model may
        avoid some repeated tokens while using others more.
        """
        # Use higher temperature so penalties can actually affect token selection.
        # The default temperature (0.05) is near-greedy, making penalty adjustments
        # to logits ineffective since the top token still dominates.
        baseline_params = baseline_params.copy()
        penalty_params = penalty_params.copy()
        baseline_params.setdefault("temperature", 0.8)
        penalty_params.setdefault("temperature", 0.8)

        # Run multiple iterations to get more reliable results
        # Use fixed seeds for deterministic behavior
        base_seed = 42
        baseline_diversities = []
        penalty_diversities = []

        for i in range(5):
            seed = base_seed + i
            baseline_output = self.run_generate_with_prompt(
                prompt, baseline_params, max_tokens, seed=seed
            )
            penalty_output = self.run_generate_with_prompt(
                prompt, penalty_params, max_tokens, seed=seed
            )

            baseline_diversities.append(self._get_vocab_diversity(baseline_output))
            penalty_diversities.append(self._get_vocab_diversity(penalty_output))

        avg_baseline = sum(baseline_diversities) / len(baseline_diversities)
        avg_penalty = sum(penalty_diversities) / len(penalty_diversities)

        if expected_reduction:
            # Penalty should increase vocabulary diversity (less repetition)
            self.assertGreater(
                avg_penalty,
                avg_baseline,
                f"Penalty should increase vocab diversity: {avg_baseline:.3f} → {avg_penalty:.3f}",
            )
        else:
            # Negative penalty should decrease diversity (more repetition)
            self.assertLess(
                avg_penalty,
                avg_baseline,
                f"Negative penalty should decrease vocab diversity: {avg_baseline:.3f} → {avg_penalty:.3f}",
            )

    def test_default_values(self):
        self.run_decode({})

    def test_frequency_penalty(self):
        self.run_decode({"frequency_penalty": 2})

    def test_min_new_tokens(self):
        self.run_decode({"min_new_tokens": 16})

    def test_presence_penalty(self):
        self.run_decode({"presence_penalty": 2})

    def test_penalty_mixed(self):
        args = [
            {},
            {},
            {},
            {"frequency_penalty": 2},
            {"presence_penalty": 1},
            {"min_new_tokens": 16},
            {"frequency_penalty": 0.2},
            {"presence_penalty": 0.4},
            {"min_new_tokens": 8},
            {"frequency_penalty": 0.4, "presence_penalty": 0.8},
            {"frequency_penalty": 0.4, "min_new_tokens": 12},
            {"presence_penalty": 0.8, "min_new_tokens": 12},
            {"presence_penalty": -0.3, "frequency_penalty": 1.3, "min_new_tokens": 32},
            {"presence_penalty": 0.3, "frequency_penalty": -1.3, "min_new_tokens": 32},
        ]
        random.shuffle(args * 5)
        with ThreadPoolExecutor(8) as executor:
            list(executor.map(self.run_decode, args))

    def test_frequency_penalty_reduces_word_repetition(self):
        """Test that frequency penalty increases vocabulary diversity."""
        prompt = "Write exactly 10 very small sentences, each containing the word 'data'. Use the word 'data' as much as possible."
        baseline_params = {"frequency_penalty": 0.0, "repetition_penalty": 1.0}
        penalty_params = {"frequency_penalty": 1.99, "repetition_penalty": 1.0}
        self._test_penalty_effect(prompt, baseline_params, penalty_params)

    def test_presence_penalty_reduces_topic_repetition(self):
        """Test that presence penalty increases vocabulary diversity."""
        prompt = "Write the word 'machine learning' exactly 20 times in a row, separated by spaces."
        baseline_params = {"presence_penalty": 0.0, "repetition_penalty": 1.0}
        penalty_params = {"presence_penalty": 1.99, "repetition_penalty": 1.0}
        self._test_penalty_effect(prompt, baseline_params, penalty_params)

    def test_combined_penalties_reduce_repetition(self):
        """Test that combined penalties increase vocabulary diversity."""
        prompt = "Write exactly 10 short sentences, each containing the word 'data'. Use the word 'data' as much as possible."
        baseline_params = {
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "repetition_penalty": 1.0,
        }
        penalty_params = {
            "frequency_penalty": 1.99,
            "presence_penalty": 1.99,
            "repetition_penalty": 1.99,
        }
        self._test_penalty_effect(prompt, baseline_params, penalty_params)

    def test_penalty_edge_cases_negative_penalty_values(self):
        """Test that negative penalties decrease vocabulary diversity."""
        prompt = "Write the word 'test' exactly 15 times in a row, separated by spaces."
        baseline_params = {
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "repetition_penalty": 1.0,
        }
        negative_penalty_params = {
            "frequency_penalty": -0.5,
            "presence_penalty": -0.25,
            "repetition_penalty": 1.0,
        }
        self._test_penalty_effect(
            prompt,
            baseline_params,
            negative_penalty_params,
            expected_reduction=False,
        )

    def test_penalty_edge_cases_extreme_penalty_values(self):
        """Test that extreme penalties strongly increase vocabulary diversity."""
        prompt = (
            "Write the word 'extreme' exactly 20 times in a row, separated by spaces."
        )
        baseline_params = {
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "repetition_penalty": 1.0,
        }
        extreme_penalty_params = {
            "frequency_penalty": 2.0,
            "presence_penalty": 2.0,
            "repetition_penalty": 2.0,
        }
        self._test_penalty_effect(
            prompt,
            baseline_params,
            extreme_penalty_params,
        )


if __name__ == "__main__":
    unittest.main(verbosity=3)
