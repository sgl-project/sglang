import json
import re
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestPenaltyB(CustomTestCase):
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

    def run_generate_with_prompt(self, prompt, sampling_params, max_tokens=100):
        """Helper method to generate text with a specific prompt and parameters."""
        sampling_params.setdefault("temperature", 0.05)
        sampling_params.setdefault("top_p", 1.0)

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

    def count_word_repetitions(self, text, word):
        """Count how many times a specific word appears in the text."""
        return len(re.findall(r"\b" + re.escape(word) + r"\b", text.lower()))

    def _test_penalty_effect(
        self,
        prompt,
        baseline_params,
        penalty_params,
        target_word,
        expected_reduction=True,
        max_tokens=50,
    ):
        """Generic test for penalty effects."""
        # Run multiple iterations to get more reliable results
        baseline_counts = []
        penalty_counts = []

        for i in range(5):
            baseline_output = self.run_generate_with_prompt(
                prompt, baseline_params, max_tokens
            )
            penalty_output = self.run_generate_with_prompt(
                prompt, penalty_params, max_tokens
            )

            baseline_count = self.count_word_repetitions(baseline_output, target_word)
            penalty_count = self.count_word_repetitions(penalty_output, target_word)

            baseline_counts.append(baseline_count)
            penalty_counts.append(penalty_count)

        # Calculate averages
        avg_baseline = sum(baseline_counts) / len(baseline_counts)
        avg_penalty = sum(penalty_counts) / len(penalty_counts)

        if expected_reduction:
            # Simple check: penalty should reduce repetition
            self.assertLess(
                avg_penalty,
                avg_baseline,
                f"Penalty should reduce '{target_word}' repetition: {avg_baseline:.1f} → {avg_penalty:.1f}",
            )
        else:
            self.assertGreater(
                avg_penalty,
                avg_baseline,
                f"Negative penalty should increase '{target_word}' repetition",
            )

    def test_frequency_penalty_reduces_word_repetition(self):
        """Test frequency penalty using word repetition."""
        prompt = "Write exactly 10 very small sentences, each containing the word 'data'. Use the word 'data' as much as possible."
        baseline_params = {"frequency_penalty": 0.0, "repetition_penalty": 1.0}
        penalty_params = {"frequency_penalty": 1.99, "repetition_penalty": 1.0}
        self._test_penalty_effect(prompt, baseline_params, penalty_params, "data")

    def test_presence_penalty_reduces_topic_repetition(self):
        """Test presence penalty using topic repetition."""
        prompt = "Write the word 'machine learning' exactly 20 times in a row, separated by spaces."
        baseline_params = {"presence_penalty": 0.0, "repetition_penalty": 1.0}
        penalty_params = {"presence_penalty": 1.99, "repetition_penalty": 1.0}
        self._test_penalty_effect(
            prompt, baseline_params, penalty_params, "machine learning"
        )

    def test_combined_penalties_reduce_repetition(self):
        """Test combined penalty effects."""
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
        self._test_penalty_effect(
            prompt, baseline_params, penalty_params, "data", max_tokens=100
        )

    def test_penalty_edge_cases_negative_penalty_values(self):
        """Test edge cases with negative penalty values."""
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
        # Negative penalties should increase repetition (expected_reduction=False)
        self._test_penalty_effect(
            prompt,
            baseline_params,
            negative_penalty_params,
            "test",
            expected_reduction=False,
            max_tokens=60,
        )

    def test_penalty_edge_cases_extreme_penalty_values(self):
        """Test edge cases with extreme penalty values."""
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
        # Extreme penalties should strongly reduce repetition
        self._test_penalty_effect(
            prompt,
            baseline_params,
            extreme_penalty_params,
            "extreme",
            expected_reduction=True,
            max_tokens=80,
        )


if __name__ == "__main__":
    unittest.main(verbosity=3)
