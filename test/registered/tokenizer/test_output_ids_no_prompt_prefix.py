"""
Test that output_ids contains only generated tokens, not input prompt tokens.

This test verifies the fix for GitHub issue #17240:
https://github.com/sgl-project/sglang/issues/17240

The bug was that when calling engine.generate(input_ids=...), the returned
output_ids contained tokens from the input prompt prefix, not just the
generated tokens. However, the text field was correctly decoded.

Run with:
python3 -m pytest test/registered/tokenizer/test_output_ids_no_prompt_prefix.py -v
"""

import unittest

import requests
from transformers import AutoTokenizer

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=60, suite="stage-b-test-small-1-gpu")


class TestOutputIdsNoPromptPrefix(CustomTestCase):
    """Test that output_ids only contains generated tokens."""

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model)
        # Use default settings (skip_tokenizer_init=False)
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_output_ids_matches_completion_tokens(self):
        """Test that len(output_ids) equals completion_tokens count."""
        prompt = "What is 2+2? Answer:"
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        max_new_tokens = 10

        response = requests.post(
            self.base_url + "/generate",
            json={
                "input_ids": input_ids,
                "sampling_params": {
                    "max_new_tokens": max_new_tokens,
                    "temperature": 0,
                },
            },
        )
        result = response.json()

        output_ids = result["output_ids"]
        completion_tokens = result["meta_info"]["completion_tokens"]

        # The key assertion: output_ids should only contain generated tokens
        self.assertEqual(
            len(output_ids),
            completion_tokens,
            f"output_ids length ({len(output_ids)}) should equal "
            f"completion_tokens ({completion_tokens}). "
            f"If output_ids is longer, it likely contains input prompt tokens.",
        )

    def test_output_ids_decodes_to_text(self):
        """Test that tokenizer.decode(output_ids) matches the returned text."""
        prompt = "The capital of France is"
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        max_new_tokens = 20

        response = requests.post(
            self.base_url + "/generate",
            json={
                "input_ids": input_ids,
                "sampling_params": {
                    "max_new_tokens": max_new_tokens,
                    "temperature": 0,
                },
            },
        )
        result = response.json()

        output_ids = result["output_ids"]
        returned_text = result["text"]

        # Decode the output_ids and compare with returned text
        decoded_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)

        # The decoded output_ids should match the returned text
        # Note: There might be minor whitespace differences, so we compare stripped versions
        self.assertEqual(
            decoded_text.strip(),
            returned_text.strip(),
            f"Decoded output_ids ({repr(decoded_text)}) should match "
            f"returned text ({repr(returned_text)}). "
            f"If decoded text contains prompt, output_ids has input tokens.",
        )

    def test_output_ids_no_prompt_tokens(self):
        """Test that output_ids does not contain any tokens from the input prompt."""
        prompt = "Hello world test prompt:"
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        max_new_tokens = 15

        response = requests.post(
            self.base_url + "/generate",
            json={
                "input_ids": input_ids,
                "sampling_params": {
                    "max_new_tokens": max_new_tokens,
                    "temperature": 0,
                },
            },
        )
        result = response.json()

        output_ids = result["output_ids"]

        # Check that the first few output_ids don't match the last few input_ids
        # This would indicate prompt tokens leaking into output_ids
        num_check = min(5, len(output_ids), len(input_ids))
        last_input_tokens = input_ids[-num_check:]
        first_output_tokens = output_ids[:num_check]

        # They should not match exactly (unless the model generates the same tokens,
        # which is unlikely for typical prompts)
        if last_input_tokens == first_output_tokens:
            # Additional check: if they match, verify completion_tokens
            completion_tokens = result["meta_info"]["completion_tokens"]
            self.assertEqual(
                len(output_ids),
                completion_tokens,
                f"First {num_check} output_ids match last {num_check} input_ids. "
                f"This suggests prompt tokens are leaking into output_ids.",
            )


if __name__ == "__main__":
    unittest.main()
