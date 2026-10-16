"""Unit tests for sglang/benchmark/datasets/speed_bench.py"""

import json
import os
import tempfile
import unittest

from sglang.benchmark.datasets.speed_bench import SpeedBenchDataset
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeTokenizer:
    """Tokenizes by whitespace. `as_dict` mimics transformers versions whose
    apply_chat_template(tokenize=True) returns a BatchEncoding."""

    def __init__(self, as_dict):
        self.as_dict = as_dict

    def apply_chat_template(self, messages, add_generation_prompt, tokenize):
        ids = list(range(len(messages[0]["content"].split()) + 3))  # + template tokens
        if self.as_dict:
            return {"input_ids": ids, "attention_mask": [1] * len(ids)}
        return ids

    def decode(self, ids):
        return " ".join(str(i) for i in ids)

    def encode(self, text):
        return text.split()


class TestSpeedBenchPromptLen(CustomTestCase):
    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".jsonl")
        with os.fdopen(fd, "w") as f:
            f.write(
                json.dumps({"category": "mixed", "turns": ["one two three four five"]})
                + "\n"
            )

    def tearDown(self):
        os.remove(self.path)

    def _load(self, tokenizer):
        ds = SpeedBenchDataset(
            dataset_path=self.path, category=None, output_len=16, num_requests=1
        )
        return ds.load(tokenizer)

    def test_prompt_len_with_list_return(self):
        (row,) = self._load(_FakeTokenizer(as_dict=False))
        self.assertEqual(row.prompt_len, 8)

    def test_prompt_len_with_batch_encoding_return(self):
        # Before the fix this was 2: the number of keys in the dict.
        (row,) = self._load(_FakeTokenizer(as_dict=True))
        self.assertEqual(row.prompt_len, 8)
        self.assertEqual(row.prompt, " ".join(str(i) for i in range(8)))


if __name__ == "__main__":
    unittest.main()
