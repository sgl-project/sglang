"""Unit test for TokenizerManager._validate_input_ids_in_vocab.

Bug: user-supplied input_ids (native /generate and OpenAI token-id prompts) were
never validated -- _validate_input_ids_in_vocab existed but had zero callers, and
even it only checked id >= vocab_size (missing id < 0). An out-of-range id indexes
the model's embedding out of bounds (on CUDA a device-side assert that crashes the
whole server); a negative id silently wraps to the end of the table.
"""

import unittest
from types import SimpleNamespace

from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

VOCAB_SIZE = 32000


class TestValidateInputIds(CustomTestCase):
    def setUp(self):
        self.tm = TokenizerManager.__new__(TokenizerManager)
        self.tm.model_config = SimpleNamespace(vocab_size=VOCAB_SIZE)

    def _validate(self, ids):
        self.tm._validate_input_ids_in_vocab(ids, VOCAB_SIZE)

    def test_above_vocab_raises(self):
        with self.assertRaises(ValueError):
            self._validate([1, 2, VOCAB_SIZE])
        with self.assertRaises(ValueError):
            self._validate([1, 2, 2_000_000_000])

    def test_negative_raises(self):
        # The previous check only guarded id >= vocab_size, so this used to pass.
        with self.assertRaises(ValueError):
            self._validate([1, -1, 2])

    def test_batch_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            self._validate([[1, 2], [3, VOCAB_SIZE + 5]])
        with self.assertRaises(ValueError):
            self._validate([[1, 2], [-4, 5]])

    def test_in_range_passes(self):
        self._validate([0, 1, VOCAB_SIZE - 1])
        self._validate([[0, 1], [2, VOCAB_SIZE - 1]])

    def test_empty_passes(self):
        self._validate([])


if __name__ == "__main__":
    unittest.main()
