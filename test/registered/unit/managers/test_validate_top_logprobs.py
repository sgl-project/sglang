"""top_logprobs must be validated against the vocab at the API boundary.

top_logprobs_num flows into logprobs.topk(k); k > vocab_size raises a
RuntimeError deep in the model forward, so reject it up front with a clean
error. Mirrors vLLM's SamplingParams._validate_logprobs.
"""

import unittest
from types import SimpleNamespace

from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(3, "base-a-test-cpu")

VOCAB_SIZE = 1000


def _validate(top_logprobs_num):
    tm = TokenizerManager.__new__(TokenizerManager)
    tm.model_config = SimpleNamespace(vocab_size=VOCAB_SIZE)
    obj = GenerateReqInput(text="hello")
    obj.top_logprobs_num = top_logprobs_num
    tm._validate_top_logprobs_num(obj)


class TestValidateTopLogprobsNum(unittest.TestCase):
    def test_in_range_ok(self):
        for ok in (None, 0, 20, VOCAB_SIZE):  # topk(vocab_size) is valid
            with self.subTest(top_logprobs_num=ok):
                _validate(ok)

    def test_non_integer_raises(self):
        for bad in (False, True, 0.0, [], ""):
            with self.subTest(top_logprobs_num=bad):
                with self.assertRaises(ValueError):
                    _validate(bad)

    def test_out_of_range_raises(self):
        for bad in (VOCAB_SIZE + 1, 100000, -1):
            with self.subTest(top_logprobs_num=bad):
                with self.assertRaises(ValueError):
                    _validate(bad)


if __name__ == "__main__":
    unittest.main()
