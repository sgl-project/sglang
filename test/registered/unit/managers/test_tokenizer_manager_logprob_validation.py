"""Unit test for TokenizerManager._validate_logprob_nums.

Bug: top_logprobs_num (native /generate) and the OpenAI logprobs / top_logprobs
fields had no upper bound. The value flows into logprobs.topk(k) over the
[*, vocab_size] logits; k > vocab_size is an out-of-range topk (on CUDA a
device-side assert that crashes the whole server) and k < 0 is likewise invalid.
verify() bounded logit_bias and token_ids_logprob keys but not this count.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

VOCAB_SIZE = 32000


def _manager():
    tm = TokenizerManager.__new__(TokenizerManager)
    tm.model_config = SimpleNamespace(vocab_size=VOCAB_SIZE)
    return tm


def _req(top_logprobs_num):
    obj = Mock(spec=GenerateReqInput)
    obj.top_logprobs_num = top_logprobs_num
    return obj


class TestValidateLogprobNums(CustomTestCase):
    def setUp(self):
        self.tm = _manager()

    def test_above_vocab_raises(self):
        # k > vocab_size -> topk(k) is out of range (the DoS).
        with self.assertRaises(ValueError):
            self.tm._validate_logprob_nums(_req(VOCAB_SIZE + 1))
        with self.assertRaises(ValueError):
            self.tm._validate_logprob_nums(_req(2_000_000_000))

    def test_negative_raises(self):
        with self.assertRaises(ValueError):
            self.tm._validate_logprob_nums(_req(-1))

    def test_in_range_passes(self):
        for k in (0, 1, 20, VOCAB_SIZE):
            self.tm._validate_logprob_nums(_req(k))

    def test_none_passes(self):
        self.tm._validate_logprob_nums(_req(None))


if __name__ == "__main__":
    unittest.main()
