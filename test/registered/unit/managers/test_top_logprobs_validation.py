import unittest
from types import SimpleNamespace

from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestTopLogprobsValidation(CustomTestCase):
    def setUp(self):
        override = get_context().override_server_args()
        override.install()
        self.addCleanup(override.restore)
        self.manager = TokenizerManager.__new__(TokenizerManager)
        self.manager.context_len = 128
        self.manager.num_reserved_tokens = 0
        self.manager.allow_auto_truncate = False
        self.manager.validate_total_tokens = False
        self.manager.is_generation = True
        self.manager.model_config = SimpleNamespace(
            joint_head_config=None, vocab_size=32000
        )

    def test_top_logprobs_num_above_vocab_raises(self):
        req = GenerateReqInput(
            input_ids=[1, 2, 3],
            sampling_params={"top_k": 10},
            top_logprobs_num=32001,
        )
        with self.assertRaisesRegex(
            ValueError, r"top_logprobs_num must be in \[0, vocab_size"
        ):
            self.manager._validate_one_request(req, req.input_ids)

    def test_top_logprobs_num_negative_raises(self):
        req = GenerateReqInput(
            input_ids=[1, 2, 3],
            sampling_params={"top_k": 10},
            top_logprobs_num=-1,
        )
        with self.assertRaises(ValueError):
            self.manager._validate_one_request(req, req.input_ids)

    def test_top_logprobs_num_int32_overflow_raises(self):
        req = GenerateReqInput(
            input_ids=[1, 2, 3],
            sampling_params={"top_k": 10},
            top_logprobs_num=2**31,
        )
        with self.assertRaises(ValueError):
            self.manager._validate_one_request(req, req.input_ids)

    def test_top_logprobs_num_valid(self):
        for value in (0, 20, 32000):
            with self.subTest(value=value):
                req = GenerateReqInput(
                    input_ids=[1, 2, 3],
                    sampling_params={"top_k": 10},
                    top_logprobs_num=value,
                )
                self.manager._validate_one_request(req, req.input_ids)

    def test_top_logprobs_num_none_valid(self):
        req = GenerateReqInput(
            input_ids=[1, 2, 3],
            sampling_params={"top_k": 10},
            top_logprobs_num=None,
        )
        self.manager._validate_one_request(req, req.input_ids)


if __name__ == "__main__":
    unittest.main()
