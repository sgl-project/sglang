import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestNemotronDecisionAdmission(CustomTestCase):
    def setUp(self):
        override = get_context().override_server_args(max_prefill_tokens=64)
        override.install()
        self.addCleanup(override.restore)
        self.manager = TokenizerManager.__new__(TokenizerManager)
        self.manager.is_single_token_read = True
        self.manager.single_read_max_input_tokens = 128
        self.manager.model_config = SimpleNamespace(
            joint_head_config=None, pointer_head=None
        )
        self.manager.context_len = 128
        self.manager.max_req_input_len = 128
        self.manager.num_reserved_tokens = 0
        self.manager.allow_auto_truncate = True
        self.manager.validate_total_tokens = False
        self.manager.is_generation = True
        self.manager._validate_token_ids_logprob = Mock()

    def validate(self, length, max_new_tokens):
        req = GenerateReqInput(
            input_ids=[1] * length, sampling_params={"max_new_tokens": max_new_tokens}
        )
        self.manager._validate_one_request(req, req.input_ids)
        return req

    def test_single_read_limit_does_not_truncate_or_generate_an_invalid_second_token(
        self,
    ):
        self.validate(64, 0)
        self.validate(64, 1)
        with self.assertRaisesRegex(ValueError, "complete prompt"):
            self.validate(65, 0)
        for limit in (None, 2):
            with (
                self.subTest(limit=limit),
                self.assertRaisesRegex(ValueError, "single-token"),
            ):
                self.validate(20, limit)
        self.manager.context_len = 64
        self.validate(63, 0)
        with self.assertRaisesRegex(ValueError, "complete prompt"):
            self.validate(64, 0)
        self.manager.context_len = 128
        self.manager.max_req_input_len = 33
        self.validate(32, 0)
        with self.assertRaisesRegex(ValueError, "complete prompt"):
            self.validate(33, 0)
        self.manager.max_req_input_len = 128
        self.manager.single_read_max_input_tokens = 12
        self.validate(12, 0)
        with self.assertRaisesRegex(ValueError, "complete prompt"):
            self.validate(13, 0)
        self.manager.is_single_token_read = False
        self.validate(65, 128)


if __name__ == "__main__":
    unittest.main()
