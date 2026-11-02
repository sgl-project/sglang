"""Unit tests for TokenizerManager auto-truncation token accounting.

No server or model loading: the manager is built with ``__new__`` and only the
attributes read by ``_validate_one_request`` are set.
"""

import unittest
from types import SimpleNamespace

from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

CONTEXT_LEN = 1024
NUM_RESERVED_TOKENS = 4


class TestAutoTruncateReservedTokens(CustomTestCase):
    def setUp(self):
        # With Eagle, compute_num_reserved_tokens() reserves output slots, so the
        # input budget is context_len minus those slots.
        override = get_context().override_server_args()
        override.install()
        self.addCleanup(override.restore)
        self.manager = TokenizerManager.__new__(TokenizerManager)
        self.manager.context_len = CONTEXT_LEN
        self.manager.num_reserved_tokens = NUM_RESERVED_TOKENS
        self.manager.allow_auto_truncate = True
        self.manager.validate_total_tokens = True
        self.manager.is_generation = True
        self.manager.model_config = SimpleNamespace(
            joint_head_config=None, pointer_head=None
        )

    def _validate(self, input_len, max_new_tokens):
        req = GenerateReqInput(
            input_ids=[1] * input_len,
            sampling_params={"max_new_tokens": max_new_tokens},
        )
        self.manager._validate_one_request(req, req.input_ids)
        return req

    def test_truncated_input_keeps_reserved_slots_out_of_budget(self):
        req = self._validate(CONTEXT_LEN - 1, max_new_tokens=8)
        used = (
            len(req.input_ids)
            + NUM_RESERVED_TOKENS
            + req.sampling_params["max_new_tokens"]
        )
        self.assertLessEqual(used, CONTEXT_LEN)

    def test_truncation_without_max_new_tokens_keeps_reserved_slots(self):
        req = self._validate(CONTEXT_LEN, max_new_tokens=None)
        self.assertLessEqual(len(req.input_ids) + NUM_RESERVED_TOKENS, CONTEXT_LEN)


if __name__ == "__main__":
    unittest.main()
