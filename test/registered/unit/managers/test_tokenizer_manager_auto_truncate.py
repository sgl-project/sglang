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

CONTEXT_LEN = 100


class TestAutoTruncate(CustomTestCase):
    def setUp(self):
        override = get_context().override_server_args()
        override.install()
        self.addCleanup(override.restore)

    def _manager(self, num_reserved_tokens=0):
        manager = TokenizerManager.__new__(TokenizerManager)
        manager.context_len = CONTEXT_LEN
        manager.num_reserved_tokens = num_reserved_tokens
        manager.allow_auto_truncate = True
        manager.validate_total_tokens = True
        manager.is_generation = True
        manager.model_config = SimpleNamespace(
            joint_head_config=None, pointer_head=None
        )
        return manager

    def _validate(self, manager, input_len, sampling_params):
        req = GenerateReqInput(
            input_ids=[1] * input_len, sampling_params=sampling_params
        )
        manager._validate_one_request(req, req.input_ids)
        return req

    def test_truncated_input_leaves_room_for_max_new_tokens(self):
        req = self._validate(self._manager(), 200, {"max_new_tokens": 10})
        self.assertEqual(len(req.input_ids), 90)
        self.assertEqual(req.sampling_params["max_new_tokens"], 10)

    def test_truncated_input_leaves_room_for_reserved_tokens(self):
        req = self._validate(
            self._manager(num_reserved_tokens=20), 200, {"max_new_tokens": 10}
        )
        self.assertEqual(len(req.input_ids), 70)
        self.assertEqual(req.sampling_params["max_new_tokens"], 10)

    def test_truncation_without_max_new_tokens_keeps_reserved_slots_in_budget(self):
        req = self._validate(self._manager(num_reserved_tokens=4), 200, {})
        self.assertLessEqual(len(req.input_ids) + 4, CONTEXT_LEN)

    def test_prompt_that_fits_is_kept_and_max_new_tokens_clamped(self):
        # Only an overflowing prompt is truncated; a prompt that fits stays
        # intact and the total-token check clamps max_new_tokens instead.
        req = self._validate(self._manager(), 99, {"max_new_tokens": 50})
        self.assertEqual(len(req.input_ids), 99)
        self.assertEqual(req.sampling_params["max_new_tokens"], 1)

    def test_oversized_max_new_tokens_keeps_one_input_token(self):
        req = self._validate(self._manager(), 200, {"max_new_tokens": 500})
        self.assertEqual(len(req.input_ids), 1)


if __name__ == "__main__":
    unittest.main()
