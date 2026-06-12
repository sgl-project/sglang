"""Unit tests for tokenizer_manager._validate_one_request truncation behavior
when allow_auto_truncate is enabled. A too-long input should leave room for
max_new_tokens and num_reserved_tokens so the request can still produce
output, instead of being truncated to exactly context_len which forces
max_new_tokens to 0.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


def _make_mock(context_len=100, num_reserved_tokens=0, allow_auto_truncate=True):
    mock = MagicMock(spec=TokenizerManager)
    mock.context_len = context_len
    mock.num_reserved_tokens = num_reserved_tokens
    mock.validate_total_tokens = True
    mock.is_generation = True
    mock.server_args = SimpleNamespace(
        allow_auto_truncate=allow_auto_truncate,
        enable_return_hidden_states=False,
    )
    return mock


def _validate(mock, obj, input_ids):
    """Invoke the unbound _validate_one_request against a mock self."""
    return TokenizerManager._validate_one_request(mock, obj, input_ids)


def _gen_req(max_new_tokens):
    obj = GenerateReqInput(text="x", sampling_params={"max_new_tokens": max_new_tokens})
    return obj


class TestAutoTruncateLeavesRoomForGeneration(CustomTestCase):
    def test_long_input_leaves_room_for_max_new_tokens(self):
        # context_len=100, max_new_tokens=10. After truncation the input
        # should be 90 tokens, and max_new_tokens should remain 10 — the
        # second check (input + max_new_tokens >= context_len) must not
        # force max_new_tokens to 0.
        mock = _make_mock(context_len=100, num_reserved_tokens=0)
        obj = _gen_req(max_new_tokens=10)
        input_ids = [1] * 200

        _validate(mock, obj, input_ids)

        self.assertEqual(len(input_ids), 90)
        self.assertEqual(obj.sampling_params["max_new_tokens"], 10)

    def test_long_input_with_reserved_tokens(self):
        # EAGLE-style reserved tokens must be subtracted from the truncation
        # cap so the post-truncation total stays within context_len.
        mock = _make_mock(context_len=100, num_reserved_tokens=20)
        obj = _gen_req(max_new_tokens=10)
        input_ids = [1] * 200

        _validate(mock, obj, input_ids)

        # 100 - 10 - 20 = 70
        self.assertEqual(len(input_ids), 70)
        self.assertEqual(obj.sampling_params["max_new_tokens"], 10)

    def test_long_input_without_max_new_tokens(self):
        # When max_new_tokens is unset, the truncation should still succeed
        # and not strip the input below 1 token.
        mock = _make_mock(context_len=100, num_reserved_tokens=0)
        obj = GenerateReqInput(text="x", sampling_params={})
        input_ids = [1] * 200

        _validate(mock, obj, input_ids)

        self.assertEqual(len(input_ids), 100)

    def test_short_input_unchanged(self):
        # An input that already fits should not be touched.
        mock = _make_mock(context_len=100, num_reserved_tokens=0)
        obj = _gen_req(max_new_tokens=10)
        input_ids = [1] * 30

        _validate(mock, obj, input_ids)

        self.assertEqual(len(input_ids), 30)
        self.assertEqual(obj.sampling_params["max_new_tokens"], 10)

    def test_long_input_raises_when_truncate_disabled(self):
        mock = _make_mock(context_len=100, allow_auto_truncate=False)
        obj = _gen_req(max_new_tokens=10)
        input_ids = [1] * 200

        with self.assertRaises(ValueError):
            _validate(mock, obj, input_ids)

    def test_truncation_target_clamped_to_one_when_max_new_tokens_huge(self):
        # If max_new_tokens >= context_len, the cap would go negative; we
        # clamp to 1 so we still have at least one input token (the second
        # check then drops max_new_tokens to fit).
        mock = _make_mock(context_len=100, num_reserved_tokens=0)
        obj = _gen_req(max_new_tokens=200)
        input_ids = [1] * 500

        _validate(mock, obj, input_ids)

        self.assertEqual(len(input_ids), 1)
        self.assertEqual(obj.sampling_params["max_new_tokens"], 99)


if __name__ == "__main__":
    unittest.main(verbosity=3)
