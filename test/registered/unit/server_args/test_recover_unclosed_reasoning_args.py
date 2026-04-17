"""Unit tests for the unclosed-reasoning recovery configuration: server CLI
flags + per-request override on ChatCompletionRequest.
"""

import unittest
from unittest.mock import patch

from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.server_args import ServerArgs, prepare_server_args
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN,
    CustomTestCase,
)

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")

_mock_device = patch("sglang.srt.server_args.get_device", return_value="cuda")
_mock_device.start()


class TestServerArgsDefaults(CustomTestCase):
    def test_defaults_are_off(self):
        # All three new fields default to safe / off values.
        s = ServerArgs(model_path="dummy")
        self.assertFalse(s.redirect_unclosed_reasoning)
        self.assertEqual(s.redirect_eos_prob_threshold, 0.5)
        self.assertFalse(s.auto_recover_unclosed_reasoning)


class TestServerArgsCli(CustomTestCase):
    def _parse(self, *extra):
        return prepare_server_args(
            ["--model-path", DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN, *extra]
        )

    def test_redirect_flag_is_parsed(self):
        s = self._parse("--redirect-unclosed-reasoning")
        self.assertTrue(s.redirect_unclosed_reasoning)

    def test_threshold_flag_is_parsed(self):
        s = self._parse("--redirect-eos-prob-threshold", "0.75")
        self.assertAlmostEqual(s.redirect_eos_prob_threshold, 0.75)

    def test_auto_recover_flag_is_parsed(self):
        s = self._parse("--auto-recover-unclosed-reasoning")
        self.assertTrue(s.auto_recover_unclosed_reasoning)

    def test_all_flags_combined(self):
        s = self._parse(
            "--redirect-unclosed-reasoning",
            "--redirect-eos-prob-threshold",
            "0.6",
            "--auto-recover-unclosed-reasoning",
        )
        self.assertTrue(s.redirect_unclosed_reasoning)
        self.assertAlmostEqual(s.redirect_eos_prob_threshold, 0.6)
        self.assertTrue(s.auto_recover_unclosed_reasoning)


class TestChatCompletionRequestField(CustomTestCase):
    """Covers the new per-request `recover_unclosed_reasoning` override."""

    _MIN_REQ = {
        "model": "test",
        "messages": [{"role": "user", "content": "hi"}],
    }

    def test_default_is_none(self):
        req = ChatCompletionRequest(**self._MIN_REQ)
        self.assertIsNone(req.recover_unclosed_reasoning)

    def test_explicit_true(self):
        req = ChatCompletionRequest(**self._MIN_REQ, recover_unclosed_reasoning=True)
        self.assertTrue(req.recover_unclosed_reasoning)

    def test_explicit_false(self):
        req = ChatCompletionRequest(**self._MIN_REQ, recover_unclosed_reasoning=False)
        self.assertFalse(req.recover_unclosed_reasoning)

    def test_field_round_trips_through_model_dump(self):
        req = ChatCompletionRequest(**self._MIN_REQ, recover_unclosed_reasoning=True)
        # Pydantic v2 model_dump() / json round-trip preserves the field.
        as_dict = req.model_dump()
        self.assertIn("recover_unclosed_reasoning", as_dict)
        self.assertTrue(as_dict["recover_unclosed_reasoning"])


if __name__ == "__main__":
    unittest.main()
