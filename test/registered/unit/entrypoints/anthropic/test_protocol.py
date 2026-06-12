"""Tests for Anthropic API protocol models"""

import unittest

from pydantic import ValidationError

from sglang.srt.entrypoints.anthropic.protocol import AnthropicMessagesRequest
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=7, suite="base-a-test-cpu")


class TestAnthropicMessagesRequest(unittest.TestCase):
    def _make_request(self, **kwargs):
        data = {
            "model": "claude-test",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 16,
        }
        data.update(kwargs)
        return AnthropicMessagesRequest(**data)

    def test_temperature_range(self):
        self._make_request(temperature=0.0)
        self._make_request(temperature=1.0)
        self._make_request(temperature=None)

        with self.assertRaises(ValidationError):
            self._make_request(temperature=-0.1)

        with self.assertRaises(ValidationError):
            self._make_request(temperature=1.1)

    def test_top_p_range(self):
        self._make_request(top_p=0.0)
        self._make_request(top_p=1.0)
        self._make_request(top_p=None)

        with self.assertRaises(ValidationError):
            self._make_request(top_p=-0.1)

        with self.assertRaises(ValidationError):
            self._make_request(top_p=1.1)

    def test_top_k_range(self):
        self._make_request(top_k=0)
        self._make_request(top_k=500)
        self._make_request(top_k=None)

        with self.assertRaises(ValidationError):
            self._make_request(top_k=-1)

        with self.assertRaises(ValidationError):
            self._make_request(top_k=501)


if __name__ == "__main__":
    unittest.main()
