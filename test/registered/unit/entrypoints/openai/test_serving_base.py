"""
Unit-tests for OpenAIServingBase error-response shaping.
Run with either:
    python test/registered/unit/entrypoints/openai/test_serving_base.py -v
or
    pytest test/registered/unit/entrypoints/openai/test_serving_base.py -v
"""

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede any import that pulls in sgl_kernel

import json
import unittest
from unittest.mock import Mock

import orjson

from sglang.srt.entrypoints.openai.serving_base import OpenAIServingBase
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _StubServing(OpenAIServingBase):
    """Minimal concrete subclass -- both error helpers live on the base class."""

    def _request_id_prefix(self) -> str:
        return "stub-"

    def _convert_to_internal_request(self, request, raw_request=None):
        raise NotImplementedError


class TestCreateErrorResponse(unittest.TestCase):
    def setUp(self):
        tokenizer_manager = Mock()
        tokenizer_manager.server_args = Mock(
            tokenizer_metrics_allowed_custom_labels=None
        )
        self.serving = _StubServing(tokenizer_manager)

    @staticmethod
    def _body(response):
        return orjson.loads(response.body)

    def test_error_response_is_wrapped_in_error_envelope(self):
        response = self.serving.create_error_response(
            "max_tokens=999999 cannot be greater than the model context length",
            err_type="BadRequestError",
            status_code=400,
            param="max_tokens",
        )
        self.assertEqual(response.status_code, 400)

        body = self._body(response)
        # Clients built on the official OpenAI SDK read the failure out of
        # body["error"]; a flat body makes them report an empty error.
        self.assertIn("error", body)
        self.assertNotIn("message", body)

        error = body["error"]
        self.assertEqual(error["type"], "BadRequestError")
        self.assertEqual(error["param"], "max_tokens")
        self.assertEqual(error["code"], 400)
        self.assertIn("cannot be greater than", error["message"])

    def test_streaming_and_non_streaming_agree_on_shape(self):
        kwargs = dict(message="boom", err_type="BadRequestError", status_code=400)
        non_streaming = self._body(self.serving.create_error_response(**kwargs))
        streaming = json.loads(self.serving.create_streaming_error_response(**kwargs))

        self.assertEqual(non_streaming.keys(), streaming.keys())
        self.assertEqual(
            non_streaming["error"]["message"], streaming["error"]["message"]
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
