"""Routes the OpenAI API defines answer errors in OpenAI's ``{"error": {...}}`` envelope."""

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede any import that pulls in sgl_kernel

import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient
from starlette.requests import Request

from sglang.srt.entrypoints import http_server
from sglang.srt.entrypoints.openai.protocol import (
    is_openai_api_route,
    openai_error_body,
)
from sglang.srt.entrypoints.openai.serving_base import OpenAIServingBase
from sglang.srt.utils.auth import AuthLevel, add_api_key_middleware, auth_level
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

# HTTP status -> OpenAI error.type
EXPECTED_TYPES = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "invalid_request_error",
    422: "invalid_request_error",
    429: "rate_limit_error",
    500: "server_error",
    501: "server_error",
    503: "service_unavailable_error",
}


class _Serving(OpenAIServingBase):
    openai_error_envelope = True

    def __init__(self, convert_error=None):
        tokenizer_manager = Mock(server_args=None)
        tokenizer_manager.request_logger.log_requests = False
        super().__init__(tokenizer_manager)
        self._convert_error = convert_error

    def _request_id_prefix(self):
        return "test-"

    def _convert_to_internal_request(self, request, raw_request=None):
        raise self._convert_error


class _NativeServing(_Serving):
    openai_error_envelope = False


def _request(path, root_path=""):
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": root_path + path,
            "root_path": root_path,
            "headers": [],
        }
    )


def _handle(exc_type, path, exc, root_path=""):
    handler = http_server.app.exception_handlers[exc_type]
    return asyncio.run(handler(_request(path, root_path), exc))


def _assert_envelope(test, body, message, error_type, param=None, code=None):
    test.assertEqual(
        body,
        {
            "error": {
                "message": message,
                "type": error_type,
                "param": param,
                "code": code,
            }
        },
    )


class TestOpenAIErrorEnvelope(CustomTestCase):
    def test_error_response_uses_envelope_and_keeps_status(self):
        serving = _Serving()
        for status, error_type in EXPECTED_TYPES.items():
            with self.subTest(status=status):
                response = serving.create_error_response(
                    "bad", err_type="BadRequestError", status_code=status, param="n"
                )
                self.assertEqual(response.status_code, status)
                _assert_envelope(
                    self, json.loads(response.body), "bad", error_type, param="n"
                )

    def test_streaming_error_frame_uses_same_envelope(self):
        serving = _Serving()
        for status, error_type in EXPECTED_TYPES.items():
            with self.subTest(status=status):
                frame = serving.create_streaming_error_response(
                    "bad", "SERVICE_UNAVAILABLE", status
                )
                _assert_envelope(self, json.loads(frame), "bad", error_type)

    def test_sglang_routes_keep_flat_error_response(self):
        serving = _NativeServing()
        response = serving.create_error_response(
            "bad", err_type="BadRequestError", status_code=400
        )
        self.assertEqual(
            json.loads(response.body),
            {
                "object": "error",
                "message": "bad",
                "type": "BadRequestError",
                "param": None,
                "code": 400,
            },
        )
        frame = json.loads(serving.create_streaming_error_response("bad"))
        self.assertEqual(frame["error"]["type"], "BadRequestError")
        self.assertEqual(frame["error"]["code"], 400)

    def test_openai_api_routes(self):
        for path in (
            "/v1/chat/completions",
            "/v1/completions",
            "/v1/embeddings",
            "/v1/audio/transcriptions",
            "/v1/models",
            "/v1/models/served",
            "/invocations",
        ):
            with self.subTest(path=path):
                self.assertTrue(is_openai_api_route(path))
        for path in ("/generate", "/v1/score", "/v1/tokenize", "/v1/completionsx"):
            with self.subTest(path=path):
                self.assertFalse(is_openai_api_route(path))

    def test_internal_error_does_not_leak_exception_text(self):
        serving = _Serving(convert_error=RuntimeError("secret /srv/path"))
        response = asyncio.run(
            serving.handle_request(SimpleNamespace(), _request("/v1/x"))
        )
        self.assertEqual(response.status_code, 500)
        _assert_envelope(
            self, json.loads(response.body), "Internal server error", "server_error"
        )

    def test_http_exception_handler_uses_envelope(self):
        exc = HTTPException(status_code=503, detail="The request queue is full.")
        for root_path in ("", "/sglang"):
            with self.subTest(root_path=root_path):
                response = _handle(
                    HTTPException, "/v1/chat/completions", exc, root_path
                )
                self.assertEqual(response.status_code, 503)
                _assert_envelope(
                    self,
                    json.loads(response.body),
                    "The request queue is full.",
                    "service_unavailable_error",
                )

    def test_openai_route_servings_opt_in(self):
        from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
        from sglang.srt.entrypoints.openai.serving_completions import (
            OpenAIServingCompletion,
        )
        from sglang.srt.entrypoints.openai.serving_embedding import (
            OpenAIServingEmbedding,
        )
        from sglang.srt.entrypoints.openai.serving_transcription import (
            OpenAIServingTranscription,
        )

        for serving in (
            OpenAIServingChat,
            OpenAIServingCompletion,
            OpenAIServingEmbedding,
            OpenAIServingTranscription,
        ):
            with self.subTest(serving=serving.__name__):
                self.assertTrue(serving.openai_error_envelope)
        self.assertFalse(OpenAIServingBase.openai_error_envelope)
        from sglang.srt.entrypoints.openai.serving_responses import (
            OpenAIServingResponses,
        )

        self.assertFalse(OpenAIServingResponses.openai_error_envelope)

    def test_missing_message_falls_back_to_status_phrase(self):
        self.assertEqual(openai_error_body(None, 404)["error"]["message"], "Not Found")
        self.assertEqual(
            openai_error_body(None, 432)["error"]["message"], "Request failed"
        )

    def test_http_exception_detail_is_a_string(self):
        for detail, message in (
            ({"reason": "bad"}, "{'reason': 'bad'}"),
            (None, "Bad Request"),
        ):
            with self.subTest(detail=detail):
                response = _handle(
                    HTTPException,
                    "/v1/completions",
                    HTTPException(status_code=400, detail=detail),
                )
                _assert_envelope(
                    self, json.loads(response.body), message, "invalid_request_error"
                )

    def test_http_exception_on_native_route_keeps_flat_shape(self):
        response = _handle(
            HTTPException,
            "/generate",
            HTTPException(status_code=503, detail="The request queue is full."),
        )
        body = json.loads(response.body)
        self.assertEqual(body["object"], "error")
        self.assertEqual(body["message"], "The request queue is full.")
        self.assertEqual(body["code"], 503)

    def test_transcription_response_format_error_uses_envelope(self):
        response = asyncio.run(
            http_server.openai_v1_audio_transcriptions(
                _request("/v1/audio/transcriptions"), file=Mock(), response_format="srt"
            )
        )
        self.assertEqual(response.status_code, 400)
        _assert_envelope(
            self,
            json.loads(response.body),
            "Only 'json', 'text', and 'verbose_json' formats supported",
            "invalid_request_error",
            param="response_format",
        )

    def test_validation_error_handler_uses_envelope(self):
        handler = http_server.app.exception_handlers[RequestValidationError]
        exc = RequestValidationError(
            [{"loc": ("body", "n"), "msg": "bad n", "type": "value_error"}]
        )
        # /invocations shares the chat serving, so it answers in the same shape.
        for path in ("/v1/chat/completions", "/invocations"):
            with self.subTest(path=path):
                response = asyncio.run(handler(_request(path), exc))
                self.assertEqual(response.status_code, 400)
                self.assertIn("bad n", json.loads(response.body)["error"]["message"])
        response = asyncio.run(handler(_request("/v1/chat/completions"), exc))
        self.assertEqual(response.status_code, 400)
        error = json.loads(response.body)["error"]
        self.assertEqual(error["type"], "invalid_request_error")
        self.assertIn("bad n", error["message"])
        self.assertIsNone(error["param"])
        self.assertIsNone(error["code"])

        native = json.loads(asyncio.run(handler(_request("/generate"), exc)).body)
        self.assertEqual(native["object"], "error")
        self.assertEqual(native["code"], 400)

    def test_unknown_model_404_uses_envelope(self):
        state = SimpleNamespace(
            tokenizer_manager=SimpleNamespace(served_model_name="served")
        )
        with patch.object(http_server, "_global_state", state):
            response = asyncio.run(http_server.retrieve_model("other"))
        self.assertEqual(response.status_code, 404)
        _assert_envelope(
            self,
            json.loads(response.body),
            "The model 'other' does not exist",
            "invalid_request_error",
            param="model",
            code="model_not_found",
        )

    def test_api_key_middleware_rejection_uses_envelope(self):
        app = FastAPI()

        @app.get("/v1/models")
        def models():
            return {"object": "list", "data": []}

        @app.get("/get_model_info")
        def model_info():
            return {}

        @app.post("/flush_cache")
        @auth_level(AuthLevel.ADMIN_FORCE)
        def flush_cache():
            return {}

        add_api_key_middleware(app, api_key="secret", admin_api_key=None)
        client = TestClient(app)

        response = client.get("/v1/models")
        self.assertEqual(response.status_code, 401)
        _assert_envelope(self, response.json(), "Unauthorized", "authentication_error")
        ok = client.get("/v1/models", headers={"Authorization": "Bearer secret"})
        self.assertEqual(ok.status_code, 200)

        # SGLang's own routes keep their existing rejection bodies.
        response = client.get("/get_model_info")
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json(), {"error": "Unauthorized"})
        response = client.post(
            "/flush_cache", headers={"Authorization": "Bearer secret"}
        )
        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json(), {"error": "Forbidden"})


if __name__ == "__main__":
    unittest.main()
