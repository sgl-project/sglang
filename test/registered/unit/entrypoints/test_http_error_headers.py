"""HTTP errors retain headers and route-specific bodies, including mounted apps."""

import asyncio
import json
import unittest
from types import SimpleNamespace

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.exceptions import RequestValidationError  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from sglang.srt.entrypoints.http_server import app as serving_app  # noqa: E402
from sglang.srt.entrypoints.openai.serving_base import OpenAIServingBase  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Payload(BaseModel):
    count: int


class TestHTTPErrorHeaders(CustomTestCase):
    def test_http_exception_headers_and_existing_bodies_survive_mounts(self):
        for prefix in ("", "/mounted"):
            for path in (
                "/v1/chat/completions",
                "/v1/responses",
                "/v1/messages",
                "/generate",
            ):
                with self.subTest(prefix=prefix, path=path):
                    inner = FastAPI()
                    inner.add_exception_handler(
                        HTTPException, serving_app.exception_handlers[HTTPException]
                    )

                    async def reject():
                        raise HTTPException(
                            429, "wait", headers={"Retry-After": "7", "Allow": "POST"}
                        )

                    inner.add_api_route(path, reject, methods=["POST"])
                    outer = FastAPI()
                    outer.mount(prefix or "/", inner)
                    with TestClient(outer) as client:
                        response = client.post(prefix + path)
                    self.assertEqual(response.status_code, 429)
                    self.assertEqual(response.headers.get("retry-after"), "7")
                    self.assertEqual(response.headers.get("allow"), "POST")
                    body = response.json()
                    if path == "/v1/messages":
                        self.assertEqual(
                            body,
                            {
                                "type": "error",
                                "error": {
                                    "type": "rate_limit_error",
                                    "message": "wait",
                                },
                            },
                        )
                    elif path == "/v1/responses":
                        self.assertEqual(body["error"]["code"], 429)
                        self.assertEqual(body["error"]["type"], "Too Many Requests")
                    else:
                        self.assertEqual(body["object"], "error")
                        self.assertEqual(body["code"], 429)
                        self.assertEqual(body["type"], "429")

    def test_mounted_validation_uses_the_existing_protocol_body(self):
        for path in ("/v1/messages", "/v1/responses", "/v1/systemone"):
            with self.subTest(path=path):
                inner = FastAPI()
                inner.add_exception_handler(
                    RequestValidationError,
                    serving_app.exception_handlers[RequestValidationError],
                )

                async def validate(payload: _Payload):
                    return {"count": payload.count}

                inner.add_api_route(path, validate, methods=["POST"])
                outer = FastAPI()
                outer.mount("/mounted", inner)
                with TestClient(outer) as client:
                    response = client.post("/mounted" + path, json={"count": "invalid"})
                body = response.json()
                if path == "/v1/systemone":
                    self.assertEqual(response.status_code, 422)
                    self.assertIn("detail", body)
                else:
                    self.assertEqual(response.status_code, 400)
                    self.assertIn("error", body)
                    expected = (
                        "invalid_request_error"
                        if path == "/v1/messages"
                        else "Bad Request"
                    )
                    self.assertEqual(body["error"]["type"], expected)

    def test_serving_handler_retains_exception_headers(self):
        def reject(*args):
            raise HTTPException(
                429, "wait", headers={"Retry-After": "7", "Allow": "POST"}
            )

        handler = SimpleNamespace(
            _validate_request=lambda request: None,
            tokenizer_manager=SimpleNamespace(
                request_logger=SimpleNamespace(log_requests=False)
            ),
            _convert_to_internal_request=reject,
        )
        handler.create_error_response = OpenAIServingBase.create_error_response.__get__(
            handler
        )
        response = asyncio.run(
            OpenAIServingBase.handle_request(
                handler, SimpleNamespace(stream=False), None
            )
        )
        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.headers.get("retry-after"), "7")
        self.assertEqual(response.headers.get("allow"), "POST")
        self.assertEqual(json.loads(response.body)["code"], 429)


if __name__ == "__main__":
    unittest.main()
