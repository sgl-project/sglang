"""HTTP and SSE errors use the same OpenAI envelope, including mounted apps."""

import json
import unittest

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


class TestOpenAIErrorEnvelopes(CustomTestCase):
    def test_serving_http_and_streaming_errors_have_the_same_body(self):
        response = OpenAIServingBase.create_error_response(
            None, "busy", status_code=503
        )
        stream = OpenAIServingBase.create_streaming_error_response(
            None, "busy", status_code=503
        )
        self.assertEqual(response.status_code, 503)
        self.assertEqual(json.loads(response.body), json.loads(stream))
        self.assertEqual(
            json.loads(stream)["error"],
            {
                "message": "busy",
                "type": "server_error",
                "param": None,
                "code": "service_unavailable",
            },
        )

    def test_mounted_validation_and_http_errors_keep_the_wire_contract(self):
        inner = FastAPI()
        for exc in (HTTPException, RequestValidationError):
            inner.add_exception_handler(exc, serving_app.exception_handlers[exc])

        @inner.get("/v1/models")
        def models():
            raise HTTPException(429, "busy", headers={"Retry-After": "3"})

        @inner.post("/v1/chat/completions")
        def chat(payload: _Payload):
            return payload

        @inner.post("/v1/systemone")
        def systemone(payload: _Payload):
            return payload

        outer = FastAPI()
        outer.mount("/gateway", inner)
        with TestClient(outer) as client:
            response = client.get("/gateway/v1/models")
            self.assertEqual(response.status_code, 429)
            self.assertEqual(response.headers["retry-after"], "3")
            self.assertEqual(response.json()["error"]["type"], "rate_limit_error")
            response = client.post("/gateway/v1/chat/completions", json={})
            self.assertEqual(response.status_code, 400)
            self.assertEqual(response.json()["error"]["param"], "count")
            response = client.post("/gateway/v1/systemone", json={})
            self.assertEqual(response.status_code, 422)
            self.assertIsInstance(response.json()["detail"], list)


if __name__ == "__main__":
    unittest.main()
