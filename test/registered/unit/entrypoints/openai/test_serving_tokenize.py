from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import unittest
from types import SimpleNamespace

from fastapi import FastAPI, Request
from fastapi.responses import ORJSONResponse
from fastapi.testclient import TestClient

from sglang.srt.entrypoints.openai.protocol import TokenizeRequest
from sglang.srt.entrypoints.openai.serving_tokenize import OpenAIServingTokenize
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class SentinelTokenizer:
    model_max_length = int(1e30)

    def encode(self, text, add_special_tokens=True):
        return [101, 7592, 102] if add_special_tokens else [7592]


class OpenAIServingTokenizeTest(CustomTestCase):
    def test_serializes_effective_context_length(self):
        tokenizer_manager = SimpleNamespace(
            tokenizer=SentinelTokenizer(),
            model_config=SimpleNamespace(context_len=65_536),
            server_args=SimpleNamespace(),
            request_logger=SimpleNamespace(log_requests=False),
        )
        serving = OpenAIServingTokenize(tokenizer_manager)
        app = FastAPI()
        app.state.openai_serving_tokenize = serving

        @app.post("/v1/tokenize", response_class=ORJSONResponse)
        async def tokenize(request: TokenizeRequest, raw_request: Request):
            return await raw_request.app.state.openai_serving_tokenize.handle_request(
                request, raw_request
            )

        with TestClient(app) as client:
            response = client.post("/v1/tokenize", json={"prompt": "hello"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "tokens": [101, 7592, 102],
                "count": 3,
                "max_model_len": 65_536,
            },
        )


if __name__ == "__main__":
    unittest.main()
