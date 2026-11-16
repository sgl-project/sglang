"""JSON encoding of the OpenAI `/v1/embeddings` route, checked without a server."""

import asyncio
import unittest
from types import SimpleNamespace

from fastapi import Response

from sglang.srt.entrypoints import http_server
from sglang.srt.entrypoints.openai.protocol import (
    EmbeddingObject,
    EmbeddingRequest,
    EmbeddingResponse,
    UsageInfo,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestEmbeddingResponses(CustomTestCase):
    def test_openai_embeddings_route_serializes_the_response_model_once(self):
        """The route hands back JSON bytes, so FastAPI's recursive
        jsonable_encoder never walks the float vectors."""
        response_model = EmbeddingResponse(
            data=[EmbeddingObject(embedding=[0.25, -0.5], index=0)],
            model="m",
            usage=UsageInfo(prompt_tokens=3, total_tokens=3),
        )

        class FakeServing:
            async def handle_request(self, request, raw_request):
                return response_model

        raw_request = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(openai_serving_embedding=FakeServing())
            )
        )
        response = asyncio.run(
            http_server.openai_v1_embeddings(
                EmbeddingRequest(input="hi", model="m"), raw_request
            )
        )

        self.assertIsInstance(response, Response)
        self.assertEqual(
            response.body, response_model.model_dump_json(by_alias=True).encode()
        )


if __name__ == "__main__":
    unittest.main()
