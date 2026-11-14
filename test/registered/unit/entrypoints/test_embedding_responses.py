"""JSON encoding of embedding routes, checked without a server."""

import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import Response

from sglang.srt.entrypoints import http_server
from sglang.srt.entrypoints.cohere.encoding import build_embed_response
from sglang.srt.entrypoints.cohere.protocol import CohereEmbedRequest
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

    def _cohere_response(self, request, response_model):
        raw_request = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(
                    cohere_serving_embedding=SimpleNamespace(
                        handle_request=AsyncMock(return_value=response_model)
                    )
                )
            )
        )
        return asyncio.run(http_server.cohere_v2_embed(request, raw_request))

    def test_cohere_response_preserves_formats_and_metadata(self):
        request = CohereEmbedRequest(
            texts=["hello", "文档🙂"],
            embedding_types=["float", "base64", "binary", "ubinary"],
        )
        response_model = build_embed_response(
            request,
            [
                {
                    "embedding": [-1.0, 0.0, -0.0, 1.0, 3.25, -4.5, 1e-9, -1e-9],
                    "meta_info": {"prompt_tokens": 3},
                },
                {"embedding": [0.25] * 8, "meta_info": {"prompt_tokens": 5}},
            ],
            request_id="embd-fixed",
        )
        response = self._cohere_response(request, response_model)
        self.assertIsInstance(response, Response)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.media_type, "application/json")
        body = json.loads(response.body)
        self.assertEqual(
            body["embeddings"],
            {
                "float": response_model.embeddings.float,
                "base64": response_model.embeddings.base64,
                "binary": response_model.embeddings.binary,
                "ubinary": response_model.embeddings.ubinary,
            },
        )
        self.assertEqual(body["id"], "embd-fixed")
        self.assertEqual(body["texts"], request.texts)
        self.assertEqual(body["response_type"], "embeddings_by_type")
        self.assertEqual(
            body["meta"],
            {
                "api_version": {"version": "2"},
                "billed_units": {"input_tokens": 8, "image_tokens": 0},
            },
        )

    def test_cohere_nan_preserves_error_behavior(self):
        for requested, status_code in (
            (["float", "base64"], 500),
            (["base64"], 200),
        ):
            with self.subTest(embedding_types=requested):
                request = CohereEmbedRequest(texts=["hello"], embedding_types=requested)
                response_model = build_embed_response(
                    request,
                    [
                        {
                            "embedding": [float("nan")] + [0.25] * 7,
                            "meta_info": {"prompt_tokens": 3},
                        },
                    ],
                    request_id="embd-fixed",
                )
                with patch.object(http_server.logger, "exception"):
                    response = self._cohere_response(request, response_model)
                self.assertEqual(response.status_code, status_code)
                self.assertEqual(response.media_type, "application/json")
                body = json.loads(response.body)
                if status_code == 500:
                    self.assertEqual(
                        body,
                        {
                            "error": {
                                "message": "Internal server error",
                                "type": "InternalServerError",
                                "param": None,
                                "code": 500,
                            }
                        },
                    )
                else:
                    self.assertEqual(
                        body["embeddings"],
                        {"base64": response_model.embeddings.base64},
                    )


if __name__ == "__main__":
    unittest.main()
