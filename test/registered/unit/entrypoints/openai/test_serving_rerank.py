"""CPU unit tests for /v1/rerank backend routing with structured text content.

Regression: a query or document given as a list of text parts (an accepted
RerankContent representation) must not be routed to the VL decoder under a
Qwen3 text reranker template. The VL renderer only provides query/document
variables, so the messages-based text template raises UndefinedError and the
request is rejected before scoring. Only content that actually carries
image/video parts is a VL signal on the request side.
"""

import asyncio
import json
import math
import pathlib
import unittest
from types import SimpleNamespace

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede any import that pulls in sgl_kernel

from starlette.requests import Request as StarletteRequest

from sglang.srt.entrypoints.openai.protocol import V1RerankReqInput
from sglang.srt.entrypoints.openai.serving_rerank import OpenAIServingRerank
from sglang.srt.managers.io_struct import EmbeddingReqInput
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[5]
TEXT_TEMPLATE = (_REPO_ROOT / "examples/chat_template/qwen3_reranker.jinja").read_text()
VL_TEMPLATE = (
    _REPO_ROOT / "examples/chat_template/qwen3_vl_reranker.jinja"
).read_text()

Q = "What is machine learning?"
D = ["ML is a branch of AI.", "Paris is in France."]


def _text_part(text):
    return {"type": "text", "text": text}


def _image_part(url="https://example.invalid/image.png"):
    return {"type": "image_url", "image_url": {"url": url}}


def _video_part(url="https://example.invalid/video.mp4"):
    return {"type": "video_url", "video_url": {"url": url}}


class _RecordingTokenizerManager:
    """Minimal TokenizerManager double: only inference/token-ID lookup is faked."""

    def __init__(
        self, template, model_path="Qwen/Qwen3-Reranker-0.6B", generation=True
    ):
        self.server_args = object()
        self.model_config = SimpleNamespace(
            model_path=model_path, is_generation=generation
        )
        self.request_logger = SimpleNamespace(log_requests=False)
        self.tokenizer = SimpleNamespace(
            chat_template=template,
            encode=lambda text, **_: [9693 if text == "yes" else 2152],
        )
        self.prompts = []
        self.generated = []
        self.pairs = []

    async def score_prompts(self, prompts, **kwargs):
        self.prompts.extend(prompts)
        assert kwargs["label_token_ids"] == [9693, 2152]
        assert kwargs["apply_softmax"] is False
        return SimpleNamespace(scores=[[0.2, 0.8], [0.9, 0.1]][: len(prompts)])

    async def generate_request(self, req, raw):
        if isinstance(req, EmbeddingReqInput):
            self.pairs.extend(req.text)
            yield [
                {"embedding": [x], "meta_info": {"id": str(i)}}
                for i, x in enumerate([0.2, 0.9][: len(req.text)])
            ]
        else:
            self.generated.append(
                dict(
                    text=req.text,
                    image_data=req.image_data,
                    video_data=req.video_data,
                )
            )
            yield {
                "meta_info": {
                    "output_top_logprobs": [
                        [(math.log(0.8), 9693, "yes"), (math.log(0.2), 2152, "no")]
                    ]
                }
            }


def _raw_request():
    return StarletteRequest(
        {"type": "http", "method": "POST", "path": "/v1/rerank", "headers": []}
    )


async def _run(
    query,
    documents,
    template=TEXT_TEMPLATE,
    model_path="Qwen/Qwen3-Reranker-0.6B",
    generation=True,
    **options,
):
    request = V1RerankReqInput.model_validate(
        dict(query=query, documents=documents, **options)
    )
    tm = _RecordingTokenizerManager(
        template, model_path=model_path, generation=generation
    )
    handler = OpenAIServingRerank(tm)
    result = await handler.handle_request(request, _raw_request())
    if isinstance(result, list):
        return 200, [r.model_dump() for r in result], tm
    return result.status_code, json.loads(result.body), tm


class TestRerankStructuredTextRouting(CustomTestCase):
    def test_plain_string_baseline(self):
        status, body, _ = asyncio.run(_run(Q, D))
        self.assertEqual(status, 200)
        self.assertEqual([(r["index"], r["score"]) for r in body], [(1, 0.9), (0, 0.2)])
        self.assertEqual([r["document"] for r in body], [D[1], D[0]])

    def test_structured_query_default_return_matches_string(self):
        status, body, tm = asyncio.run(_run([_text_part(Q)], D))
        self.assertEqual(status, 200)
        self.assertEqual([(r["index"], r["score"]) for r in body], [(1, 0.9), (0, 0.2)])
        self.assertEqual([r["document"] for r in body], [D[1], D[0]])

    def test_structured_documents_return_documents_false(self):
        status, body, _ = asyncio.run(
            _run(
                Q,
                [[_text_part(D[0])], D[1]],
                return_documents=False,
            )
        )
        self.assertEqual(status, 200)
        self.assertTrue(all("document" not in r for r in body))
        self.assertEqual([(r["index"], r["score"]) for r in body], [(1, 0.9), (0, 0.2)])

    def test_multi_part_query_and_document_text_preserved(self):
        status, body, tm = asyncio.run(
            _run(
                [_text_part("What is"), _text_part("machine learning?")],
                [[_text_part("ML is a"), _text_part("branch of AI.")], D[1]],
                return_documents=False,
            )
        )
        self.assertEqual(status, 200)
        # Existing extraction joins text parts with a single space.
        self.assertIn(Q, tm.prompts[0])
        self.assertIn(D[0], tm.prompts[0])

    def test_top_n_and_instruct_preserved(self):
        status, body, tm = asyncio.run(
            _run([_text_part(Q)], D, top_n=1, instruct="Find evidence.")
        )
        self.assertEqual(status, 200)
        self.assertEqual(len(body), 1)
        self.assertEqual(body[0]["index"], 1)
        self.assertTrue(all("<Instruct>: Find evidence." in p for p in tm.prompts))

    def test_prompts_identical_between_string_and_text_parts(self):
        _, _, tm_plain = asyncio.run(_run(Q, D, return_documents=False))
        _, _, tm_parts = asyncio.run(
            _run([_text_part(Q)], [[_text_part(d)] for d in D], return_documents=False)
        )
        self.assertEqual(tm_parts.prompts, tm_plain.prompts)

    def test_document_media_not_scored_as_text_or_dropped(self):
        # Media in a document with the text template is unsupported: it must
        # keep failing (no silent text scoring, no URL silently discarded).
        status, body, tm = asyncio.run(
            _run(Q, [[_text_part(D[0]), _image_part()], D[1]])
        )
        self.assertEqual(status, 400)
        self.assertEqual(tm.prompts, [])

    def test_query_media_not_scored_as_text(self):
        status, body, tm = asyncio.run(_run([_text_part(Q), _video_part()], D))
        self.assertEqual(status, 400)
        self.assertEqual(tm.prompts, [])

    def test_document_media_kept_on_vl_path(self):
        # Structured documents default to return_documents=True, which fails on
        # an independent response-schema issue (RerankResponse.document is
        # str-only). Route/media preservation is checked with documents off.
        status, body, tm = asyncio.run(
            _run(
                Q,
                [[_text_part(D[0]), _image_part()], D[1]],
                template=VL_TEMPLATE,
                model_path="local-vl",
                return_documents=False,
            )
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            tm.generated[0]["image_data"], ["https://example.invalid/image.png"]
        )
        self.assertIn("<|image_pad|>", tm.generated[0]["text"])

    def test_video_media_kept_on_vl_path(self):
        status, body, tm = asyncio.run(
            _run(
                [_text_part(Q), _video_part()],
                D,
                template=VL_TEMPLATE,
                model_path="local-vl",
            )
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            tm.generated[0]["video_data"], ["https://example.invalid/video.mp4"]
        )
        self.assertIn("<|video_pad|>", tm.generated[0]["text"])

    def test_vl_template_routes_string_request_to_vl(self):
        # Template/model precedence is unchanged: a plain string request under
        # the VL template stays on the VL decoder.
        status, body, tm = asyncio.run(
            _run(Q, D, template=VL_TEMPLATE, model_path="local-vl")
        )
        self.assertEqual(status, 200)
        self.assertEqual(len(tm.generated), len(D))

    def test_vl_model_path_not_downgraded_to_text_decoder(self):
        # A qwen3-vl model path with the text template is a misconfiguration
        # (docs require the VL template), but model-path precedence must be
        # preserved: the request stays on the VL path instead of being scored
        # by the text decoder.
        status, body, tm = asyncio.run(
            _run([_text_part(Q)], D, model_path="/models/Qwen3-VL-Reranker-2B")
        )
        self.assertEqual(status, 400)
        self.assertEqual(tm.prompts, [])
        self.assertEqual(tm.generated, [])

    def test_cross_encoder_extracts_structured_text(self):
        status, body, tm = asyncio.run(
            _run(
                [_text_part(Q)],
                D,
                template="",
                model_path="bge-reranker",
                generation=False,
            )
        )
        self.assertEqual(status, 200)
        self.assertEqual(tm.pairs, [[Q, d] for d in D])

    def test_generation_guard_still_enforced(self):
        status, body, _ = asyncio.run(_run([_text_part(Q)], D, generation=False))
        self.assertEqual(status, 400)
        self.assertIn("--is-embedding", body["message"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
