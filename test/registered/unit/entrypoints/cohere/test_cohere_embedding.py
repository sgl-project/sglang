"""CPU smoke tests for the Cohere-compatible /v2/embed endpoint: error envelope,
embedding encodings, request-scoped truncation, and the HTTP route.

GPU-backed parity with /v1/embeddings lives in prefill_only/test_embedding_models.py.
"""

import asyncio
import base64
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from fastapi import FastAPI, HTTPException
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from starlette.exceptions import HTTPException as StarletteHTTPException
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import PretrainedConfig, PreTrainedTokenizerFast

from sglang.srt.entrypoints import http_server
from sglang.srt.entrypoints.cohere import serving_embedding
from sglang.srt.entrypoints.cohere.encoding import (
    build_typed_embeddings,
)
from sglang.srt.entrypoints.cohere.protocol import (
    create_error_response,
)
from sglang.srt.entrypoints.cohere.serving_embedding import (
    CohereServingEmbedding,
    _load_sentence_transformer_prompts,
)
from sglang.srt.managers.async_dynamic_batch_tokenizer import AsyncDynamicbatchTokenizer
from sglang.srt.managers.embedding_truncation import (
    truncate_embedding_input,
)
from sglang.srt.managers.io_struct import EmbeddingReqInput
from sglang.srt.managers.req_types import MultimodalDataItem, MultimodalProcessorOutput
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.multimodal.modality import Modality
from sglang.srt.runtime_context import get_context
from sglang.srt.tokenizer.tiktoken_tokenizer import (
    DEFAULT_SPECIAL_TOKENS,
    TiktokenTokenizer,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


class TestCohereErrorResponse(unittest.TestCase):
    def test_status_selects_type_and_server_errors_hide_detail(self):
        for status, expected_type, expected_message in (
            (404, "NotFoundError", "detail"),
            (429, "BadRequestError", "detail"),
            (500, "InternalServerError", "Internal server error"),
            (503, "InternalServerError", "Internal server error"),
        ):
            with self.subTest(status=status):
                response = create_error_response(
                    "detail", status_code=status, param="model"
                )
                self.assertEqual(response.status_code, status)
                self.assertEqual(
                    json.loads(response.body)["error"],
                    {
                        "message": expected_message,
                        "type": expected_type,
                        "param": "model",
                        "code": status,
                    },
                )


class TestCohereEncoding(unittest.TestCase):
    def test_base64_is_little_endian_float32(self):
        result = build_typed_embeddings([[1.0, -2.5, 0.0]], ["base64"])
        # 1.0 = 0x3f800000; -2.5 = 0xc0200000; 0.0 = 0x00000000.
        self.assertEqual(
            base64.b64decode(result.base64[0]),
            bytes.fromhex("0000803f000020c000000000"),
        )
        self.assertEqual(set(result.model_dump(exclude_none=True)), {"base64"})

    def test_binary_sign_zero_and_msb_order(self):
        result = build_typed_embeddings(
            [
                [-1.0, 0.0, -0.0, 1.0, -1.0, -1.0, 1.0, -1.0],
                [-1.0] * 8,
                [1.0] * 8,
            ],
            ["binary", "ubinary"],
        )
        # 01110010 = 114; signed representation subtracts 128 (not int8 cast).
        self.assertEqual(result.ubinary, [[114], [0], [255]])
        self.assertEqual(result.binary, [[-14], [-128], [127]])


class TestEmbeddingTruncation(unittest.TestCase):
    def truncate(self, ids, policy="END", max_tokens=None, limit=8, **kwargs):
        return truncate_embedding_input(
            ids,
            truncate=policy,
            max_tokens=max_tokens,
            input_token_limit=limit,
            **kwargs,
        )

    @staticmethod
    def mm(ids, offsets, **kwargs):
        return MultimodalProcessorOutput(
            input_ids=list(ids),
            padded_input_ids=list(ids),
            token_type_ids=torch.zeros(len(ids), dtype=torch.long),
            mm_items=[
                MultimodalDataItem(modality=Modality.IMAGE, offsets=spans)
                for spans in offsets
            ],
            **kwargs,
        )

    def test_mm_rejects_cut_media_removed_media_and_closing_delimiter(self):
        ids = list(range(14))
        for budget in (2, 5, 8, 9):
            with self.subTest(budget=budget):
                mm = self.mm(ids, [[(1, 2), (4, 5)], [(7, 8)]])
                with self.assertRaisesRegex(ValueError, "remove media tokens"):
                    self.truncate(ids, max_tokens=budget, limit=20, mm_inputs=mm)
                self.assertEqual(mm.input_ids, ids)


class TestEmbeddingTruncationManager(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        override = get_context().override_server_args(
            chunked_prefill_size=-1, max_prefill_tokens=100
        )
        override.install()
        self.addCleanup(override.restore)
        self.manager = TokenizerManager.__new__(TokenizerManager)
        self.manager.context_len = 16
        self.manager.num_reserved_tokens = 1
        self.manager.max_req_input_len = 14
        self.manager.allow_auto_truncate = True
        self.manager.validate_total_tokens = False
        self.manager.is_generation = False
        self.manager.model_config = SimpleNamespace(
            joint_head_config=None,
            pointer_head=None,
            hf_config=SimpleNamespace(architectures=[]),
            is_embedding_gemma=False,
        )
        self.manager.mm_processor = None
        self.manager.tokenizer = Mock(spec=PreTrainedTokenizerFast)
        self.manager.async_dynamic_batch_tokenizer = None
        self.manager._create_tokenized_object = Mock(
            side_effect=lambda obj, text, ids, embeds, mm, types: SimpleNamespace(
                input_ids=ids, mm_inputs=mm, token_type_ids=types
            )
        )

    @staticmethod
    def request(**kwargs):
        obj = EmbeddingReqInput(**kwargs)
        obj.normalize_batch_and_arguments()
        return obj

    def async_tokenizer(self, tokenizer):
        async_tokenizer = AsyncDynamicbatchTokenizer(tokenizer)

        async def close():
            async_tokenizer._batcher_task.cancel()
            await asyncio.gather(async_tokenizer._batcher_task, return_exceptions=True)
            async_tokenizer._executor.shutdown()
            self.manager.async_dynamic_batch_tokenizer = None

        self.addAsyncCleanup(close)
        return async_tokenizer

    @staticmethod
    def text_tokenizer(template):
        vocabulary = {
            token: index
            for index, token in enumerate(
                ["[UNK]", "[BOS]", "[EOS]", "[SEP]", *list("abcdefghij")]
            )
        }
        backend = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
        backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        backend.post_processor = processors.TemplateProcessing(
            single=template,
            special_tokens=[
                (token, vocabulary[token]) for token in ("[BOS]", "[EOS]", "[SEP]")
            ],
        )
        return PreTrainedTokenizerFast(
            tokenizer_object=backend,
            unk_token="[UNK]",
            bos_token="[BOS]",
            eos_token="[EOS]",
            sep_token="[SEP]",
        )

    async def test_text_truncation_matches_tokenizer_for_single_batch_and_async(self):
        texts = ["a b [EOS] c d e f g", "h i j a b c d e"]
        for template, modes in (
            ("$A [EOS]", ("single",)),
            ("[BOS] $A [EOS]", ("single",)),
            ("[BOS] $A [EOS] [SEP]", ("single", "batch", "async")),
            ("$A", ("single",)),
        ):
            for policy, side in (("END", "right"), ("START", "left")):
                tokenizer = self.text_tokenizer(template)
                self.manager.tokenizer = tokenizer
                oracle = copy.deepcopy(tokenizer)
                oracle.truncation_side = side
                expected = oracle(texts, truncation=True, max_length=5)["input_ids"]
                for mode in modes:
                    with self.subTest(template=template, policy=policy, mode=mode):
                        self.manager.async_dynamic_batch_tokenizer = (
                            self.async_tokenizer(tokenizer) if mode == "async" else None
                        )
                        if mode == "batch":
                            results = await self.manager._batch_tokenize_and_process(
                                2,
                                self.request(
                                    text=texts,
                                    embedding_truncate=policy,
                                    embedding_max_tokens=5,
                                ),
                            )
                        else:
                            results = await asyncio.gather(
                                *(
                                    self.manager._tokenize_one_request(
                                        self.request(
                                            text=text,
                                            embedding_truncate=policy,
                                            embedding_max_tokens=5,
                                        )
                                    )
                                    for text in texts
                                )
                            )
                        self.assertEqual([r.input_ids for r in results], expected)
                        self.assertEqual(tokenizer.truncation_side, "right")

    async def test_tiktoken_text_is_tokenized_under_every_truncation_policy(self):
        """The tiktoken wrapper adds no special tokens; END (the /v2/embed default)
        and START must still tokenize plain text in the sync, batch, and async paths."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "tokenizer.json"
        path.write_text(
            json.dumps(
                {
                    "regular_tokens": [
                        {"bytes": [byte], "token": byte} for byte in range(256)
                    ],
                    "special_tokens": [
                        {"bytes": list(text.encode()), "token": 256 + index}
                        for index, text in enumerate(DEFAULT_SPECIAL_TOKENS)
                    ],
                    "word_split": "V1",
                }
            )
        )
        tokenizer = TiktokenTokenizer(str(path))
        self.manager.tokenizer = tokenizer
        texts = ["hello world", "world hello"]
        for policy, budget, kept in (
            (None, None, slice(None)),
            ("NONE", None, slice(None)),
            ("END", None, slice(None)),
            ("END", 5, slice(0, 5)),
            ("START", 5, slice(-5, None)),
        ):
            expected = [list(text.encode())[kept] for text in texts]
            for mode in ("single", "batch", "async"):
                with self.subTest(policy=policy, budget=budget, mode=mode):
                    self.manager.async_dynamic_batch_tokenizer = (
                        self.async_tokenizer(tokenizer) if mode == "async" else None
                    )
                    if mode == "batch":
                        results = await self.manager._batch_tokenize_and_process(
                            2,
                            self.request(
                                text=texts,
                                embedding_truncate=policy,
                                embedding_max_tokens=budget,
                            ),
                        )
                    else:
                        # Sequential awaits keep each async call a lone string.
                        results = [
                            await self.manager._tokenize_one_request(
                                self.request(
                                    text=text,
                                    embedding_truncate=policy,
                                    embedding_max_tokens=budget,
                                )
                            )
                            for text in texts
                        ]
                    self.assertEqual([r.input_ids for r in results], expected)


CHAT_TEMPLATE = (
    "{% for message in messages %}{{ message.role }}:"
    "{% if message.content is string %}{{ message.content }}"
    "{% else %}{% for part in message.content %}"
    "{% if part.type == 'text' %}{{ part.text }}"
    "{% elif part.type == 'image' %}[IMG]{% endif %}"
    "{% endfor %}{% endif %}|{% endfor %}"
)
IMAGE_A = "data:image/png;base64,YQ=="
IMAGE_B = "data:image/png;base64,Yg=="


def _bos_tokenizer():
    tokenizer = Tokenizer(
        models.WordLevel(
            {"[UNK]": 0, "[BOS]": 1, "hello": 2, "world": 3, "a": 4, "b": 5, "c": 6},
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[BOS] $A", special_tokens=[("[BOS]", 1)]
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="[UNK]", bos_token="[BOS]"
    )


class _InferenceDouble:
    def __init__(self, model_path):
        self.model_path = model_path
        self.served_model_name = "embedding-model"
        self.is_generation = False
        self.server_args = SimpleNamespace()
        self.request_logger = SimpleNamespace(log_requests=False)
        self.model_config = SimpleNamespace(
            model_path=model_path,
            revision=None,
            is_multimodal=True,
            is_matryoshka=True,
            matryoshka_dimensions=[8, 16],
            hidden_size=16,
            hf_config=PretrainedConfig(
                task_instructions={"query": "query: ", "passage": "passage: "}
            ),
        )
        self.tokenizer = _bos_tokenizer()
        self.text_tokenization = TokenizerManager.__new__(TokenizerManager)
        self.text_tokenization.tokenizer = self.tokenizer
        self.text_tokenization.async_dynamic_batch_tokenizer = None
        self.text_tokenization.model_config = SimpleNamespace(is_embedding_gemma=False)
        self.requests = []
        self.token_ids = []
        self.failure = None

    async def generate_request(self, request, raw_request):
        self.requests.append(request)
        if self.failure is not None:
            raise self.failure
        request.normalize_batch_and_arguments()
        results = []
        self.token_ids = []
        for index in range(request.batch_size):
            item = request[index]
            TokenizerManager._validate_for_matryoshka_dim(self, item)
            ids, _, mask = await self.text_tokenization._tokenize_texts(
                item.text, **TokenizerManager._embedding_tokenizer_kwargs(item)
            )
            ids, _, _ = truncate_embedding_input(
                ids,
                truncate=item.embedding_truncate,
                max_tokens=item.embedding_max_tokens,
                input_token_limit=32,
                special_tokens_mask=mask,
            )
            if not ids:
                raise ValueError("Embedding input cannot be empty after truncation.")
            self.token_ids.append(ids)
            # Deterministic distinct rows verify response order/encoding without
            # claiming that this test executes a model or decodes image bytes.
            results.append(
                {
                    "embedding": [float(index + 1)] * (item.dimensions or 16),
                    "meta_info": {"prompt_tokens": len(ids)},
                }
            )
        yield results


class TestCohereEmbeddingHTTP(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.model_dir = Path(directory.name)
        self.manager = _InferenceDouble(str(self.model_dir))
        self.templates = SimpleNamespace(
            chat_template_name=None,
            python_chat_template=None,
            jinja_template_content_format="openai",
        )
        self.serving = CohereServingEmbedding(self.manager, self.templates)
        self.app = FastAPI(
            middleware=[
                middleware
                for middleware in http_server.app.user_middleware
                if middleware.cls is http_server.CORSMiddleware
            ]
        )
        self.app.router.route_class = http_server.ORJSONRoute
        # Reuse the actual registered routes and exception handlers, without
        # starting the production lifespan or modifying its shared app state.
        self.app.router.routes.extend(
            route
            for route in http_server.app.routes
            if isinstance(route, APIRoute)
            and route.path in ("/v2/embed", "/v1/embeddings", "/v1/messages")
        )
        self.app.exception_handlers.update(http_server.app.exception_handlers)
        self.app.state.cohere_serving_embedding = self.serving
        # Inspect the response a real HTTP client sees, including the default
        # 500 response on unrelated APIs, without propagating server exceptions.
        self.client = TestClient(self.app, raise_server_exceptions=False)
        self.addCleanup(self.client.close)
        self.addCleanup(_load_sentence_transformer_prompts.cache_clear)

    def post(self, **payload):
        return self.client.post("/v2/embed", json=payload)

    def assert_error(self, response, status=400):
        self.assertEqual(response.status_code, status, response.text)
        error = response.json()["error"]
        self.assertEqual(error["code"], status)
        self.assertIn("message", error)
        self.assertIsNone(error["param"])
        return error

    def test_text_batch_response_encodings_and_single_inference(self):
        response = self.post(
            texts=["hello", "hello world"],
            embedding_types=["float", "base64", "binary", "ubinary"],
        )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(
            set(body["embeddings"]), {"float", "base64", "binary", "ubinary"}
        )
        self.assertEqual(body["embeddings"]["float"], [[1.0] * 16, [2.0] * 16])
        self.assertEqual(body["embeddings"]["ubinary"], [[255, 255], [255, 255]])
        self.assertEqual(body["embeddings"]["binary"], [[127, 127], [127, 127]])
        self.assertEqual(len(body["embeddings"]["base64"]), 2)
        self.assertEqual(body["texts"], ["hello", "hello world"])
        self.assertEqual(body["response_type"], "embeddings_by_type")
        self.assertEqual(body["meta"]["api_version"], {"version": "2"})
        self.assertEqual(
            body["meta"]["billed_units"], {"input_tokens": 5, "image_tokens": 0}
        )
        self.assertEqual(len(self.manager.requests), 1)

    def test_image_batch_and_mixed_inputs_keep_sample_boundaries(self):
        self.manager.tokenizer.chat_template = CHAT_TEMPLATE
        response = self.post(images=[IMAGE_A, IMAGE_B])
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.manager.requests[-1].image_data, [[IMAGE_A], [IMAGE_B]])
        self.assertNotIn("texts", response.json())
        response = self.post(
            inputs=[
                {
                    "content": [
                        {"type": "text", "text": "before"},
                        {"type": "image_url", "image_url": {"url": IMAGE_A}},
                        {"type": "text", "text": "middle"},
                        {"type": "image_url", "image_url": {"url": IMAGE_B}},
                    ]
                },
                {"content": [{"type": "text", "text": "hello"}]},
            ]
        )
        self.assertEqual(response.status_code, 200, response.text)
        request = self.manager.requests[-1]
        self.assertEqual(request.text, ["user:before[IMG]middle[IMG]|", "user:hello|"])
        self.assertEqual(request.image_data, [[IMAGE_A, IMAGE_B], None])
        self.assertEqual(request[0].image_data, [IMAGE_A, IMAGE_B])
        self.assertIsNone(request[1].image_data)
        self.assertEqual(len(response.json()["embeddings"]["float"]), 2)
        self.assertNotIn("texts", response.json())
        billed = response.json()["meta"]["billed_units"]
        self.assertEqual(billed["input_tokens"], 0)
        self.assertEqual(billed["image_tokens"], sum(map(len, self.manager.token_ids)))

    def test_truncation_is_token_based_and_none_rejects(self):
        for policy, expected in (("END", [1, 2]), ("START", [1, 3])):
            with self.subTest(policy=policy):
                response = self.post(
                    texts=["hello world"], truncate=policy, max_tokens=2
                )
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(self.manager.token_ids, [expected])
        self.assert_error(
            self.post(texts=["hello world"], truncate="NONE", max_tokens=2)
        )
        self.assert_error(self.post(texts=["hello"], max_tokens=33))

    def test_nonfinite_float_output_is_an_error_but_base64_carries_it(self):
        async def nonfinite_result(request, raw_request):
            yield [{"embedding": [float("nan")], "meta_info": {"prompt_tokens": 1}}]

        self.manager.generate_request = nonfinite_result
        for embedding_types in (["float"], ["float", "base64"]):
            error = self.assert_error(
                self.post(texts=["hello"], embedding_types=embedding_types), 500
            )
            self.assertEqual(error["type"], "InternalServerError")
        response = self.post(texts=["hello"], embedding_types=["base64"])
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["embeddings"], {"base64": ["AADAfw=="]})

    def test_dependency_http_errors_keep_status_headers_and_envelope(self):
        """An HTTP error raised by a route dependency keeps its status and headers;
        Starlette's base class must not be reported as a 500."""
        # The copied route resolves dependency overrides on the production app.
        overrides = http_server.app.dependency_overrides
        self.addCleanup(overrides.pop, http_server.validate_json_request, None)
        for failure, error_type, message in (
            (StarletteHTTPException(400, "client detail"), "BadRequestError", None),
            (HTTPException(400, "client detail"), "BadRequestError", None),
            (
                StarletteHTTPException(
                    503, "backend detail", headers={"Retry-After": "7"}
                ),
                "InternalServerError",
                "Internal server error",
            ),
        ):

            async def dependency(failure=failure):
                raise failure

            overrides[http_server.validate_json_request] = dependency
            with (
                self.subTest(failure=failure),
                patch.object(serving_embedding.logger, "error") as log_error,
            ):
                response = self.client.post(
                    "/v2/embed",
                    json={"texts": ["hello"]},
                    headers={"origin": "https://example.com"},
                )
                error = self.assert_error(response, failure.status_code)
                self.assertEqual(error["type"], error_type)
                self.assertEqual(error["message"], message or "client detail")
                self.assertEqual(
                    response.headers.get("retry-after"),
                    (failure.headers or {}).get("Retry-After"),
                )
                self.assertIn(
                    response.headers["access-control-allow-origin"],
                    ("*", "https://example.com"),
                )
                self.assertEqual(log_error.call_count, failure.status_code >= 500)
                if log_error.called:
                    self.assertIn("backend detail", log_error.call_args.args[0])
        self.assertFalse(self.manager.requests)


if __name__ == "__main__":
    unittest.main()
