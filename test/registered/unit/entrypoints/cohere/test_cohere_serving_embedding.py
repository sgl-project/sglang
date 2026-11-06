"""HTTP/adapter tests using production routes and global exception handlers.

GPU-backed coverage lives in prefill_only/test_cohere_embedding.py.
"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast

from sglang.srt.entrypoints import http_server
from sglang.srt.entrypoints.cohere.serving_embedding import (
    CohereServingEmbedding,
    _load_sentence_transformer_prompts,
)
from sglang.srt.managers.embedding_truncation import truncate_embedding_input
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

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


def make_tokenizer():
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
            hf_config=SimpleNamespace(
                task_instructions={"query": "query: ", "passage": "passage: "}
            ),
        )
        self.tokenizer = make_tokenizer()
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
            kwargs = (
                {"add_special_tokens": item.embedding_add_special_tokens}
                if item.embedding_add_special_tokens is not None
                else {}
            )
            ids = self.tokenizer.encode(item.text, **kwargs)
            ids, _, _ = truncate_embedding_input(
                ids,
                truncate=item.embedding_truncate,
                max_tokens=item.embedding_max_tokens,
                input_token_limit=32,
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
            if getattr(route, "path", None)
            in ("/v2/embed", "/v1/embeddings", "/v1/messages")
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
        self.assertEqual(body["texts"], ["hello", "hello world"])
        self.assertEqual(body["response_type"], "embeddings_by_type")
        self.assertEqual(body["meta"]["api_version"], {"version": "2"})
        self.assertEqual(
            body["meta"]["billed_units"], {"input_tokens": 5, "image_tokens": 0}
        )
        self.assertEqual(len(self.manager.requests), 1)

    def test_plain_text_bypasses_template_and_preserves_default_special_tokens(self):
        self.manager.tokenizer.chat_template = CHAT_TEMPLATE
        response = self.post(texts=["hello"])
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.manager.requests[-1].text, ["hello"])
        self.assertEqual(self.manager.token_ids, [[1, 2]])

    def test_task_prefix_without_template_is_exact(self):
        self.assertEqual(
            self.post(texts=["hello"], input_type="query").status_code, 200
        )
        self.assertEqual(self.manager.requests[-1].text, ["query: hello"])

    def test_structured_system_prefix_and_no_duplicate_bos(self):
        # Iterating every role's content exposes string-vs-block system bugs.
        self.manager.tokenizer.chat_template = (
            "{% for m in messages %}{{m.role}}:"
            "{% for p in m.content %}{{p.text}}{% endfor %}|{% endfor %}"
        )
        response = self.post(texts=["hello"], input_type="query")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.manager.requests[-1].text, ["system:query: |user:hello|"])
        self.assertFalse(self.manager.requests[-1].embedding_add_special_tokens)
        self.assertNotEqual(self.manager.token_ids[0][0], 1)

    def test_string_template_receives_flattened_prefix(self):
        self.templates.jinja_template_content_format = "string"
        self.manager.tokenizer.chat_template = (
            "{% for m in messages %}{{m.content}}|{% endfor %}"
        )
        self.assertEqual(
            self.post(texts=["hello"], input_type="passage").status_code, 200
        )
        self.assertEqual(self.manager.requests[-1].text, ["passage: |hello|"])

    def test_empty_task_instruction_uses_raw_text(self):
        self.manager.model_config.hf_config.task_instructions["empty"] = ""
        self.manager.tokenizer.chat_template = CHAT_TEMPLATE
        self.assertEqual(
            self.post(texts=["hello"], input_type="empty").status_code, 200
        )
        self.assertEqual(self.manager.requests[-1].text, ["hello"])

    def test_optional_sentence_transformer_prompts(self):
        self.manager.model_config.hf_config.task_instructions = {"ignored": 123}
        (self.model_dir / "config_sentence_transformers.json").write_text(
            json.dumps({"prompts": {"search_query": "search: "}})
        )
        response = self.post(texts=["hello"], input_type="search_query")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.manager.requests[-1].text, ["search: hello"])

    def test_missing_malformed_and_nonobject_optional_metadata(self):
        self.manager.model_config.hf_config.task_instructions = None
        for contents in (None, "not json", "[]"):
            with self.subTest(contents=contents):
                _load_sentence_transformer_prompts.cache_clear()
                if contents is not None:
                    (self.model_dir / "config_sentence_transformers.json").write_text(
                        contents
                    )
                self.assert_error(self.post(texts=["hello"], input_type="query"))

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
        billed = response.json()["meta"]["billed_units"]
        self.assertEqual(billed["input_tokens"], 0)
        self.assertEqual(billed["image_tokens"], sum(map(len, self.manager.token_ids)))

    def test_inputs_text_only_still_uses_chat_path(self):
        payload = [{"content": [{"type": "text", "text": "hello"}]}]
        self.assert_error(self.post(inputs=payload))
        self.manager.tokenizer.chat_template = CHAT_TEMPLATE
        self.assertEqual(self.post(inputs=payload).status_code, 200)
        self.assertEqual(self.manager.requests[-1].text, ["user:hello|"])

    def test_incompatible_images_fail_before_inference(self):
        self.manager.model_config.is_multimodal = False
        self.assert_error(self.post(images=[IMAGE_A]))
        self.manager.model_config.is_multimodal = True
        self.assert_error(self.post(images=[IMAGE_A]))
        self.manager.tokenizer.chat_template = CHAT_TEMPLATE
        self.templates.jinja_template_content_format = "string"
        self.assert_error(self.post(images=[IMAGE_A]))
        self.assertFalse(self.manager.requests)

    def test_named_template_does_not_append_assistant(self):
        self.templates.chat_template_name = "test-template"
        conv = SimpleNamespace(
            messages=[["user", "hello"], ["assistant", None]],
            get_prompt=lambda: "named prompt",
        )
        with patch(
            "sglang.srt.entrypoints.cohere.serving_embedding.generate_chat_conv",
            return_value=conv,
        ) as build:
            response = self.post(texts=["hello"], input_type="query")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(conv.messages, [["user", "hello"]])
        self.assertEqual(build.call_args.args[0].messages[0].content[0].text, "query: ")

    def test_dimensions_forwarded_and_existing_model_validation_reused(self):
        response = self.post(texts=["hello"], output_dimension=8)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()["embeddings"]["float"][0]), 8)
        self.assert_error(self.post(texts=["hello"], output_dimension=4))
        self.assert_error(self.post(texts=["hello"], output_dimension=32))
        self.manager.model_config.is_matryoshka = False
        self.assert_error(self.post(texts=["hello"], output_dimension=8))

    def test_truncation_is_token_based_and_none_rejects(self):
        for policy, expected in (("END", [1, 2]), ("START", [2, 3])):
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

    def test_sentinel_zero_and_none_literal_budget(self):
        for policy in ("END", "START"):
            response = self.post(texts=["hello"], truncate=policy, max_tokens=-1)
            self.assertEqual(response.status_code, 200, response.text)
            self.assert_error(self.post(texts=["hello"], truncate=policy, max_tokens=0))
        for budget in (-1, 0):
            self.assert_error(
                self.post(texts=["hello"], truncate="NONE", max_tokens=budget)
            )
        self.assertEqual(
            self.post(texts=["hello"], truncate="NONE", max_tokens=100).status_code, 200
        )

    def test_priority_and_model_alias(self):
        response = self.client.post(
            "/v2/embed",
            json={
                "texts": ["hello"],
                "model": "embedding-model:adapter",
                "priority": 5,
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        request = self.manager.requests[-1]
        self.assertEqual(request.priority, 5)
        self.assertEqual(request.lora_path, ["adapter"])

    def test_schema_malformed_json_and_content_type_errors(self):
        for payload in (
            {},
            {"texts": []},
            {"texts": ["a"], "images": [IMAGE_A]},
            {"texts": ["a"], "truncate": "MIDDLE"},
            {"texts": ["a"], "output_dimension": 0},
            {"texts": ["a"], "embedding_types": ["int8"]},
        ):
            with self.subTest(payload=payload):
                self.assert_error(self.post(**payload))
        self.assert_error(
            self.client.post(
                "/v2/embed", content="{", headers={"content-type": "application/json"}
            )
        )
        self.assert_error(
            self.client.post(
                "/v2/embed",
                content='{"texts":["hello"]}',
                headers={"content-type": "text/plain"},
            )
        )
        self.assertFalse(self.manager.requests)

    def test_business_and_internal_errors_share_envelope(self):
        self.assert_error(self.post(texts=["hello"], model="unknown"), 404)
        self.assert_error(self.post(texts=["hello"], input_type="unknown"))
        self.manager.failure = ValueError("model rejected input")
        self.assert_error(self.post(texts=["hello"]))
        self.manager.failure = RuntimeError("test internal error")
        self.assert_error(self.post(texts=["hello"]), 500)

    def test_serialization_errors_keep_json_envelope(self):
        async def nonfinite_result(request, raw_request):
            yield [{"embedding": [float("nan")], "meta_info": {"prompt_tokens": 1}}]

        self.manager.generate_request = nonfinite_result
        error = self.assert_error(self.post(texts=["hello"]), 500)
        self.assertEqual(error["type"], "InternalServerError")

    def test_direct_route_uses_shared_orjson_parser(self):
        route = next(
            r for r in http_server.app.routes if getattr(r, "path", None) == "/v2/embed"
        )
        self.assertIs(type(route), http_server.ORJSONRoute)
        payload = b'{"texts":["hello"]}'
        with patch.object(
            http_server.orjson, "loads", wraps=http_server.orjson.loads
        ) as loads:
            response = self.client.post(
                "/v2/embed",
                content=payload,
                headers={"content-type": "application/json"},
            )
        self.assertEqual(response.status_code, 200, response.text)
        loads.assert_called_once_with(payload)

    def test_nonstandard_json_fails_before_serving(self):
        # Stdlib json accepts these constants, even in ignored fields; orjson
        # rejects them before schema validation or model execution.
        for constant in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(constant=constant):
                self.assert_error(
                    self.client.post(
                        "/v2/embed",
                        content='{"texts":["hello"],"unused":' + constant + "}",
                        headers={"content-type": "application/json"},
                    )
                )
        self.assertFalse(self.manager.requests)

    def test_global_errors_respect_deployment_prefix(self):
        self.app.root_path = "/gateway"
        self.assert_error(self.client.post("/gateway/v2/embed", json={"texts": []}))
        self.assert_error(
            self.client.post(
                "/gateway/v2/embed",
                content="{",
                headers={"content-type": "application/json"},
            )
        )
        self.assertFalse(self.manager.requests)

    def test_global_http_exception_keeps_cohere_envelope(self):
        for status in (400, 404, 503):
            with (
                self.subTest(status=status),
                patch.object(
                    self.serving,
                    "handle_request",
                    side_effect=HTTPException(status_code=status, detail="test detail"),
                ),
            ):
                error = self.assert_error(self.post(texts=["hello"]), status)
                if status >= 500:
                    self.assertEqual(error["message"], "Internal server error")
                    self.assertEqual(error["type"], "InternalServerError")

    def test_other_api_validation_formats_are_preserved(self):
        response = self.client.post("/v1/embeddings", json={})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["object"], "error")
        self.assertNotIn("error", response.json())
        response = self.client.post("/v1/messages", json={})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["type"], "error")
        self.assertEqual(response.json()["error"]["type"], "invalid_request_error")
        # Matching must be exact, not a broad startswith('/v2/embed').
        self.app.add_api_route(
            "/v2/embed-extra", http_server.cohere_v2_embed, methods=["POST"]
        )
        response = self.client.post("/v2/embed-extra", json={})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["object"], "error")
        self.assertNotIn("error", response.json())

    def test_app_cors_applies_to_success_validation_and_internal_errors(self):
        headers = {"origin": "https://example.com"}
        for payload, status in (({"texts": ["hello"]}, 200), ({"texts": []}, 400)):
            with self.subTest(status=status):
                response = self.client.post("/v2/embed", json=payload, headers=headers)
                self.assertEqual(response.status_code, status, response.text)
                self.assertIn(
                    response.headers["access-control-allow-origin"],
                    ("*", "https://example.com"),
                )

        async def nonfinite_result(request, raw_request):
            yield [{"embedding": [float("nan")], "meta_info": {"prompt_tokens": 1}}]

        self.manager.generate_request = nonfinite_result
        response = self.client.post(
            "/v2/embed", json={"texts": ["hello"]}, headers=headers
        )
        error = self.assert_error(response, 500)
        self.assertEqual(error["message"], "Internal server error")
        self.assertIn(
            response.headers["access-control-allow-origin"],
            ("*", "https://example.com"),
        )

    def test_other_api_unhandled_error_keeps_default_response(self):
        @self.app.get("/test-internal-error")
        async def fail():
            raise RuntimeError("test detail")

        response = self.client.get("/test-internal-error")
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.text, "Internal Server Error")
        self.assertEqual(response.headers["content-type"], "text/plain; charset=utf-8")

    def test_openapi_exposes_separate_request_and_response(self):
        schema = self.client.get("/openapi.json").json()
        endpoint = schema["paths"]["/v2/embed"]["post"]
        self.assertIn(
            "CohereEmbedRequest",
            endpoint["requestBody"]["content"]["application/json"]["schema"]["$ref"],
        )
        self.assertIn(
            "CohereEmbedResponse",
            endpoint["responses"]["200"]["content"]["application/json"]["schema"][
                "$ref"
            ],
        )


if __name__ == "__main__":
    unittest.main()
