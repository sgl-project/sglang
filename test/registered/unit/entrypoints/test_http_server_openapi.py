"""Exercise production OpenAPI routes without lifespan, model loading, or inference."""

import json
import os
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi.responses import ORJSONResponse, StreamingResponse
from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator, ValidationError

from sglang.srt.entrypoints.http_server import app
from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    CompletionResponse,
    CompletionResponseChoice,
    CompletionResponseStreamChoice,
    CompletionStreamResponse,
    DeltaMessage,
    ErrorResponse,
    SglExt,
    UsageInfo,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

ROUTES = (
    (
        "/v1/chat/completions",
        "openai_serving_chat",
        {"messages": [{"role": "user", "content": "Hello"}]},
        ChatCompletionResponse,
        ChatCompletionStreamResponse,
    ),
    (
        "/v1/completions",
        "openai_serving_completion",
        {"prompt": "Hello"},
        CompletionResponse,
        CompletionStreamResponse,
    ),
)


class TestCompletionOpenAPI(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        """Read the actual app's schema without entering its serving lifespan."""
        super().setUpClass()
        cls.client = TestClient(app)
        cls.addClassCleanup(cls.client.close)
        response = cls.client.get("/openapi.json")
        response.raise_for_status()
        cls.schema = response.json()

    def _validate_payload(self, payload, response_schema):
        """Validate a wire payload with the published schema and its components."""
        Draft202012Validator(
            {**response_schema, "components": self.schema["components"]}
        ).validate(payload)

    def test_unary_models_and_extensions_are_declared(self):
        """Both JSON success contracts expose native usage and extension fields."""
        for path, _, _, model, _ in ROUTES:
            with self.subTest(path=path):
                responses = self.schema["paths"][path]["post"]["responses"]
                root = responses["200"]["content"]["application/json"]["schema"]
                self.assertEqual(root["$ref"], f"#/components/schemas/{model.__name__}")
                properties = self.schema["components"]["schemas"][model.__name__][
                    "properties"
                ]
                self.assertIn("usage", properties)
                self.assertIn("sglext", properties)

    def test_validation_status_and_other_errors(self):
        """Declare the native errors without promising one body for all failures."""
        for path, *_ in ROUTES:
            with self.subTest(path=path):
                responses = self.schema["paths"][path]["post"]["responses"]
                self.assertNotIn("422", responses)
                for status in ("400", "500"):
                    self.assertEqual(
                        responses[status]["content"]["application/json"]["schema"][
                            "$ref"
                        ],
                        "#/components/schemas/ErrorResponse",
                    )
                self.assertNotIn("content", responses["default"])

    def _event_schema(self, path):
        """Return the published parsed-event schema for a target route."""
        return self.schema["paths"][path]["post"]["responses"]["200"]["content"][
            "text/event-stream"
        ]["schema"]

    def _validate_event(self, path, event):
        """Validate a parsed event and explicitly check its decoded JSON content.

        Recognize the declared sentinel before JSON decoding. Raise ValueError
        for malformed JSON or ValidationError if no contentSchema accepts it;
        ordinary JSON Schema validation alone ignores contentSchema annotations.
        """
        schema = self._event_schema(path)
        self._validate_payload(event, schema)
        alternatives = schema["properties"]["data"]["anyOf"]
        data = event["data"]
        if any(data == branch.get("const") for branch in alternatives):
            return
        payload = json.loads(data)
        for branch in alternatives:
            if "contentSchema" not in branch:
                continue
            self.assertEqual(branch["contentMediaType"], "application/json")
            try:
                self._validate_payload(payload, branch["contentSchema"])
                return
            except ValidationError:
                pass
        raise ValidationError("Decoded SSE data matches no declared contentSchema")

    @staticmethod
    def _parse_fixture_events(wire):
        """Parse data-only UTF-8 test frames, joining data lines and ignoring comments.

        This fixture parser accepts CRLF/LF framing and discards an unfinished
        final frame. It is not a general SSE client or an event-order validator.
        """
        events = []
        data = []
        for line in wire.decode().replace("\r\n", "\n").splitlines():
            if not line:
                if data:
                    events.append({"data": "\n".join(data)})
                data = []
            elif line.startswith("data:"):
                value = line[5:]
                data.append(value[1:] if value.startswith(" ") else value)
        return events

    @staticmethod
    def _chunk(model, with_usage=False):
        """Build a native chunk fixture, optionally including usage and extensions."""
        choice = (
            ChatCompletionResponseStreamChoice(
                index=0, delta=DeltaMessage(content="Hello")
            )
            if model is ChatCompletionStreamResponse
            else CompletionResponseStreamChoice(index=0, text="Hello")
        )
        return model(
            id="stream-test",
            model="test-model",
            created=1,
            choices=[choice],
            usage=UsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=2)
            if with_usage
            else None,
            sglext=SglExt(input_ids=[1]) if with_usage else None,
        )

    def test_sse_references_native_payloads(self):
        """Expose native chunk refs and nested errors using the 3.1 event convention."""
        self.assertEqual(self.schema["openapi"], "3.1.0")
        for path, _, _, _, chunk_model in ROUTES:
            with self.subTest(path=path):
                media = self.schema["paths"][path]["post"]["responses"]["200"][
                    "content"
                ]["text/event-stream"]
                self.assertNotIn("itemSchema", media)
                schema = media["schema"]
                self.assertEqual(schema["type"], "object")
                self.assertEqual(schema["required"], ["data"])
                branches = schema["properties"]["data"]["anyOf"]
                self.assertEqual(branches[0], {"const": "[DONE]"})
                self.assertEqual(
                    branches[1]["contentSchema"],
                    {"$ref": f"#/components/schemas/{chunk_model.__name__}"},
                )
                error_schema = branches[2]["contentSchema"]
                self.assertEqual(error_schema["required"], ["error"])
                self.assertEqual(
                    error_schema["properties"]["error"],
                    {"$ref": "#/components/schemas/ErrorResponse"},
                )
                properties = self.schema["components"]["schemas"][chunk_model.__name__][
                    "properties"
                ]
                self.assertIn("choices", properties)
                self.assertIn("usage", properties)
                self.assertIn("sglext", properties)

    def test_sse_rejects_invalid_decoded_payloads(self):
        """Reject bad JSON, malformed chunks/errors, flat errors and false sentinels."""
        for path, _, _, _, model in ROUTES:
            chunk = json.loads(self._chunk(model).model_dump_json())
            wrong_choices = {**chunk, "choices": "not an array"}
            wrong_choice = {**chunk, "choices": [17]}
            missing_choices = {
                key: value for key, value in chunk.items() if key != "choices"
            }
            flat_error = ErrorResponse(
                message="bad", type="TestError", code=400
            ).model_dump()
            payloads = (
                wrong_choices,
                wrong_choice,
                missing_choices,
                flat_error,
                {"error": {}},
                {"error": "bad"},
                {},
                None,
            )
            for data in [
                *(json.dumps(value) for value in payloads),
                "not JSON",
                "[DONE] trailing",
            ]:
                with self.subTest(path=path, data=data):
                    with self.assertRaises((ValueError, ValidationError)):
                        self._validate_event(path, {"data": data})

    def test_sse_event_envelope(self):
        """Require string data and validate optional SSE envelope field types."""
        for path, *_ in ROUTES:
            self._validate_event(
                path, {"data": "[DONE]", "event": "message", "id": "1", "retry": 0}
            )
            for event in (
                {},
                {"data": {}},
                {"data": "[DONE]", "retry": -1},
                {"data": "[DONE]", "event": 1},
            ):
                with self.subTest(path=path, event=event):
                    with self.assertRaises(ValidationError):
                        self._validate_event(path, event)

    def test_reachable_references_resolve(self):
        """Resolve every reference reachable from either target operation."""
        pending = [self.schema["paths"][path]["post"] for path, *_ in ROUTES]
        visited = set()
        while pending:
            node = pending.pop()
            if isinstance(node, list):
                pending.extend(node)
            elif isinstance(node, dict):
                pending.extend(node.values())
                reference = node.get("$ref")
                if reference is not None and reference not in visited:
                    self.assertTrue(reference.startswith("#/"), reference)
                    visited.add(reference)
                    target = self.schema
                    for part in reference[2:].split("/"):
                        target = target[part.replace("~1", "/").replace("~0", "~")]
                    pending.append(target)

    def test_invalid_requests_use_the_declared_400_body(self):
        """Exercise production body and content-type validation before serving."""
        for path, *_ in ROUTES:
            for request in (
                {"json": {}},
                {"content": "{", "headers": {"content-type": "application/json"}},
                {"content": "{}", "headers": {"content-type": "text/plain"}},
            ):
                with self.subTest(path=path, request=request):
                    response = self.client.post(path, **request)
                    self.assertEqual(response.status_code, 400, response.text)
                    body = response.json()
                    self.assertEqual(body["code"], 400)
                    self.assertNotIn("error", body)
                    schema = self.schema["paths"][path]["post"]["responses"]["400"][
                        "content"
                    ]["application/json"]["schema"]
                    self._validate_payload(body, schema)

    def test_native_unary_serialization_matches_published_schema(self):
        """Keep native serialization and field omission through the real routes."""
        for path, service, request, model, _ in ROUTES:
            for with_extensions in (False, True):
                with self.subTest(path=path, with_extensions=with_extensions):
                    choice = (
                        ChatCompletionResponseChoice(
                            index=0,
                            message=ChatMessage(role="assistant", content="Hello"),
                            finish_reason="stop",
                        )
                        if model is ChatCompletionResponse
                        else CompletionResponseChoice(
                            index=0, text="Hello", finish_reason="stop"
                        )
                    )
                    result = model(
                        id="test-completion",
                        model="test-model",
                        created=1,
                        choices=[choice],
                        usage=UsageInfo(
                            prompt_tokens=1, completion_tokens=1, total_tokens=2
                        ),
                        sglext=SglExt(input_ids=[1]) if with_extensions else None,
                    )
                    serving = SimpleNamespace(
                        handle_request=AsyncMock(return_value=result)
                    )
                    with patch.object(app.state, service, serving, create=True):
                        response = self.client.post(path, json=request)
                    self.assertEqual(response.status_code, 200, response.text)
                    self.assertEqual(
                        response.content, result.model_dump_json(by_alias=True).encode()
                    )
                    schema = self.schema["paths"][path]["post"]["responses"]["200"][
                        "content"
                    ]["application/json"]["schema"]
                    self._validate_payload(response.json(), schema)

    def test_serving_errors_pass_through(self):
        """Route metadata must not rewrite serving-layer 400 or 500 responses."""
        for path, service, request, *_ in ROUTES:
            for status in (400, 500):
                with self.subTest(path=path, status=status):
                    error = ErrorResponse(
                        message="Test error", type="TestError", code=status
                    )
                    result = ORJSONResponse(error.model_dump(), status_code=status)
                    serving = SimpleNamespace(
                        handle_request=AsyncMock(return_value=result)
                    )
                    with patch.object(app.state, service, serving, create=True):
                        response = self.client.post(path, json=request)
                    self.assertEqual(response.status_code, status)
                    self.assertEqual(response.content, result.body)

    def test_stream_bytes_and_decoded_payloads(self):
        """Preserve real-route SSE bytes and validate parsed native fixtures.

        Serving results are mocked; comments and multiline data exercise the
        fixture parser, not model generation, timing, ordering or disconnects.
        """
        self.assertEqual(self._parse_fixture_events(b"data: unfinished\n"), [])
        for path, service, request, _, chunk_model in ROUTES:
            with self.subTest(path=path):
                plain = self._chunk(chunk_model)
                extended = self._chunk(chunk_model, with_usage=True)
                error = ErrorResponse(message="Test error", type="TestError", code=400)
                multiline = json.dumps(json.loads(extended.model_dump_json()), indent=2)
                wire = (
                    ": keep-alive\r\n\r\n"
                    f"data: {plain.model_dump_json()}\r\n\r\n"
                    + "".join(f"data: {line}\n" for line in multiline.splitlines())
                    + "\n"
                    + f"data: {json.dumps({'error': error.model_dump()})}\n\n"
                    + "data: [DONE]\n\n"
                ).encode()
                result = StreamingResponse(
                    iter([wire[:11], wire[11:]]), media_type="text/event-stream"
                )
                serving = SimpleNamespace(handle_request=AsyncMock(return_value=result))
                with patch.object(app.state, service, serving, create=True):
                    response = self.client.post(path, json={**request, "stream": True})
                self.assertEqual(response.status_code, 200)
                self.assertTrue(
                    response.headers["content-type"].startswith("text/event-stream")
                )
                self.assertEqual(response.content, wire)
                events = self._parse_fixture_events(response.content)
                self.assertEqual(len(events), 4)
                self.assertEqual(
                    json.loads(events[0]["data"]), json.loads(plain.model_dump_json())
                )
                self.assertEqual(
                    json.loads(events[1]["data"]),
                    json.loads(extended.model_dump_json()),
                )
                self.assertEqual(
                    json.loads(events[2]["data"]), {"error": error.model_dump()}
                )
                self.assertEqual(events[3], {"data": "[DONE]"})
                for event in events:
                    self._validate_event(path, event)

    def test_cached_schema_is_stable(self):
        """Repeated HTTP retrieval returns the same cached declarations."""
        self.assertEqual(self.client.get("/openapi.json").json(), self.schema)

    def test_export_can_be_disabled(self):
        """Check the import-time export switch in a fresh process."""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from sglang.srt.entrypoints.http_server import app; "
                "assert app.openapi_url is None; "
                "assert not any(getattr(r, 'path', '') in "
                "('/openapi.json', '/docs', '/redoc') for r in app.routes)",
            ],
            env={**os.environ, "DISABLE_OPENAPI_DOC": "true"},
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
