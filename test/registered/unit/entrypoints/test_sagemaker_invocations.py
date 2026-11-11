"""SageMaker /invocations: dispatch by body key to the chat, completions and
embeddings routes, checked against fake serving handlers without a server."""

import unittest

from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from sglang.srt.entrypoints import http_server
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_HANDLERS = (
    "openai_serving_chat",
    "openai_serving_completion",
    "openai_serving_embedding",
)
_CHAT_BODY = {"messages": [{"role": "user", "content": "hi"}]}


class _RecordingHandler:
    def __init__(self, name, response=None):
        self.name = name
        self.response = response
        self.requests = []

    async def handle_request(self, request, raw_request):
        self.requests.append(request)
        if self.response is not None:
            return self.response
        return {"handler": self.name, "request": type(request).__name__}


class TestSageMakerInvocations(CustomTestCase):
    def setUp(self):
        state = http_server.app.state
        self._saved = {name: getattr(state, name, None) for name in _HANDLERS}
        self.handlers = {name: _RecordingHandler(name) for name in _HANDLERS}
        for name, handler in self.handlers.items():
            setattr(state, name, handler)
        self.client = TestClient(http_server.app)

    def tearDown(self):
        for name, handler in self._saved.items():
            setattr(http_server.app.state, name, handler)

    def post(self, **kwargs):
        return self.client.post("/invocations", **kwargs)

    def test_messages_takes_precedence(self):
        """A chat body that also carries `prompt` or `input` must keep reaching chat."""
        response = self.post(json={**_CHAT_BODY, "prompt": "p", "input": "i"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"handler": "openai_serving_chat", "request": "ChatCompletionRequest"},
        )

    def test_routes_by_key(self):
        for body, handler, request in (
            ({"prompt": "hi"}, "openai_serving_completion", "CompletionRequest"),
            ({"input": "hi"}, "openai_serving_embedding", "EmbeddingRequest"),
        ):
            with self.subTest(body=body):
                response = self.post(json=body)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(
                    response.json(), {"handler": handler, "request": request}
                )

    def test_unknown_body_is_rejected_without_echo(self):
        for body in ({"query": "secret-query", "documents": ["d"]}, {}):
            with self.subTest(body=body):
                response = self.post(json=body)
                self.assertEqual(response.status_code, 400)
                self.assertNotIn("secret-query", response.text)
                self.assertTrue(all(not h.requests for h in self.handlers.values()))

    def test_non_object_body_is_400(self):
        """Bodies that are not a JSON object used to reach the chat model and return 500."""
        for kwargs in (
            {"content": b'{"messages": []}', "headers": {"content-type": "text/plain"}},
            {"json": [1]},
        ):
            with self.subTest(kwargs=kwargs):
                self.assertEqual(self.post(**kwargs).status_code, 400)
        response = self.post(
            json=_CHAT_BODY, headers={"content-type": "application/vnd.api+json"}
        )
        self.assertEqual(response.status_code, 200)

    def test_invalid_field_is_400_on_that_field(self):
        response = self.post(json={"prompt": 1.5})
        self.assertEqual(response.status_code, 400)
        self.assertIn("'body', 'prompt'", response.json()["message"])

    def test_streaming_passes_through(self):
        """A stream from the chosen route must reach the client as-is, not be re-serialized."""

        async def chunks():
            yield b"data: one\n\n"
            yield b"data: [DONE]\n\n"

        self.handlers["openai_serving_completion"].response = StreamingResponse(
            chunks(), media_type="text/event-stream"
        )
        response = self.post(json={"prompt": "hi", "stream": True})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(
            response.headers["content-type"].startswith("text/event-stream")
        )
        self.assertEqual(response.content, b"data: one\n\ndata: [DONE]\n\n")


if __name__ == "__main__":
    unittest.main(verbosity=2)
