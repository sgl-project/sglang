"""Regression tests for the sglang.lang frontend, run against stub backends."""

import contextlib
import copy
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import patch

import httpx

import sglang as sgl
from sglang.lang.backend import openai as openai_backend
from sglang.lang.backend.anthropic import Anthropic
from sglang.lang.backend.runtime_endpoint import RuntimeEndpoint
from sglang.lang.interpreter import StreamExecutor
from sglang.lang.ir import SglSamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class _SglangServerHandler(BaseHTTPRequestHandler):
    """Serves the routes an SGLang server has; every other route is a 404."""

    def _reply(self, status, payload=None):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(payload or {}).encode())

    def do_GET(self):  # noqa: N802
        if self.path == "/get_model_info":
            self._reply(200, {"model_path": "stub-model"})
        else:
            self._reply(404)

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length)) if length else None
        self.server.posts.append((self.path, body))
        if self.path == "/generate":
            self._reply(200, {"text": "", "meta_info": {}})
        else:
            self._reply(404)

    def log_message(self, fmt, *args):
        pass


class _StubServerTestCase(CustomTestCase):
    def setUp(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _SglangServerHandler)
        self.server.posts = []
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"


class TestRuntimeEndpointJoin(_StubServerTestCase):
    def test_concate_and_append_join_appends_the_fork_texts(self):
        """join(mode="concate_and_append") must work against an SGLang server, which
        has no route for merging the forks' KV caches."""

        @sgl.function
        def program(s):
            s += "A"
            forks = s.fork(2)
            forks += lambda i: f"[{i}]"
            forks.join(mode="concate_and_append")

        state = program.run(backend=RuntimeEndpoint(self.base_url))

        self.assertIsNone(state.error())
        self.assertEqual(state.text(), "A[0][1]")


_USAGE = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}


def _sse(chunks):
    lines = [f"data: {json.dumps(chunk)}\n\n" for chunk in chunks]
    return "".join(lines + ["data: [DONE]\n\n"]).encode()


class _FakeTiktoken:
    """Stands in for the encoding tiktoken would download."""

    _mergeable_ranks = {b"7": 7}
    _special_tokens = {"<|endoftext|>": 0}

    def decode(self, token_ids):
        return "7"

    def encode(self, text):
        return [7]


class TestOpenAIBackendTokenLimit(CustomTestCase):
    """Every OpenAI call must carry the one output-length limit its API takes: the
    completions API has no max_completion_tokens parameter, and a chat call must
    not send max_tokens and max_completion_tokens together."""

    def setUp(self):
        self.bodies = []

    def _handle(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.bodies.append(body)
        is_chat = request.url.path.endswith("/chat/completions")
        base = {"id": "c", "created": 0, "model": body["model"]}
        if body.get("stream"):
            kind = "chat.completion.chunk" if is_chat else "text_completion"
            piece = {"delta": {"content": "7"}} if is_chat else {"text": "7"}
            chunks = [
                {**base, "object": kind, "choices": [{"index": 0, **piece}]},
                {**base, "object": kind, "choices": [], "usage": _USAGE},
            ]
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, content=_sse(chunks)
            )
        if is_chat:
            message = {"role": "assistant", "content": "A: yes"}
            choice = {"index": 0, "message": message, "finish_reason": "stop"}
            kind = "chat.completion"
        else:
            choice = {"index": 0, "text": "7", "finish_reason": "stop"}
            kind = "text_completion"
        return httpx.Response(
            200, json={**base, "object": kind, "choices": [choice], "usage": _USAGE}
        )

    def _backend(self, model_name, is_chat_model):
        fake_tiktoken = SimpleNamespace(encoding_for_model=lambda _: _FakeTiktoken())
        with patch.object(openai_backend, "tiktoken", fake_tiktoken):
            return openai_backend.OpenAI(
                model_name,
                is_chat_model=is_chat_model,
                api_key="test",
                max_retries=0,
                http_client=httpx.Client(transport=httpx.MockTransport(self._handle)),
            )

    def _assert_only_max_tokens(self, expected):
        body = self.bodies[-1]
        self.assertEqual(body["max_tokens"], expected)
        self.assertNotIn("max_completion_tokens", body)

    def test_typed_gen_on_a_completion_model(self):
        backend = self._backend("gpt-3.5-turbo-instruct", is_chat_model=False)
        for dtype in (int, str):
            with self.subTest(dtype=dtype):

                @sgl.function
                def program(s):
                    s += "n=" + sgl.gen("n", dtype=dtype, max_tokens=4)

                state = program.run(backend=backend, use_thread=False)

                self.assertIn("7", state["n"])
                self._assert_only_max_tokens(4)

    def test_stream_on_completion_and_chat_models(self):
        for model_name, is_chat_model in [
            ("gpt-3.5-turbo-instruct", False),
            ("gpt-4", True),
        ]:
            with self.subTest(model_name=model_name):
                backend = self._backend(model_name, is_chat_model=is_chat_model)
                executor = StreamExecutor(
                    backend, {}, SglSamplingParams(), None, stream=False, use_thread=False
                )
                executor.submit(sgl.user("q"))
                executor.submit(sgl.assistant_begin())

                pieces = backend.generate_stream(
                    executor, sampling_params=SglSamplingParams(max_new_tokens=4)
                )

                self.assertEqual("".join(text for text, _ in pieces), "7")
                self._assert_only_max_tokens(4)

    def test_api_speculative_execution_on_a_chat_model(self):
        backend = self._backend("gpt-4", is_chat_model=True)

        @sgl.function(num_api_spec_tokens=8)
        def program(s):
            s += sgl.user("q")
            s += sgl.assistant("A: " + sgl.gen("v", max_tokens=4, stop="\n"))

        state = program.run(backend=backend, use_thread=False)

        self.assertEqual(state["v"], "yes")
        self._assert_only_max_tokens(8)


class _FakeAnthropicMessages:
    """Records what each call was sent and answers "hi"."""

    def __init__(self):
        self.calls = []

    def _record(self, system, messages):
        self.calls.append({"system": system, "messages": copy.deepcopy(messages)})

    def create(self, *, model, system, messages, **kwargs):
        self._record(system, messages)
        return SimpleNamespace(content=[SimpleNamespace(text="hi")])

    @contextlib.contextmanager
    def stream(self, *, model, system, messages, **kwargs):
        self._record(system, messages)
        yield SimpleNamespace(text_stream=iter(["h", "i"]))


class TestAnthropicBackendSystemMessage(CustomTestCase):
    """The system prompt must stay in the program's chat history, so every turn of
    a conversation is sent with it and state.messages() still shows it."""

    SYSTEM = {"role": "system", "content": "be brief"}

    def setUp(self):
        self.backend = Anthropic("claude-test", api_key="test")
        self.backend.client = SimpleNamespace(messages=_FakeAnthropicMessages())
        self.calls = self.backend.client.messages.calls

    def test_every_turn_is_sent_with_the_system_prompt(self):
        @sgl.function
        def program(s):
            s += sgl.system(self.SYSTEM["content"])
            s += sgl.user("q1")
            s += sgl.assistant(sgl.gen("a1"))
            s += sgl.user("q2")
            s += sgl.assistant(sgl.gen("a2"))

        state = program.run(backend=self.backend, use_thread=False)

        self.assertEqual([call["system"] for call in self.calls], ["be brief"] * 2)
        self.assertEqual(
            [m["role"] for m in self.calls[1]["messages"]],
            ["user", "assistant", "user"],
        )
        self.assertEqual(state.messages()[0], self.SYSTEM)

    def test_streaming_keeps_the_system_prompt(self):
        executor = StreamExecutor(
            self.backend, {}, SglSamplingParams(), None, stream=False, use_thread=False
        )
        executor.submit(sgl.system(self.SYSTEM["content"]))
        executor.submit(sgl.user("q"))

        pieces = self.backend.generate_stream(
            executor, sampling_params=SglSamplingParams(max_new_tokens=4)
        )

        self.assertEqual("".join(text for text, _ in pieces), "hi")
        self.assertEqual(self.calls[0]["system"], "be brief")
        self.assertEqual(executor.messages()[0], self.SYSTEM)


if __name__ == "__main__":
    unittest.main()
