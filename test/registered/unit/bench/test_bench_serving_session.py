"""Unit tests for bench_serving's --session-per-conversation multi-turn sessions."""

import asyncio
import json
import threading
import unittest
from argparse import Namespace
from http.server import BaseHTTPRequestHandler, HTTPServer

from sglang.benchmark.serving import (
    RequestFuncInput,
    async_request_openai_chat_completions,
    set_global_args,
    wrap_multi_turn_request_func,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

CHAT_PATH = "/v1/chat/completions"
CLOSE_PATH = "/close_session"


class _RecordingHandler(BaseHTTPRequestHandler):
    """Records every POST as (path, body) and answers chat requests non-streamed."""

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.posts.append((self.path, body))
        payload = b"{}"
        if self.path == CHAT_PATH:
            message = {"role": "assistant", "content": "hi"}
            choice = {"index": 0, "message": message, "finish_reason": "length"}
            usage = {"completion_tokens": 1, "prompt_tokens": 1}
            payload = json.dumps({"choices": [choice], "usage": usage}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt, *args):
        pass


class TestBenchServingSessionPerConversation(CustomTestCase):
    def setUp(self):
        self.server = HTTPServer(("127.0.0.1", 0), _RecordingHandler)
        self.server.posts = []
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"
        set_global_args(
            Namespace(
                disable_stream=True,
                disable_ignore_eos=True,
                print_requests=False,
                tokenizer="",
                header=None,
                cache_report=False,
            )
        )

    def _run_conversation(self, multi_turn):
        request = RequestFuncInput(
            prompt=["one", "two", "three"],
            api_url=self.base_url + CHAT_PATH,
            prompt_len=1,
            output_len=1,
            model="dummy-model",
            lora_name="",
            image_data=None,
            extra_request_body={},
        )
        for output in asyncio.run(multi_turn(request)):
            self.assertTrue(output.success, output.error)

    def test_each_conversation_gets_its_own_session_closed_after_its_last_round(self):
        """All rounds of a conversation must share one session, conversations must
        not share one, and the close must follow the last round."""
        multi_turn = wrap_multi_turn_request_func(
            async_request_openai_chat_completions,
            backend="sglang-oai-chat",
            close_session_url=self.base_url + CLOSE_PATH,
        )
        self._run_conversation(multi_turn)
        self._run_conversation(multi_turn)

        self.assertEqual(
            [path for path, _ in self.server.posts],
            ([CHAT_PATH] * 3 + [CLOSE_PATH]) * 2,
        )
        first, second = self.server.posts[:4], self.server.posts[4:]
        for conversation in (first, second):
            self.assertEqual(len({body["session_id"] for _, body in conversation}), 1)
        self.assertNotEqual(first[0][1]["session_id"], second[0][1]["session_id"])

    def test_without_close_url_no_session_is_opened_or_closed(self):
        multi_turn = wrap_multi_turn_request_func(
            async_request_openai_chat_completions, backend="sglang-oai-chat"
        )
        self._run_conversation(multi_turn)

        self.assertEqual([path for path, _ in self.server.posts], [CHAT_PATH] * 3)
        self.assertTrue(all("session_id" not in body for _, body in self.server.posts))


if __name__ == "__main__":
    unittest.main()
