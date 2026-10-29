"""Request logging supports non-chat backends and groups all conversation turns."""

import asyncio
import copy
import io
import json
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch
from uuid import UUID

from sglang.benchmark import serving
from sglang.benchmark.datasets import DatasetRow
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"{}")

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.bodies.append(body)
        usage = {"prompt_tokens": 3, "completion_tokens": 2}
        if self.path == "/generate":
            reply = {
                "text": "answer sentinel",
                "meta_info": {"prompt_tokens": 3, "completion_tokens": 2},
            }
        elif self.path == "/v1/chat/completions":
            reply = {
                "choices": [{"message": {"content": "answer sentinel"}}],
                "usage": usage,
            }
        else:
            reply = {"choices": [{"text": "answer sentinel"}], "usage": usage}
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(reply).encode())

    def log_message(self, fmt, *args):
        pass


class TestBenchServingRequestLogging(CustomTestCase):
    def setUp(self):
        self.enterContext(patch.object(serving, "args", None, create=True))
        self.tmp = self.enterContext(tempfile.TemporaryDirectory())
        self.server = HTTPServer(("127.0.0.1", 0), _Handler)
        self.server.bodies = []
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 5)
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.capture = io.StringIO()

    def _log_lines(self):
        return [
            line
            for line in self.capture.getvalue().splitlines()
            if line.startswith("rid=")
        ]

    def _assert_pairs(self, count):
        lines = self._log_lines()
        self.assertEqual(len(lines), 2 * count)
        ids = set()
        for start, end in zip(lines[::2], lines[1::2]):
            rid = start.split()[0]
            UUID(rid.removeprefix("rid="))
            self.assertEqual(rid, end.split()[0])
            self.assertNotIn(rid, ids)
            ids.add(rid)
            self.assertIn('message="request start"', start)
            self.assertIn('message="request end"', end)
            self.assertIn("time_delta=", end)
        self.assertNotIn("prompt sentinel", "\n".join(lines))
        self.assertNotIn("answer sentinel", "\n".join(lines))
        return lines

    def _args(self, backend, enabled=True, warmup=0):
        argv = [
            "bench_serving",
            "--backend",
            backend,
            "--model",
            "dummy",
            "--tokenizer",
            "dummy",
            "--dataset-name",
            "random",
            "--base-url",
            f"http://127.0.0.1:{self.server.server_port}",
            "--ready-check-timeout-sec",
            "0",
            "--disable-stream",
            "--warmup-requests",
            str(warmup),
            "--max-concurrency",
            "1",
            "--output-file",
            str(Path(self.tmp) / "result.jsonl"),
        ]
        if enabled:
            argv.append("--print-requests")
        with patch("sys.argv", argv), patch.object(serving, "run_benchmark") as run:
            serving.cli_main()
        return run.call_args.args[0]

    def _run(self, args, rows):
        tokenizer = Mock()
        tokenizer.encode.return_value = [1, 2]
        with (
            patch.object(serving, "get_dataset", return_value=rows),
            patch.object(serving, "get_tokenizer", return_value=tokenizer),
            patch.object(serving, "check_chat_template", return_value=True),
            patch.object(serving, "set_ulimit"),
            patch.object(serving, "_get_bool_env_var", return_value=False),
            redirect_stdout(self.capture),
        ):
            return serving.run_benchmark(args)

    def test_result_and_input_are_not_mutated(self):
        request = serving.RequestFuncInput(
            prompt="prompt sentinel",
            api_url="http://unused",
            prompt_len=3,
            output_len=2,
            model="dummy",
            lora_name=None,
            image_data=None,
            extra_request_body={"temperature": 0},
        )
        original = copy.deepcopy(request)
        for success in (True, False):
            with self.subTest(success=success):
                self.capture = io.StringIO()
                output = serving.RequestFuncOutput(
                    generated_text="answer sentinel",
                    text_chunks=["answer sentinel"],
                    success=success,
                    error="" if success else "HTTP 500",
                )
                saved_output = copy.deepcopy(output)
                backend = AsyncMock(return_value=output)
                pbar = Mock()
                with redirect_stdout(self.capture):
                    actual = asyncio.run(
                        serving.wrap_request_func_with_logging(backend)(request, pbar)
                    )
                self.assertIs(actual, output)
                self.assertEqual(output, saved_output)
                self.assertEqual(request, original)
                backend.assert_awaited_once_with(request, pbar=pbar)
                self._assert_pairs(1)
                self.assertIn(f"success={success}", self._log_lines()[-1])

    def test_exceptions_and_cancellation_propagate(self):
        request = serving.RequestFuncInput(
            "prompt sentinel", "http://unused", 3, 2, "dummy", None, None, {}
        )
        for error in (RuntimeError("failed"), asyncio.CancelledError()):
            with self.subTest(error=type(error).__name__):
                self.capture = io.StringIO()
                backend = AsyncMock(side_effect=error)
                with (
                    redirect_stdout(self.capture),
                    self.assertRaises(type(error)) as caught,
                ):
                    asyncio.run(
                        serving.wrap_request_func_with_logging(backend)(request)
                    )
                self.assertIs(caught.exception, error)
                self._assert_pairs(1)
                self.assertIn('output="None"', self._log_lines()[-1])

    def test_non_chat_backends_log_requests(self):
        for backend in ("sglang", "sglang-oai"):
            with self.subTest(backend=backend):
                self.capture = io.StringIO()
                result = self._run(
                    self._args(backend),
                    [DatasetRow(prompt="prompt sentinel", prompt_len=3, output_len=2)],
                )
                self.assertEqual(result["completed"], 1)
                self.assertEqual(result["generated_texts"], ["answer sentinel"])
                self._assert_pairs(1)

    def test_multiturn_including_warmup_logs_once_per_conversation(self):
        row = DatasetRow(
            prompt=["prompt sentinel one", "prompt sentinel two"],
            prompt_len=3,
            output_len=2,
        )
        original = copy.deepcopy(row)
        with patch.object(serving, "tqdm") as progress:
            result = self._run(self._args("sglang-oai-chat", warmup=1), [row])
        self.assertEqual(row, original)
        self.assertEqual(result["generated_texts"], ["answer sentinel"] * 2)
        self.assertEqual(len(self.server.bodies), 4)
        self.assertEqual(
            self.server.bodies[-1]["messages"][1]["content"], "answer sentinel"
        )
        progress.return_value.update.assert_called_once_with(1)
        self._assert_pairs(2)

    def test_logging_starts_after_concurrency_admission_and_can_be_disabled(self):
        row = DatasetRow(prompt="prompt sentinel", prompt_len=3, output_len=2)
        for enabled in (True, False):
            with self.subTest(enabled=enabled):
                self.capture = io.StringIO()
                calls = 0

                async def backend(request_func_input, pbar=None):
                    nonlocal calls
                    calls += 1
                    # Give the other scheduled request a chance to wait for its slot.
                    await asyncio.sleep(0)
                    self.assertEqual(
                        len(self._log_lines()), 2 * calls - 1 if enabled else 0
                    )
                    return serving.RequestFuncOutput(
                        generated_text="answer sentinel",
                        success=True,
                        output_len=2,
                        prompt_len=3,
                        latency=0.01,
                    )

                with (
                    patch.dict(serving.ASYNC_REQUEST_FUNCS, {"sglang-oai": backend}),
                    patch.object(
                        serving.uuid, "uuid4", wraps=serving.uuid.uuid4
                    ) as uuid4,
                ):
                    result = self._run(
                        self._args("sglang-oai", enabled=enabled, warmup=2), [row, row]
                    )
                self.assertEqual(result["completed"], 2)
                self.assertEqual(calls, 4)
                self._assert_pairs(4 if enabled else 0)
                self.assertEqual(uuid4.call_count, 4 if enabled else 0)


if __name__ == "__main__":
    unittest.main()
