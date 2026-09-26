"""Audio benchmark payload and usage accounting without models or accelerators."""

import asyncio
import base64
import io
import json
import os
import random
import resource
import sys
import tempfile
import threading
import unittest
import wave
from argparse import Namespace
from contextlib import contextmanager, redirect_stdout
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import patch

import numpy as np

from sglang.benchmark import serving
from sglang.benchmark.datasets.audio import AudioDataset
from sglang.benchmark.datasets.common import DatasetRow
from sglang.benchmark.serving import (
    RequestFuncInput,
    async_request_openai_chat_completions,
    calculate_metrics,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _dataset(**overrides):
    settings = dict(
        num_requests=2,
        audio_duration=0.05,
        audio_count=1,
        random_audio_count=False,
        output_len=16,
        seed=17,
        backend="sglang-oai-chat",
    )
    settings.update(overrides)
    return AudioDataset(**settings)


def _audio_parts(row):
    return [
        part
        for message in row.prompt
        for part in message["content"]
        if part["type"] == "input_audio"
    ]


@contextmanager
def _chat_endpoint(stream, usage):
    """Exercise the real HTTP client against an OpenAI-shaped response."""
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b"{}")

        def do_POST(self):
            size = int(self.headers["Content-Length"])
            requests.append(json.loads(self.rfile.read(size)))
            response_usage = usage(len(requests) - 1) if callable(usage) else usage
            self.send_response(200)
            self.send_header(
                "Content-Type", "text/event-stream" if stream else "application/json"
            )
            self.end_headers()
            if stream:
                chunks = [
                    {"choices": [{"index": 0, "delta": {"content": "heard "}}]},
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": "audio"},
                                "finish_reason": "stop",
                            }
                        ]
                    },
                ]
                if response_usage is not None:
                    chunks.append({"choices": [], "usage": response_usage})
                for chunk in chunks:
                    self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
                self.wfile.write(b"data: [DONE]\n\n")
            else:
                response = {
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "heard audio"},
                            "finish_reason": "stop",
                        }
                    ]
                }
                if response_usage is not None:
                    response["usage"] = response_usage
                self.wfile.write(json.dumps(response).encode())
            self.wfile.flush()

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1/chat/completions", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


class _TextTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()


class TestAudioDataset(CustomTestCase):
    def test_wav_round_trip_and_duration(self):
        """Wire audio must decode to the requested duration, not a data URI string."""
        rows = _dataset(audio_count=2).load()
        self.assertEqual(len(rows), 2)
        payloads = []
        for row in rows:
            self.assertTrue(row.prompt_len_from_usage)
            self.assertEqual(row.prompt_len, 0)
            self.assertAlmostEqual(row.audio_duration, 0.1)
            parts = _audio_parts(row)
            self.assertEqual(len(parts), 2)
            for part in parts:
                self.assertEqual(part["input_audio"]["format"], "wav")
                payload = base64.b64decode(part["input_audio"]["data"], validate=True)
                with wave.open(io.BytesIO(payload), "rb") as audio:
                    self.assertEqual(audio.getnchannels(), 1)
                    self.assertEqual(audio.getsampwidth(), 2)
                    self.assertEqual(audio.getframerate(), 16000)
                    self.assertEqual(audio.getnframes(), 800)
                    self.assertEqual(len(audio.readframes(801)), 1600)
                payloads.append(payload)
        # Reusing one identical waveform silently benchmarks multimodal cache hits.
        self.assertEqual(len(set(payloads)), 4)

    def test_seed_is_independent_of_process_rng_state(self):
        """Dataset replay must survive unrelated Python/NumPy RNG consumers."""
        dataset = _dataset(random_audio_count=True, audio_count=3, num_requests=8)
        first = dataset.load()
        python_state, numpy_state = random.getstate(), np.random.get_state()
        try:
            random.seed(999)
            np.random.seed(999)
            random.random()
            np.random.random(20)
            second = dataset.load()
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
        self.assertEqual(first, second)
        self.assertNotEqual(
            first,
            _dataset(
                seed=18, random_audio_count=True, audio_count=3, num_requests=8
            ).load(),
        )
        for row in first:
            count = len(_audio_parts(row))
            self.assertLessEqual(count, 3)
            self.assertAlmostEqual(row.audio_duration, count * 0.05)

    def test_zero_clip_boundaries_produce_text_only_messages(self):
        for random_count in (False, True):
            with self.subTest(random_audio_count=random_count):
                rows = _dataset(audio_count=0, random_audio_count=random_count).load()
                for row in rows:
                    self.assertEqual(_audio_parts(row), [])
                    self.assertEqual(row.audio_duration, 0)
                    self.assertTrue(
                        any(
                            part["type"] == "text" and part["text"]
                            for message in row.prompt
                            for part in message["content"]
                        )
                    )

    def test_invalid_settings_reject_before_audio_generation(self):
        for settings in (
            {"audio_count": -1},
            {"audio_duration": 0},
            {"audio_duration": -1},
            {"audio_duration": float("nan")},
            {"audio_duration": float("inf")},
            {"num_requests": -1},
            {"output_len": 0},
            {"backend": "sglang"},
            {"backend": "sglang-oai"},
        ):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                _dataset(**settings).load()

    def test_cli_settings_reach_serialized_requests(self):
        dataset = AudioDataset.from_args(
            Namespace(
                num_prompts=3,
                audio_duration=0.025,
                audio_count=2,
                random_audio_count=False,
                random_output_len=19,
                seed=17,
                backend="sglang-oai-chat",
            )
        )
        rows = dataset.load()
        self.assertEqual(len(rows), 3)
        for row in rows:
            self.assertEqual(row.output_len, 19)
            self.assertAlmostEqual(row.audio_duration, 0.05)
            self.assertEqual(len(_audio_parts(row)), 2)


class TestAudioBenchmarkRequests(CustomTestCase):
    def _request(self, row, *, stream, usage, extra_body=None):
        settings = Namespace(
            disable_stream=not stream,
            disable_ignore_eos=True,
            print_requests=False,
            tokenizer="",
            header=None,
        )
        with (
            _chat_endpoint(stream, usage) as (url, requests),
            patch.object(serving, "args", settings, create=True),
        ):
            request = RequestFuncInput(
                prompt=row.prompt,
                api_url=url,
                prompt_len=row.prompt_len,
                output_len=row.output_len,
                model="test-audio-model",
                lora_name="",
                image_data=None,
                extra_request_body=extra_body or {},
                prompt_len_from_usage=row.prompt_len_from_usage,
            )
            result = asyncio.run(async_request_openai_chat_completions(request))
        return result, requests

    def test_audio_payload_and_server_usage_in_both_response_modes(self):
        row = _dataset(num_requests=1).load()[0]
        for stream in (False, True):
            with self.subTest(stream=stream):
                extra = (
                    {"stream_options": {"continuous_usage_stats": True}}
                    if stream
                    else {}
                )
                result, requests = self._request(
                    row,
                    stream=stream,
                    usage={"prompt_tokens": 145, "completion_tokens": 2},
                    extra_body=extra,
                )
                self.assertTrue(result.success, result.error)
                self.assertEqual(result.prompt_len, 145)
                self.assertEqual(result.generated_text, "heard audio")
                self.assertEqual(result.output_len, 2)
                self.assertEqual(len(requests), 1)
                self.assertEqual(requests[0]["messages"], row.prompt)
                self.assertEqual(requests[0]["stream"], stream)
                if stream:
                    self.assertEqual(
                        requests[0]["stream_options"],
                        {"continuous_usage_stats": True, "include_usage": True},
                    )

    def test_audio_request_rejects_missing_or_invalid_prompt_usage(self):
        row = _dataset(num_requests=1).load()[0]
        usages = [None, {"completion_tokens": 2}]
        usages.extend(
            {"prompt_tokens": value, "completion_tokens": 2}
            for value in (None, True, -1, 1.5, "145")
        )
        for stream in (False, True):
            for usage in usages:
                with self.subTest(stream=stream, usage=usage):
                    result, _ = self._request(row, stream=stream, usage=usage)
                    self.assertFalse(result.success)
                    self.assertTrue(result.error)

    def test_zero_prompt_tokens_is_valid_usage(self):
        row = _dataset(num_requests=1).load()[0]
        for stream in (False, True):
            with self.subTest(stream=stream):
                result, _ = self._request(
                    row,
                    stream=stream,
                    usage={"prompt_tokens": 0, "completion_tokens": 2},
                )
                self.assertTrue(result.success, result.error)
                self.assertEqual(result.prompt_len, 0)

    def test_metrics_use_audio_usage_without_changing_text_estimates(self):
        """Audio totals must reflect expansion while text keeps its existing contract."""
        audio_row = _dataset(num_requests=1).load()[0]
        text_row = DatasetRow(prompt="plain text", prompt_len=7, output_len=2)
        outputs = []
        for row, reported_tokens in ((audio_row, 145), (text_row, 999)):
            output, _ = self._request(
                row,
                stream=True,
                usage={"prompt_tokens": reported_tokens, "completion_tokens": 2},
            )
            self.assertTrue(output.success, output.error)
            outputs.append(output)
        metrics, lengths = calculate_metrics(
            [audio_row, text_row], outputs, 2.0, _TextTokenizer(), "sglang-oai-chat"
        )
        self.assertEqual([output.prompt_len for output in outputs], [145, 7])
        self.assertEqual(metrics.total_input, 152)
        self.assertEqual(metrics.input_throughput, 76)
        self.assertEqual(metrics.total_input_text, 7)
        self.assertEqual(metrics.total_input_vision, 0)
        self.assertEqual(metrics.completed, 2)
        self.assertEqual(lengths, [2, 2])

    def test_text_request_still_allows_missing_usage(self):
        row = DatasetRow(prompt="plain text", prompt_len=7, output_len=2)
        for stream in (False, True):
            with self.subTest(stream=stream):
                output, requests = self._request(row, stream=stream, usage=None)
                self.assertTrue(output.success, output.error)
                self.assertEqual(output.prompt_len, 7)
                self.assertNotIn("stream_options", requests[0])

    def test_explicitly_disabled_usage_fails_before_sending_audio(self):
        row = _dataset(num_requests=1).load()[0]
        output, requests = self._request(
            row,
            stream=True,
            usage={"prompt_tokens": 145, "completion_tokens": 2},
            extra_body={"stream_options": {"include_usage": False}},
        )
        self.assertFalse(output.success)
        self.assertIn("include_usage", output.error)
        self.assertEqual(requests, [])


class TestAudioBenchmarkCLI(CustomTestCase):
    def _run_cli(self, url, output_path):
        argv = [
            "bench_serving",
            "--backend",
            "sglang-oai-chat",
            "--dataset-name",
            "audio",
            "--base-url",
            url.removesuffix("/v1/chat/completions"),
            "--model",
            "fake",
            "--tokenizer",
            "fake",
            "--ready-check-timeout-sec",
            "0",
            "--num-prompts",
            "2",
            "--warmup-requests",
            "1",
            "--max-concurrency",
            "1",
            "--audio-duration",
            "0.05",
            "--audio-count",
            "1",
            "--random-output-len",
            "2",
            "--disable-tqdm",
            "--output-details",
            "--output-file",
            str(output_path),
            "--extra-request-body",
            json.dumps({"stream_options": {"continuous_usage_stats": True}}),
        ]
        python_state, numpy_state = random.getstate(), np.random.get_state()
        original_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
        try:
            with (
                patch.object(sys, "argv", argv),
                patch.object(serving, "args", None, create=True),
                patch.object(serving, "get_tokenizer", return_value=_TextTokenizer()),
                patch.object(serving, "check_chat_template", return_value=True),
                patch.dict(os.environ, {"SGLANG_IS_IN_CI": "0"}),
                redirect_stdout(io.StringIO()),
            ):
                serving.cli_main()
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            resource.setrlimit(resource.RLIMIT_NOFILE, original_limit)

    def test_cli_carries_audio_usage_through_warmup_and_results(self):
        """Public CLI accounting excludes warmup and failed measured requests."""
        for fail_last in (False, True):
            with self.subTest(fail_last=fail_last):

                def usage(index):
                    if fail_last and index == 2:
                        return {"completion_tokens": 2}
                    return {
                        "prompt_tokens": (9000, 145, 146)[index],
                        "completion_tokens": 2,
                    }

                with (
                    tempfile.TemporaryDirectory() as directory,
                    _chat_endpoint(True, usage) as (url, requests),
                ):
                    output_path = Path(directory) / "audio.jsonl"
                    self._run_cli(url, output_path)
                    results = [
                        json.loads(line)
                        for line in output_path.read_text().splitlines()
                    ]
                self.assertEqual(len(results), 1)
                result = results[0]
                self.assertEqual(len(requests), 3)
                for request in requests:
                    self.assertEqual(
                        request["stream_options"],
                        {"continuous_usage_stats": True, "include_usage": True},
                    )
                    self.assertTrue(
                        any(
                            part["type"] == "input_audio"
                            for message in request["messages"]
                            for part in message["content"]
                        )
                    )
                self.assertEqual(result["completed"], 1 if fail_last else 2)
                self.assertEqual(
                    result["total_input_tokens"], 145 if fail_last else 291
                )
                self.assertEqual(result["input_lens"], [145, 0 if fail_last else 146])
                self.assertEqual(result["input_token_source"], "server_usage")
                self.assertIsNone(result["total_input_text_tokens"])
                self.assertIsNone(result["total_input_vision_tokens"])
                self.assertAlmostEqual(
                    result["total_submitted_audio_seconds"], 0.05 if fail_last else 0.1
                )
                self.assertEqual(bool(result["errors"][1]), fail_last)

    def test_cli_stops_after_warmup_without_required_usage(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            _chat_endpoint(True, {"completion_tokens": 2}) as (url, requests),
        ):
            output_path = Path(directory) / "audio.jsonl"
            with self.assertRaisesRegex(ValueError, "Warmup failed"):
                self._run_cli(url, output_path)
            self.assertEqual(len(requests), 1)
            self.assertTrue(requests[0]["stream_options"]["include_usage"])
            self.assertFalse(output_path.exists())


if __name__ == "__main__":
    unittest.main()
