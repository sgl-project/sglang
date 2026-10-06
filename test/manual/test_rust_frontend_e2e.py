"""Manual E2E tests for the Rust frontend's HTTP and runtime.v1 gRPC listeners.

Run from a SGLang source checkout with one GPU, the Rust server extension,
and grpcio-tools installed in the same environment as the sglang command:

    python3 test/manual/test_rust_frontend_e2e.py -v

Uses the standard small test model. No CI registration or baseline checkout
is required; generated bindings and server logs use a temporary directory.
"""

import importlib
import json
import re
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import grpc
import httpx
import openai
import psutil
import requests

from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    find_available_port,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

PROMPT = "Write a short sentence about the ocean."
APIS = ("text", "tokens", "completion", "chat")


def wait_for(predicate, description, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.1)
    raise AssertionError(f"Timed out: {description}")


def _payload(raw):
    value = json.loads(raw)
    assert isinstance(value, dict), f"expected JSON object: {value!r}"
    assert "error" not in value, f"server error: {value!r}"
    return value


def grpc_frames(call, api, stream):
    """Read synchronous runtime.v1 response iterators, including stream=false."""
    assert api in ("text", "tokens", "completion", "chat"), api
    native = api in ("text", "tokens")
    finished = False
    count = 0
    for chunk in call:
        assert not finished, "gRPC response after terminal marker"
        count += 1
        finished = chunk.finished
        if native:
            if not stream:
                assert finished and count == 1, (
                    "nonstream native must have one response"
                )
            value = {
                "meta_info": {
                    key: json.loads(raw) for key, raw in chunk.meta_info.items()
                }
            }
            value["text" if api == "text" else "output_ids"] = (
                chunk.text if api == "text" else list(chunk.output_ids)
            )
            assert bool(value["meta_info"].get("finish_reason")) == finished
            yield value
        elif stream:
            if finished:
                assert not chunk.json_chunk, "stream terminal must be empty"
            else:
                assert chunk.json_chunk, "empty nonterminal OpenAI chunk"
                yield _payload(chunk.json_chunk)
        else:
            assert finished and count == 1, (
                "nonstream OpenAI must have one terminal JSON chunk"
            )
            yield _payload(chunk.json_chunk)
    assert finished, "gRPC response truncated before terminal marker"


def _usage(value):
    assert isinstance(value, dict), f"missing usage: {value!r}"
    result = {
        key: value[key]
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    assert all(type(count) is int and count > 0 for count in result.values()), result
    assert (
        result["total_tokens"] == result["prompt_tokens"] + result["completion_tokens"]
    ), result
    return result


def summarize(api, frames, stream):
    """Return transport-stable output, finish_reason and usage for one request.

    Native output is text or token IDs; OpenAI output and finish_reason are
    lists ordered by choice index. Native streams must use cumulative output.
    """
    assert api in ("text", "tokens", "completion", "chat"), api
    frames = list(frames)
    assert frames, "no JSON response frames"
    if api in ("text", "tokens"):
        final = frames[-1]
        meta = final["meta_info"]
        finish = meta.get("finish_reason")
        assert isinstance(finish, dict) and finish.get("type") in ("stop", "length"), (
            finish
        )
        assert all(not frame["meta_info"].get("finish_reason") for frame in frames[:-1])
        counts = [frame["meta_info"]["completion_tokens"] for frame in frames]
        assert counts == sorted(counts), counts
        output = final["text" if api == "text" else "output_ids"]
        assert output, "empty native output"
        usage = _usage(
            {
                "prompt_tokens": meta["prompt_tokens"],
                "completion_tokens": meta["completion_tokens"],
                "total_tokens": meta["prompt_tokens"] + meta["completion_tokens"],
            }
        )
        return {"output": output, "finish_reason": finish, "usage": usage}

    outputs, finishes = {}, {}
    usage = None
    for frame in frames:
        assert "error" not in frame, frame
        choices = frame["choices"]
        if frame.get("usage") is not None:
            assert usage is None, "multiple final usage records"
            usage = _usage(frame["usage"])
            if stream:
                assert choices == [], "expected final aggregate usage chunk"
        else:
            assert usage is None, "data after final usage"
        for choice in choices:
            index = choice["index"]
            assert type(index) is int and index >= 0, index
            assert index not in finishes, "choice data after finish"
            if api == "chat":
                text = choice["delta" if stream else "message"].get("content") or ""
            else:
                text = choice["text"]
            assert isinstance(text, str), text
            outputs[index] = outputs.get(index, "") + text
            reason = choice.get("finish_reason")
            if reason is not None:
                assert reason in ("stop", "length"), reason
                finishes[index] = reason
    indexes = sorted(outputs)
    assert indexes == list(range(len(indexes))) and indexes, indexes
    assert all(outputs[index] and index in finishes for index in indexes), (
        outputs,
        finishes,
    )
    assert usage is not None, "missing aggregate usage"
    if not stream:
        assert len(frames) == 1, "nonstream OpenAI returned multiple payloads"
    return {
        "output": [outputs[index] for index in indexes],
        "finish_reason": [finishes[index] for index in indexes],
        "usage": usage,
    }


class TestRustFrontendE2E(CustomTestCase):
    process = None
    owned_children = []
    channel = None
    server_log = None
    model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
    base_url = DEFAULT_URL_FOR_TEST

    @classmethod
    def setUpClass(cls):
        if not __debug__:
            raise RuntimeError("Do not run this suite with Python -O")
        cls.source = Path(__file__).resolve().parents[2]
        temporary = tempfile.TemporaryDirectory(prefix="rust-frontend-e2e-")
        cls.addClassCleanup(temporary.cleanup)
        cls.output = Path(temporary.name)
        cls.grpc_port = find_available_port(int(cls.base_url.rsplit(":", 1)[1]) + 1)
        cls.tokenizer = get_tokenizer(cls.model)
        cls.tokens = cls.tokenizer.encode(PROMPT)

        # Generate the client from this checkout's canonical service definition.
        generated = cls.output / "client"
        generated.mkdir()
        proto = cls.source / "proto/sglang/runtime/v1"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "grpc_tools.protoc",
                f"-I{proto}",
                f"--python_out={generated}",
                f"--grpc_python_out={generated}",
                str(proto / "sglang.proto"),
            ],
            check=True,
        )
        sys.path.insert(0, str(generated))
        cls.addClassCleanup(sys.path.remove, str(generated))
        cls.pb = importlib.import_module("sglang_pb2")
        cls.stub_type = importlib.import_module("sglang_pb2_grpc").SglangServiceStub
        cls.client = openai.OpenAI(
            api_key="EMPTY",
            base_url=f"{cls.base_url}/v1",
            timeout=httpx.Timeout(30, connect=5),
            max_retries=0,
        )
        cls.addClassCleanup(cls.client.close)
        cls.launch()

    @classmethod
    def launch(cls):
        offset = len(cls.logs()) if (cls.output / "server.log").exists() else 0
        cls.server_log = (cls.output / "server.log").open("a")
        args = [
            "--served-model-name",
            "e2e-model",
            "--random-seed",
            "42",
            "--context-length",
            "4096",
            "--mem-fraction-static",
            "0.5",
            "--max-running-requests",
            "2",
            "--disable-cuda-graph",
            "--disable-prefill-cuda-graph",
            "--disable-radix-cache",
            "--log-level",
            "debug",
            "--grpc-port",
            str(cls.grpc_port),
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=args,
            return_stdout_stderr=(cls.server_log, cls.server_log),
            env={
                "SGLANG_RUST_SERVER": "1",
                "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION": "0",
                "PYTHONPATH": str(cls.source / "python"),
            },
        )
        cls.owned_children = psutil.Process(cls.process.pid).children(recursive=True)
        wait_for(
            lambda: "The server is fired up and ready to roll!" in cls.logs()[offset:],
            "model warmup",
            timeout=120,
        )
        assert "SGLANG_RUST_SERVER enabled" in cls.logs()[offset:], (
            "Not the Rust frontend"
        )
        assert "gRPC server listening" in cls.logs()[offset:], (
            "Not the Rust Tonic listener"
        )
        cls.channel = grpc.insecure_channel(f"127.0.0.1:{cls.grpc_port}")
        grpc.channel_ready_future(cls.channel).result(timeout=15)
        cls.stub = cls.stub_type(cls.channel)

    @classmethod
    def logs(cls):
        return (cls.output / "server.log").read_text(errors="replace")

    @classmethod
    def tearDownClass(cls):
        try:
            if cls.channel is not None:
                cls.channel.close()
            if cls.process is not None and cls.process.poll() is None:
                terminate_and_kill_process_tree(cls.process)
            # The launcher may have exited without cleaning up its children.
            for child in cls.owned_children:
                if child.is_running() and child.status() != psutil.STATUS_ZOMBIE:
                    child.kill()
        finally:
            if cls.server_log is not None:
                cls.server_log.close()
                print(cls.logs())

    def body(self, api, stream):
        if api in ("text", "tokens"):
            return {
                "text" if api == "text" else "input_ids": (
                    PROMPT if api == "text" else self.tokens
                ),
                "stream": stream,
                "sampling_params": {"temperature": 0, "max_new_tokens": 16},
            }
        body = {
            "model": "e2e-model",
            "stream": stream,
            "temperature": 0,
            "max_tokens": 16,
        }
        if api == "chat":
            body["messages"] = [{"role": "user", "content": PROMPT}]
        else:
            body["prompt"] = PROMPT
        if stream:
            body["stream_options"] = {"include_usage": True}
        return body

    def open_call(self, api, body, transport):
        if transport == "http":
            if api == "chat":
                return self.client.chat.completions.create(**body)
            if api == "completion":
                return self.client.completions.create(**body)
            return requests.post(
                self.base_url + "/generate",
                json=body,
                stream=body["stream"],
                timeout=(5, 30),
            )
        if api in ("text", "tokens"):
            body = dict(
                body, sampling_params=self.pb.SamplingParams(**body["sampling_params"])
            )
            request_type, method = (
                (self.pb.TextGenerateRequest, self.stub.TextGenerate)
                if api == "text"
                else (self.pb.GenerateRequest, self.stub.Generate)
            )
            return method(request_type(**body), timeout=30)
        method = self.stub.ChatComplete if api == "chat" else self.stub.Complete
        return method(
            self.pb.OpenAIRequest(json_body=json.dumps(body).encode()), timeout=30
        )

    def frames(self, call, api, stream, transport):
        if transport == "grpc":
            yield from grpc_frames(call, api, stream)
        elif api in ("completion", "chat"):
            # The OpenAI SDK handles HTTP/SSE; use JSON dictionaries for parity.
            for response in call if stream else [call]:
                yield response.model_dump(exclude_none=True)
        else:
            call.raise_for_status()
            content_type = call.headers.get("content-type", "").split(";", 1)[0].strip()
            if not stream:
                assert content_type == "application/json", content_type
                yield _payload(call.content)
                return
            assert content_type == "text/event-stream", content_type
            # Match the native /generate tests' line-based SSE reader.
            finished = False
            for line in call.iter_lines():
                if line.startswith(b"data:"):
                    assert not finished, "SSE data after [DONE]"
                    data = line[5:].strip()
                    if data == b"[DONE]":
                        finished = True
                    else:
                        yield _payload(data)
            assert finished, "SSE response truncated before [DONE]"

    def generate(self, api, stream, transport="http", body=None):
        call = self.open_call(api, body or self.body(api, stream), transport)
        try:
            frames = list(self.frames(call, api, stream, transport))
            return summarize(api, frames, stream)
        finally:
            if transport == "grpc":
                call.cancel()
            elif api in ("text", "tokens") or stream:
                call.close()

    def test_generation(self):
        for api in APIS:
            expected = None
            for stream in (False, True):
                with self.subTest(api=api, stream=stream):
                    summary = self.generate(api, stream)
                    if expected is None:
                        expected = summary
                    self.assertEqual(
                        summary, expected, "HTTP stream/nonstream mismatch"
                    )
                    self.assertEqual(self.generate(api, stream, "grpc"), summary)

    def test_metadata_and_health(self):
        for path in ("/health", "/health_generate"):
            self.assertEqual(
                requests.get(self.base_url + path, timeout=30).status_code, 200
            )
        info = requests.get(self.base_url + "/get_model_info", timeout=30).json()
        self.assertEqual(info["served_model_name"], "e2e-model")
        with self.subTest(operation="http/server_info"):
            with requests.get(self.base_url + "/server_info", timeout=30) as response:
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()["max_context_length"], 4096)
        self.assertTrue(
            self.stub.HealthCheck(self.pb.HealthCheckRequest(), timeout=30).healthy
        )
        self.assertEqual(
            json.loads(
                self.stub.GetModelInfo(
                    self.pb.GetModelInfoRequest(), timeout=30
                ).json_info
            ),
            info,
        )
        models = self.stub.ListModels(self.pb.ListModelsRequest(), timeout=30).models
        self.assertEqual(
            [(m.id, m.max_model_len) for m in models], [("e2e-model", 4096)]
        )
        with self.subTest(operation="grpc/GetServerInfo"):
            server = json.loads(
                self.stub.GetServerInfo(
                    self.pb.GetServerInfoRequest(), timeout=30
                ).json_info
            )
            self.assertEqual(server["max_context_length"], 4096)
        decoded = self.stub.Detokenize(
            self.pb.DetokenizeRequest(tokens=self.tokens), timeout=30
        ).text
        self.assertEqual(
            decoded, self.tokenizer.decode(self.tokens, skip_special_tokens=True)
        )

    def test_rejections(self):
        with requests.post(
            self.base_url + "/v1/completions",
            json=self.body("completion", False) | {"n": 0},
            timeout=30,
        ) as response:
            self.assertEqual(response.status_code, 400)
        with self.assertRaises(grpc.RpcError) as caught:
            self.stub.Tokenize(self.pb.TokenizeRequest(text=PROMPT), timeout=30)
        self.assertEqual(caught.exception.code(), grpc.StatusCode.UNIMPLEMENTED)
        for option, code in [
            ({"suffix": "!"}, grpc.StatusCode.UNIMPLEMENTED),
            ({"n": 0}, grpc.StatusCode.INVALID_ARGUMENT),
        ]:
            with self.assertRaises(grpc.RpcError) as caught:
                list(
                    self.open_call(
                        "completion",
                        self.body("completion", False) | option,
                        "grpc",
                    )
                )
            self.assertEqual(caught.exception.code(), code)
        self.generate("text", False)

    def test_concurrent_clients(self):
        # Different prompts make cross-delivery observable, not merely two successes.
        first = self.body("text", False)
        second = self.body("completion", False) | {"prompt": "The capital of France is"}
        # Distinct forced tokens make cross-delivery observable without depending
        # on identical floating-point results for single vs. batched decoding.
        for params, word in ((first["sampling_params"], " apple"), (second, " banana")):
            token = self.tokenizer.encode(word, add_special_tokens=False)[-1]
            self.assertNotIn(token, self.tokenizer.all_special_ids)
            params["logit_bias"] = {str(token): 100}
        expected = [
            self.generate("text", False, body=first),
            self.generate("completion", False, "grpc", second),
        ]
        self.assertNotEqual(expected[0]["output"], expected[1]["output"][0])
        with ThreadPoolExecutor(max_workers=2) as pool:
            pending = [
                pool.submit(self.generate, "text", False, "http", first),
                pool.submit(self.generate, "completion", False, "grpc", second),
            ]
            self.assertEqual([f.result(timeout=40) for f in pending], expected)

    def assert_cancelled(self, api, body, transport, choices=1):
        offset = len(self.logs())
        call = self.open_call(api, body, transport)
        try:
            seen = set()
            for frame in self.frames(call, api, True, transport):
                if api == "text":
                    self.assertIsNone(
                        frame["meta_info"]["finish_reason"],
                        "Generation finished before cancellation",
                    )
                    if frame["text"]:
                        break
                else:
                    for choice in frame["choices"]:
                        self.assertIsNone(
                            choice.get("finish_reason"),
                            "Choice finished before cancellation",
                        )
                        if choice["text"]:
                            seen.add(choice["index"])
                    if len(seen) == choices:
                        break
            else:
                self.fail("No live generation to cancel")
        finally:
            call.close() if transport == "http" else call.cancel()

        def aborted():
            logs = self.logs()[offset:]
            self.assertNotIn("abort dropped:", logs)
            self.assertNotIn("abort encode failed", logs)
            ids = set(
                re.findall(
                    r"Abort (?:running|queued) request\. req\.rid=['\"]([^'\"]+)", logs
                )
            )
            if "rid" in body:
                ids = {rid for rid in ids if rid.split("#")[0] == body["rid"]}
            return len(ids) == choices

        wait_for(
            aborted, f"{transport}/{api}: scheduler cancellation of {choices} choice(s)"
        )
        self.generate("text", False)

    def test_cancellation(self):
        for transport in ("http", "grpc"):
            body = self.body("text", True)
            body["rid"] = f"cancel-{uuid.uuid4().hex}"
            body["sampling_params"].update(max_new_tokens=2048, ignore_eos=True)
            self.assert_cancelled("text", body, transport)
            # Force a normal token so neither OpenAI choice finishes before cancellation.
            token = self.tokenizer.encode(" apple", add_special_tokens=False)[-1]
            self.assertNotIn(token, self.tokenizer.all_special_ids)
            body = self.body("completion", True) | {
                "n": 2,
                "logit_bias": {str(token): 100},
            }
            self.assertEqual(
                self.generate("completion", True, transport, body)["finish_reason"],
                ["length", "length"],
            )
            body["max_tokens"] = 2048
            self.assert_cancelled("completion", body, transport, choices=2)

    def stop_and_assert(self):
        self.process.terminate()
        self.process.wait(timeout=30)
        wait_for(
            lambda: all(
                not p.is_running() or p.status() == psutil.STATUS_ZOMBIE
                for p in self.owned_children
            ),
            "owned scheduler processes exiting",
            timeout=15,
        )
        for port in (int(self.base_url.rsplit(":", 1)[1]), self.grpc_port):
            with socket.socket() as probe:
                probe.settimeout(1)
                self.assertNotEqual(probe.connect_ex(("127.0.0.1", port)), 0)

    def test_shutdown_and_restart(self):
        calls = []
        try:
            for transport in ("http", "grpc"):
                body = self.body("text", True)
                body["sampling_params"].update(max_new_tokens=2048, ignore_eos=True)
                call = self.open_call("text", body, transport)
                reader = self.frames(call, "text", True, transport)
                calls.append((transport, call, reader))
                frame = next(reader)
                self.assertIsNone(frame["meta_info"]["finish_reason"])
            self.stop_and_assert()
            for transport, call, _ in calls:
                try:
                    list(call.iter_content() if transport == "http" else call)
                except (
                    requests.RequestException,
                    grpc.RpcError,
                ):
                    pass  # Bounded cancellation, not a promise to finish generation.
        finally:
            for transport, call, _ in calls:
                call.close() if transport == "http" else call.cancel()
        self.restart_and_check()
        self.stop_and_assert()  # Also check idle shutdown on the rebound ports.
        self.restart_and_check()

    def restart_and_check(self):
        self.channel.close()
        self.server_log.close()
        type(self).launch()
        self.generate("text", False)
        self.generate("text", False, "grpc")


if __name__ == "__main__":
    unittest.main()
