import asyncio
import json
import os
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import sglang.bench_serving as bench_serving
from sglang.bench_serving import RequestFuncOutput, run_benchmark
from sglang.benchmark.datasets import DatasetRow
from sglang.benchmark.utils import parse_custom_headers
from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    get_benchmark_args,
    popen_launch_server,
)

register_cuda_ci(est_time=300, suite="nightly-1-gpu", nightly=True)
register_amd_ci(est_time=300, suite="nightly-amd-1-gpu", nightly=True)

MODEL = "Qwen/Qwen3-0.6B"
NUM_CONVERSATIONS, NUM_TURNS = 4, 3


class TestBenchServingFunctionality(CustomTestCase):
    def test_gsp_multi_turn(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            process = popen_launch_server(
                MODEL,
                DEFAULT_URL_FOR_TEST,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--mem-fraction-static",
                    "0.7",
                    "--log-requests",
                    "--log-requests-level",
                    "3",
                    "--log-requests-format",
                    "json",
                    "--log-requests-target",
                    "stdout",
                    temp_dir,
                ],
            )
            try:
                args = get_benchmark_args(
                    base_url=DEFAULT_URL_FOR_TEST,
                    backend="sglang-oai-chat",
                    tokenizer=MODEL,
                    dataset_name="generated-shared-prefix",
                    num_prompts=NUM_CONVERSATIONS,
                    request_rate=float("inf"),
                    gsp_num_groups=2,
                    gsp_prompts_per_group=2,
                    gsp_system_prompt_len=64,
                    gsp_question_len=16,
                    gsp_output_len=16,
                    gsp_num_turns=NUM_TURNS,
                )
                args.warmup_requests = 0
                res = run_benchmark(args)
                self.assertEqual(res["completed"], NUM_CONVERSATIONS * NUM_TURNS)

                time.sleep(1)
                logs = "".join(f.read_text() for f in Path(temp_dir).glob("*.log"))
                self._verify_multi_turn_logs(logs)
            finally:
                kill_process_tree(process.pid)

    def _verify_multi_turn_logs(self, content: str):
        reqs = []
        for line in content.splitlines():
            idx = line.find("{")
            if idx == -1:
                continue
            try:
                obj = json.loads(line[idx:])
            except json.JSONDecodeError:
                continue
            if obj.get("event") != "request.finished":
                continue
            text = obj.get("obj", {}).get("text")
            rid = obj.get("rid", "")
            if text and not rid.startswith(HEALTH_CHECK_RID_PREFIX):
                reqs.append(text)

        self.assertGreaterEqual(len(reqs), NUM_CONVERSATIONS * NUM_TURNS)

        # Verify prefix relationships
        reqs_sorted = sorted(reqs, key=len)
        prefix_count = 0
        for i, text in enumerate(reqs_sorted):
            for j in range(i + 1, len(reqs_sorted)):
                if reqs_sorted[j].startswith(text):
                    prefix_count += 1
                    break

        expected = NUM_CONVERSATIONS * (NUM_TURNS - 1)
        self.assertGreaterEqual(
            prefix_count, expected, f"Expected at least {expected} prefix pairs"
        )


class TestBenchServingCustomHeaders(CustomTestCase):
    def test_parse_custom_headers(self):
        headers = parse_custom_headers(["MyHeader=MY_VALUE", "Another=value=hello"])
        self.assertEqual(headers, {"MyHeader": "MY_VALUE", "Another": "value=hello"})

        headers = parse_custom_headers(["InvalidNoEquals"])
        self.assertEqual(headers, {})

        headers = parse_custom_headers(["=NoKey"])
        self.assertEqual(headers, {})

    # TODO: Using well-implemented mock server, e.g. the on in sgl-router
    def test_custom_headers_sent_to_server(self):
        import queue

        received_requests = queue.Queue()

        class HeaderEchoHandler(BaseHTTPRequestHandler):
            def _handle(self):
                received_requests.put(
                    {
                        "method": self.command,
                        "path": self.path,
                        "headers": dict(self.headers),
                    }
                )
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                if self.path == "/v1/models":
                    self.wfile.write(json.dumps({"data": [{"id": "gpt2"}]}).encode())
                elif self.path == "/generate":
                    self.wfile.write(
                        json.dumps(
                            {"text": "ok", "meta_info": {"completion_tokens": 1}}
                        ).encode()
                    )
                else:
                    self.wfile.write(json.dumps({}).encode())

            do_GET = do_POST = _handle

        server = HTTPServer(("127.0.0.1", 0), HeaderEchoHandler)
        port = server.server_address[1]
        server_thread = threading.Thread(target=server.serve_forever)
        server_thread.daemon = True
        server_thread.start()

        try:
            args = get_benchmark_args(
                base_url=f"http://127.0.0.1:{port}",
                backend="sglang",
                dataset_name="random",
                tokenizer="gpt2",
                num_prompts=1,
                random_input_len=8,
                random_output_len=8,
                header=["X-Custom-Test=TestValue123", "X-Another=AnotherVal"],
            )
            args.warmup_requests = 0
            args.disable_tqdm = True
            run_benchmark(args)
        except Exception:
            pass
        finally:
            server.shutdown()

        all_reqs = []
        while not received_requests.empty():
            all_reqs.append(received_requests.get_nowait())

        generate_reqs = [r for r in all_reqs if r["path"] == "/generate"]
        self.assertGreater(
            len(generate_reqs),
            0,
            f"No /generate request. All: {[r['path'] for r in all_reqs]}",
        )
        headers = generate_reqs[0]["headers"]
        self.assertEqual(headers.get("X-Custom-Test"), "TestValue123")
        self.assertEqual(headers.get("X-Another"), "AnotherVal")


    def test_flush_cache_keeps_allocator_cache_by_default(self):
        """bench_serving --flush-cache should pass empty_cache=False by default."""

        async def fake_request(request_func_input, pbar=None):
            return RequestFuncOutput(
                generated_text="ok",
                success=True,
                latency=0.02,
                ttft=0.01,
                itl=[0.01],
                prompt_len=request_func_input.prompt_len,
                output_len=2,
                start_time=time.perf_counter(),
            )

        class FakeTokenizer:
            def encode(self, text, add_special_tokens=False):
                return [1, 2]

        class FakeResponse:
            status_code = 404

            def json(self):
                return {}

        previous_request_func = bench_serving.ASYNC_REQUEST_FUNCS["sglang"]
        with tempfile.NamedTemporaryFile(delete=False) as f:
            output_file = f.name
        bench_serving.args = SimpleNamespace(
            backend="sglang",
            dataset_name="random",
            sharegpt_output_len=None,
            random_input_len=8,
            random_output_len=2,
            random_range_ratio=0.0,
            num_prompts=1,
            warmup_requests=0,
            output_file=output_file,
            output_details=False,
            plot_throughput=False,
        )

        try:
            bench_serving.ASYNC_REQUEST_FUNCS["sglang"] = fake_request
            with patch.object(bench_serving.requests, "post") as post, patch.object(
                bench_serving.requests, "get", return_value=FakeResponse()
            ):
                asyncio.run(
                    bench_serving.benchmark(
                        backend="sglang",
                        api_url="http://127.0.0.1:30000/generate",
                        base_url="http://127.0.0.1:30000",
                        model_id="dummy",
                        tokenizer=FakeTokenizer(),
                        input_requests=[DatasetRow("hi", 1, 2)],
                        request_rate=float("inf"),
                        max_concurrency=None,
                        disable_tqdm=True,
                        lora_names=[],
                        lora_request_distribution=None,
                        lora_zipf_alpha=None,
                        extra_request_body={},
                        profile=False,
                        flush_cache=True,
                        warmup_requests=0,
                    )
                )

            post.assert_called_once()
            self.assertEqual(post.call_args.kwargs["params"], {"empty_cache": False})
        finally:
            bench_serving.ASYNC_REQUEST_FUNCS["sglang"] = previous_request_func
            os.remove(output_file)


if __name__ == "__main__":
    unittest.main()
