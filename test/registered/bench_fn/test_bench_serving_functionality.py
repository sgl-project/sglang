import json
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from sglang.bench_serving import parse_custom_headers, run_benchmark
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    get_benchmark_args,
    popen_launch_server,
)

register_cuda_ci(est_time=300, suite="nightly-1-gpu", nightly=True)

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
            if not line.startswith("{"):
                continue
            obj = json.loads(line)
            if obj.get("event") != "request.finished":
                continue
            text = obj.get("obj", {}).get("text")
            rid = obj.get("rid", "")
            if text and not rid.startswith("HEALTH_CHECK"):
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
        headers = parse_custom_headers(["MyHeader=MY_VALUE", "Another=val=ue"])
        self.assertEqual(headers, {"MyHeader": "MY_VALUE", "Another": "val=ue"})

        headers = parse_custom_headers(["InvalidNoEquals"])
        self.assertEqual(headers, {})

        headers = parse_custom_headers(["=NoKey"])
        self.assertEqual(headers, {})

    def test_custom_headers_sent_to_server(self):
        # TODO: In the future, consider using router's mock server for more realistic testing.
        received_headers = {}

        class HeaderEchoHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                received_headers.update(self.headers)
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"headers": dict(self.headers)}).encode())

            def log_message(self, format, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), HeaderEchoHandler)
        port = server.server_address[1]
        server_thread = threading.Thread(target=server.handle_request)
        server_thread.start()

        try:
            args = get_benchmark_args(
                base_url=f"http://127.0.0.1:{port}",
                backend="sglang",
                dataset_name="random",
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
            server_thread.join(timeout=5)
            server.server_close()

        self.assertEqual(received_headers.get("X-Custom-Test"), "TestValue123")
        self.assertEqual(received_headers.get("X-Another"), "AnotherVal")


if __name__ == "__main__":
    unittest.main()
