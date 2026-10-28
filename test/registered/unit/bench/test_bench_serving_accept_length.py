"""Unit tests for the speculative accept length bench_serving reports."""

import asyncio
import contextlib
import io
import json
import tempfile
import threading
import unittest
from argparse import Namespace
from http.server import BaseHTTPRequestHandler, HTTPServer

from sglang.benchmark.datasets.common import DatasetRow
from sglang.benchmark.serving import (
    RequestFuncOutput,
    benchmark,
    calculate_run_accept_length,
    set_global_args,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class _SpecServerHandler(BaseHTTPRequestHandler):
    """Native /generate with per-request spec meta_info, plus lifetime counters
    served as /server_info avg_spec_accept_length, as the scheduler keeps them."""

    def do_GET(self):  # noqa: N802
        assert self.path == "/server_info", self.path
        srv = self.server
        avg = srv.total_tokens / srv.total_verify_ct
        body = json.dumps({"internal_states": [{"avg_spec_accept_length": avg}]})
        self._reply("application/json", body.encode())

    def do_POST(self):  # noqa: N802
        self.rfile.read(int(self.headers["Content-Length"]))
        srv = self.server
        tokens, verify_ct = srv.tokens_per_request, srv.verify_ct_per_request
        srv.total_tokens += tokens
        srv.total_verify_ct += verify_ct
        meta_info = {
            "completion_tokens": tokens,
            "spec_accept_length": tokens / verify_ct,
            "spec_verify_ct": verify_ct,
        }
        if not srv.send_verify_ct:
            del meta_info["spec_verify_ct"]
        chunk = {"text": "x" * tokens, "meta_info": meta_info}
        payload = f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode()
        self._reply("text/event-stream", payload)

    def _reply(self, content_type, payload):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt, *args):
        pass


class _Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text)


def _bench_args(**overrides):
    args = dict(
        backend="sglang",
        disable_stream=False,
        disable_ignore_eos=False,
        temperature=0.0,
        top_p=1.0,
        return_logprob=False,
        return_routed_experts=False,
        logprob_start_len=-1,
        top_logprobs_num=0,
        token_ids_logprob=None,
        print_requests=False,
        header=None,
        cache_report=False,
        plot_throughput=False,
        goodput=None,
        output_file=None,
        output_details=False,
        dataset_name="random",
        num_prompts=4,
        random_input_len=8,
        random_output_len=41,
        random_range_ratio=1.0,
        sharegpt_output_len=None,
        warmup_requests=1,
        profile_num_steps=None,
        mooncake_num_rounds=1,
    )
    args.update(overrides)
    return Namespace(**args)


class TestCalculateRunAcceptLength(CustomTestCase):
    def test_weights_requests_by_verify_steps(self):
        outputs = [
            RequestFuncOutput(
                success=True, output_len=400, spec_accept_length=4.0, spec_verify_ct=100
            ),
            RequestFuncOutput(
                success=True, output_len=20, spec_accept_length=2.0, spec_verify_ct=10
            ),
        ]
        # 420 tokens in 110 verify steps, not the 3.0 mean of the two requests.
        self.assertAlmostEqual(calculate_run_accept_length(outputs), 420 / 110)

    def test_recovers_verify_steps_from_output_len(self):
        outputs = [
            RequestFuncOutput(success=True, output_len=400, spec_accept_length=4.0),
            RequestFuncOutput(success=True, output_len=20, spec_accept_length=2.0),
        ]
        self.assertAlmostEqual(calculate_run_accept_length(outputs), 420 / 110)

    def test_skips_failed_and_non_speculative_requests(self):
        outputs = [
            RequestFuncOutput(
                success=True, output_len=30, spec_accept_length=3.0, spec_verify_ct=10
            ),
            RequestFuncOutput(
                success=False, output_len=0, spec_accept_length=9.0, spec_verify_ct=1
            ),
            RequestFuncOutput(success=True, output_len=50),
        ]
        self.assertAlmostEqual(calculate_run_accept_length(outputs), 3.0)

    def test_none_without_speculative_metadata(self):
        self.assertIsNone(calculate_run_accept_length([]))
        self.assertIsNone(
            calculate_run_accept_length([RequestFuncOutput(success=True, output_len=8)])
        )


class TestBenchmarkReportsThisRun(CustomTestCase):
    def setUp(self):
        self.server = HTTPServer(("127.0.0.1", 0), _SpecServerHandler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        # An earlier workload on the same server accepted 4.54 tokens per step.
        self.server.total_tokens = 4540
        self.server.total_verify_ct = 1000
        self.server.tokens_per_request = 41
        self.server.verify_ct_per_request = 10
        self.server.send_verify_ct = True

    def _run(self):
        output_dir = tempfile.TemporaryDirectory()
        self.addCleanup(output_dir.cleanup)
        args = _bench_args(output_file=f"{output_dir.name}/result.jsonl")
        set_global_args(args)
        base_url = f"http://127.0.0.1:{self.server.server_port}"
        rows = [
            DatasetRow(prompt="hello", prompt_len=8, output_len=41)
            for _ in range(args.num_prompts)
        ]
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            result = asyncio.run(
                benchmark(
                    backend="sglang",
                    api_url=base_url + "/generate",
                    base_url=base_url,
                    model_id="dummy-model",
                    tokenizer=_Tokenizer(),
                    input_requests=rows,
                    request_rate=float("inf"),
                    max_concurrency=None,
                    disable_tqdm=True,
                    lora_names=[],
                    lora_request_distribution=None,
                    lora_zipf_alpha=None,
                    extra_request_body={},
                    profile=False,
                    warmup_requests=args.warmup_requests,
                )
            )
        return result, stdout.getvalue()

    def test_back_to_back_runs_report_their_own_accept_length(self):
        """The reported accept length must not include earlier work on the server."""
        with self.subTest("server sends spec_verify_ct"):
            result, stdout = self._run()
            self.assertAlmostEqual(result["accept_length"], 4.1)
            self.assertEqual(result["accept_length_source"], "per_request")
            self.assertIn("Accept length:                           4.10", stdout)
            # The lifetime counter still remembers the earlier 4.54 workload.
            self.assertGreater(result["server_avg_spec_accept_length"], 4.3)

        with self.subTest("older server without spec_verify_ct"):
            self.server.send_verify_ct = False
            self.server.tokens_per_request = 30
            result, _ = self._run()
            self.assertAlmostEqual(result["accept_length"], 3.0)
            self.assertEqual(result["accept_length_source"], "per_request")


if __name__ == "__main__":
    unittest.main()
