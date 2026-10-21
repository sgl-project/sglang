"""PD routing and execution-geometry regressions for the native SPS profiler."""

import json
import random
import threading
import unittest
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from sglang.benchmark import dspark_sps_profiler as mod
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@contextmanager
def pd_endpoints(*, logical_rows=12, use_graph=True):
    calls = []
    generating = threading.Event()

    def handler(role):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                self.respond(None)

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length)) if length else None
                self.respond(payload)

            def respond(self, payload):
                calls.append((role, self.path, payload))
                if role == "router" and self.path in ("/generate", "/flush_cache"):
                    if self.path == "/generate":
                        generating.set()
                    body = {}
                elif role == "decode" and self.path == "/set_internal_state":
                    body = {"updated": True}
                elif role == "decode" and self.path == "/abort_request":
                    body = {}
                elif role == "decode" and self.path == "/server_info":
                    records = (
                        [
                            {
                                "forward_ct": i,
                                "num_running_reqs": 4,
                                "num_verify_tokens": 18,
                                "verify_tokens_logical": logical_rows,
                                "verify_tokens_actual": 18,
                                "verify_used_cuda_graph": use_graph,
                                "step_cpu_ms": 100.0,
                            }
                            for i in range(1, 17)
                        ]
                        if generating.is_set()
                        else []
                    )
                    body = {
                        "speculative_algorithm": "DSPARK",
                        "tokenizer_path": "unused-test-tokenizer",
                        "tp_size": 8,
                        "internal_states": [
                            {
                                "dp_size": 1,
                                "effective_max_running_requests_per_dp": 4,
                                "cuda_graph_config": {"decode": {"bs": [1, 2, 4]}},
                                mod.INFO_RECORD_PAYLOAD_KEY: {
                                    "mode": "compact",
                                    "components": ["core", "step_cpu_time"],
                                    "simulate_acc_len": 1.0,
                                    "verify_num_draft_tokens": 6,
                                    "records": records,
                                },
                            }
                        ],
                    }
                else:
                    self.send_error(404, "wrong PD endpoint")
                    return
                data = json.dumps(body).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        return Handler

    servers = [
        ThreadingHTTPServer(("127.0.0.1", 0), handler(role))
        for role in ("router", "decode")
    ]
    threads = [
        threading.Thread(target=s.serve_forever, kwargs={"poll_interval": 0.01})
        for s in servers
    ]
    for thread in threads:
        thread.start()
    try:
        yield [f"http://127.0.0.1:{s.server_port}" for s in servers], calls
    finally:
        for server in servers:
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join()


class TestPdProfilerExecution(CustomTestCase):
    def run_round(self, urls):
        context = mod.fetch_server_context(
            base_url=urls[0],
            control_url=urls[1],
            local_tokenizer_path=None,
            allowed_modes=("compact",),
        )
        return mod.run_one_round(
            context=context,
            vocab_size=32000,
            batch_size_per_rank=4,
            settings=mod.RoundSettings(
                input_len=16,
                temperature=1.0,
                min_steady_steps=2,
                min_steady_seconds=0.1,
                round_timeout_seconds=5,
            ),
            rng=random.Random(42),
            frac=0.4,
        )

    def test_generation_uses_router_and_controls_use_decode(self):
        with pd_endpoints() as (urls, calls):
            outcome = self.run_round(urls)
        self.assertEqual(outcome.batch_tokens, 12)
        self.assertAlmostEqual(outcome.steps_per_sec, 10.0)
        self.assertIn(("router", "/flush_cache", None), calls)
        self.assertTrue(any(r == "router" and p == "/generate" for r, p, _ in calls))
        self.assertTrue(
            any(r == "decode" and p == "/abort_request" for r, p, _ in calls)
        )
        pins = [
            body["server_args"]["dspark_force_budget_frac"]
            for role, path, body in calls
            if path == "/set_internal_state"
        ]
        self.assertEqual(pins, [0.4, None])

    def test_verify_all_cannot_be_recorded_as_a_smaller_budget(self):
        with pd_endpoints(logical_rows=24) as (urls, calls):
            with self.assertRaisesRegex(RuntimeError, "observed logical=24"):
                self.run_round(urls)
        pins = [
            body["server_args"]["dspark_force_budget_frac"]
            for _, path, body in calls
            if path == "/set_internal_state"
        ]
        self.assertEqual(pins, [0.4, None])

    def test_eager_fallback_is_rejected(self):
        with pd_endpoints(use_graph=False) as (urls, _):
            with self.assertRaisesRegex(RuntimeError, "fell back to eager"):
                self.run_round(urls)

    def test_graph_capacity_alone_does_not_prove_budget_execution(self):
        rows = [
            mod.SpsRow(
                forward_ct=i,
                num_running_reqs=4,
                num_verify_tokens=24,
                step_time=0.1,
            )
            for i in range(16)
        ]
        with self.assertRaisesRegex(RuntimeError, "reports execution geometry"):
            mod.postprocess_round(
                rank_rows=[rows],
                batch_size_per_rank=4,
                dp_size=1,
                verify_num_draft_tokens=6,
                min_steady_steps=2,
                frac=0.4,
                load_info=mod.LoadInfo(
                    num_requests=4,
                    max_new_tokens=100,
                    wall_seconds=2.0,
                    reached_target=True,
                ),
            )

    def test_fraction_rounding_matches_server_full_budget(self):
        for logical_rows in (54, 53):
            with self.subTest(logical_rows=logical_rows):
                rows = [
                    mod.SpsRow(
                        forward_ct=i,
                        num_running_reqs=12,
                        num_verify_tokens=60,
                        step_time=0.1,
                        verify_tokens_logical=logical_rows,
                        verify_tokens_actual=60,
                        verify_used_cuda_graph=True,
                    )
                    for i in range(16)
                ]
                kwargs = dict(
                    rank_rows=[rows],
                    batch_size_per_rank=12,
                    dp_size=1,
                    verify_num_draft_tokens=6,
                    min_steady_steps=2,
                    frac=0.7,
                    load_info=mod.LoadInfo(
                        num_requests=12,
                        max_new_tokens=100,
                        wall_seconds=2.0,
                        reached_target=True,
                    ),
                )
                if logical_rows == 54:
                    outcome = mod.postprocess_round(**kwargs)
                    self.assertEqual(outcome.batch_tokens, 54)
                else:
                    with self.assertRaisesRegex(
                        RuntimeError, "requested M=54, observed logical=53"
                    ):
                        mod.postprocess_round(**kwargs)


if __name__ == "__main__":
    unittest.main()
