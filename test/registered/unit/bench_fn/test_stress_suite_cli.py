"""Regressions for stress-suite readiness probes and canonical goodput SLOs."""

import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sglang.benchmark import stress_suite
from sglang.benchmark.goodput import calculate_goodput
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestStressSuiteCli(CustomTestCase):
    def args(self, *extra):
        parser = stress_suite.build_parser()
        args = parser.parse_args(["--base-url", "http://127.0.0.1:30000", *extra])
        stress_suite.validate_args(parser, args)
        return args

    def test_probe_does_not_require_generation(self):
        """A ready server without recent output must not fail the health gate."""
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                requests.append(self.path)
                self.send_response(200 if self.path == "/ready" else 503)
                self.end_headers()

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            check = stress_suite.check_health(f"http://127.0.0.1:{server.server_port}")
        finally:
            server.shutdown()
            thread.join()
            server.server_close()
        self.assertTrue(check["ok"])
        self.assertEqual(requests, ["/ready"])
        self.assertEqual(stress_suite.judge_health([check], 0), "")

    def test_goodput_slos_apply_to_each_request(self):
        """One request missing a joint SLO lowers attainment, even if others pass."""
        args = self.args(
            "--goodput",
            "ttft:500",
            "tpot:50",
            "e2el:1000",
            "--min-slo-attainment",
            "0.75",
        )
        outputs = [
            SimpleNamespace(success=True, ttft=0.4, latency=0.8, output_len=11),
            SimpleNamespace(success=True, ttft=0.4, latency=0.95, output_len=11),
        ]
        goodput = calculate_goodput(outputs, dict(args.goodput), dur_s=2)
        phase = {"planned": 2, "completed": 2, "slo_attainment": goodput.slo_attainment}
        self.assertEqual(goodput.slo_attainment, 0.5)
        verdict, reason = stress_suite.judge_phase(args, phase)
        self.assertEqual(verdict, "FAIL")
        self.assertIn("50.00%", reason)
        args.min_slo_attainment = 0.5
        self.assertEqual(stress_suite.judge_phase(args, phase), ("PASS", ""))
        phase.pop("slo_attainment")
        self.assertEqual(stress_suite.judge_phase(args, phase)[0], "FAIL")

        workload = stress_suite.build_workload(args)
        command = stress_suite.build_command(
            args, Path("workload.json"), Path("result.jsonl"), workload
        )
        start = command.index("--goodput") + 1
        self.assertEqual(
            command[start : start + 3], ["ttft:500", "tpot:50", "e2el:1000"]
        )

    def test_prompt_seed_is_recorded_and_replayable(self):
        """Repeating a traffic seed must not silently warm the prefix cache."""
        args = self.args("--profile", "quick")
        with patch.object(stress_suite.secrets, "randbits", side_effect=[123, 456]):
            first = stress_suite.build_workload(args)
            second = stress_suite.build_workload(args)
        self.assertEqual(first["seed"], second["seed"])
        self.assertEqual(first["phases"], second["phases"])
        self.assertNotEqual(first["prompt_seed"], second["prompt_seed"])
        replay = stress_suite.build_workload(
            self.args("--profile", "quick", "--prompt-seed", str(first["prompt_seed"]))
        )
        self.assertEqual(replay, first)
        # Serving warmup reuses the first measured prompt; the suite must not
        # silently prime that prompt before an uncached-prefill scenario.
        command = stress_suite.build_command(
            args, Path("workload.json"), Path("result.jsonl"), first
        )
        self.assertEqual(command[command.index("--warmup-requests") + 1], "0")


if __name__ == "__main__":
    unittest.main()
