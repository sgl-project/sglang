import io
import math
import sys
import unittest
from argparse import Namespace
from contextlib import redirect_stderr
from unittest.mock import patch

from sglang.benchmark import serving
from sglang.benchmark.rust_client import build_config
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestBenchServingRequestRate(unittest.TestCase):
    def test_invalid_rates_fail_before_benchmark_setup(self):
        for client in ([], ["--rust-client"]):
            for rate in ("0", "-1", "nan", "-inf"):
                argv = ["serving", f"--request-rate={rate}", *client]
                with self.subTest(client=client, rate=rate):
                    with (
                        patch.object(sys, "argv", argv),
                        patch.object(serving, "run_benchmark") as run,
                        redirect_stderr(io.StringIO()) as stderr,
                        self.assertRaises(SystemExit) as error,
                    ):
                        serving.cli_main()

                    self.assertEqual(error.exception.code, 2)
                    self.assertIn("argument --request-rate:", stderr.getvalue())
                    run.assert_not_called()

    def test_positive_and_unpaced_rates_reach_benchmark(self):
        cases = [
            ([], math.inf),
            (["--request-rate", "inf"], math.inf),
            (["--request-rate", "0.5"], 0.5),
        ]
        for flags, expected in cases:
            with (
                self.subTest(flags=flags),
                patch.object(sys, "argv", ["serving", *flags]),
                patch.object(serving, "run_benchmark") as run,
            ):
                serving.cli_main()
                run.assert_called_once()
                self.assertEqual(run.call_args.args[0].request_rate, expected)

    def test_rust_config_preserves_the_sign_of_infinity(self):
        self.assertEqual(
            build_config(Namespace(request_rate=math.inf))["request_rate"], "inf"
        )
        self.assertEqual(
            build_config(Namespace(request_rate=-math.inf))["request_rate"], "-inf"
        )


if __name__ == "__main__":
    unittest.main()
