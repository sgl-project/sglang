"""
XPU graph tests: verifies decode full-graph and prefill tc_piecewise graph
on Intel XPU produce valid outputs.

  - TestXPUGraph : decode full-graph and prefill tc_piecewise graph enabled
    together in a single bench_one_batch invocation.

Usage:
    python3 -m unittest test_xpu_graph.TestXPUGraph
"""

import unittest

from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN,
    CustomTestCase,
    is_in_ci,
    run_bench_one_batch,
)

register_xpu_ci(est_time=600, suite="stage-b-test-1-gpu-xpu")

_COMMON_ARGS = [
    "--device",
    "xpu",
    "--attention-backend",
    "triton",
    "--disable-radix-cache",
    "--mem-fraction-static",
    "0.6",
    "--batch-size",
    "1",
]

_CI_IO_ARGS = ["--input", "64", "--output", "4"]
_FULL_IO_ARGS = ["--input", "128", "--output", "16"]


class TestXPUGraph(CustomTestCase):
    """Decode full-graph + prefill tc_piecewise together."""

    def test_full_graph_runs(self):
        args = [
            *_COMMON_ARGS,
            "--cuda-graph-config",
            '{"decode":{"backend":"full"},"prefill":{"backend":"tc_piecewise","tc_compiler":"eager"}}',
            "--cuda-graph-bs-prefill",
            "64",
            "128",
        ]
        if is_in_ci():
            args += _CI_IO_ARGS
        else:
            args += _FULL_IO_ARGS

        prefill_latency, decode_throughput, _ = run_bench_one_batch(
            DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN, args
        )
        self.assertGreater(
            prefill_latency,
            0,
            "prefill latency must be > 0 with tc_piecewise XPU graph",
        )
        self.assertGreater(
            decode_throughput, 0, "decode throughput must be > 0 with full XPU graph"
        )


if __name__ == "__main__":
    unittest.main()
