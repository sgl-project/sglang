"""Throughput of torch.compile at batch size one across two GPUs."""

import unittest

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kits.perf_bench_kit import at_least, check_perf
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    CustomTestCase,
    run_bench_offline_throughput,
)

register_cuda_ci(est_time=75, stage="extra-a", runner_config="2-gpu-large")
register_amd_ci(est_time=280, suite="stage-b-test-2-gpu-large-amd")


class TestTorchCompileThroughput(CustomTestCase):
    def test_torch_compile_tp2_bs1(self):
        output_throughput = run_bench_offline_throughput(
            DEFAULT_MODEL_NAME_FOR_TEST,
            ["--tp", "2", "--enable-torch-compile", "--cuda-graph-max-bs-decode", "2"],
        )

        check_perf(
            self,
            at_least(
                "output_throughput", output_throughput, 255, amd=200, unit="token/s"
            ),
        )


if __name__ == "__main__":
    unittest.main()
