"""Offline throughput floor of the SRT engine, kept on H100: the floor is
H100-tuned (the RTX 5090 reaches ~2970 tok/s)."""

import unittest

from sglang.benchmark.offline_throughput import BenchArgs, throughput_test
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST, CustomTestCase

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-large")
register_amd_ci(est_time=40, suite="stage-b-test-1-gpu-small-amd")


class TestSRTEngineThroughput(CustomTestCase):
    def test_engine_offline_throughput(self):
        server_args = ServerArgs(
            model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
        )
        bench_args = BenchArgs(num_prompts=10)
        result = throughput_test(server_args=server_args, bench_args=bench_args)
        self.assertGreater(result["total_throughput"], 3000)


if __name__ == "__main__":
    unittest.main()
