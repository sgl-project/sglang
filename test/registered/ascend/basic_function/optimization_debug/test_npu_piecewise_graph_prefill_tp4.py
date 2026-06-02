import unittest

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    CustomTestCase,
    run_bench_serving,
)

register_npu_ci(est_time=400, suite="stage-b-test-4-npu-a3", nightly=False)
register_npu_ci(est_time=400, suite="nightly-4-npu-a3", nightly=True)

MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
INPUT_THROUGHPUT_THRESHOLD = 14600


class TestPiecewiseGraphPrefillTp4(CustomTestCase):

    def test_pcg_serving(self):
        print(f"##=== Testing PCG serving: {MODEL} ===##")

        pcg_server_args = [
            "--trust-remote-code",
            "--mem-fraction-static",
            0.7,
            "--max-running-requests",
            32,
            "--attention-backend",
            "ascend",
            "--disable-radix-cache",
            "--cuda-graph-max-bs",
            32,
            "--tp-size",
            4,
            "--chunked-prefill-size",
            2048,
            "--cuda-graph-bs-prefill",
            128,
            256,
            512,
            1024,
            2048,
        ]

        with envs.SGLANG_NPU_ENABLE_PIECEWISE_CUDA_GRAPH.override(True):
            res = run_bench_serving(
                model=MODEL,
                num_prompts=20,
                request_rate=float("inf"),
                other_server_args=pcg_server_args,
                random_input_len=4096,
                random_output_len=10,
            )

        self.assertEqual(res["completed"], 20)
        input_throughput = res["input_throughput"]
        mean_ttft_ms = res["mean_ttft_ms"]
        print(
            f"PCG input_throughput: {input_throughput} tok/s, mean_ttft: {mean_ttft_ms} ms"
        )
        self.assertGreater(input_throughput, INPUT_THROUGHPUT_THRESHOLD)


if __name__ == "__main__":
    unittest.main()
