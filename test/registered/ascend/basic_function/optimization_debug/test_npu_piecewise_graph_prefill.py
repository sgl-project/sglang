import subprocess
import unittest

import torch

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.test_ascend_utils import (
    QWEN2_5_7B_INSTRUCT_WEIGHTS_PATH,
    write_results_to_github_step_summary,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    CustomTestCase,
    run_bench_one_batch,
)

register_npu_ci(est_time=400, suite="stage-b-test-1-npu-a2", nightly=False)
register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


TOKENS_TO_CAPTURE = [i for i in range(128, 4096, 128)]


@unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
class TestPiecewiseGraphPrefillCorrectness(GSM8KAscendMixin, CustomTestCase):
    model = QWEN2_5_7B_INSTRUCT_WEIGHTS_PATH
    other_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        0.8,
        "--attention-backend",
        "ascend",
        "--cuda-graph-bs",
        128,
        "--cuda-graph-backend-prefill=tc_piecewise",
        "--cuda-graph-bs-prefill",
        *TOKENS_TO_CAPTURE,
    ]
    accuracy = 0.84
    num_questions = 1319


@unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
class TestPiecewiseGraphPrefillBenchmark(CustomTestCase):
    model = QWEN2_5_7B_INSTRUCT_WEIGHTS_PATH
    other_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        0.8,
        "--attention-backend",
        "ascend",
        "--cuda-graph-backend-prefill=tc_piecewise",
        "--cuda-graph-bs-prefill",
    ] + TOKENS_TO_CAPTURE

    latency = 0.045

    def test_latency(self):
        print(f"##=== Testing prefill latency: {self.model} ===##")
        model_metrics = {
            "server": subprocess.list2cmdline(map(str, self.other_args)),
            "client": "bench_one_batch",
            "latency_threshold": self.latency,
        }
        try:
            prefill_latency, _, _ = run_bench_one_batch(
                self.model,
                other_args=self.other_args,
            )
            model_metrics["latency"] = float(prefill_latency)
            self.assertLess(prefill_latency, self.latency)
        except Exception as e:
            model_metrics["error"] = e
            print(f"Error testing {self.model}: {e}")
            self.fail(f"Test failed for {self.model}: {e}")
        finally:
            write_results_to_github_step_summary({self.model: model_metrics})


if __name__ == "__main__":
    unittest.main()
