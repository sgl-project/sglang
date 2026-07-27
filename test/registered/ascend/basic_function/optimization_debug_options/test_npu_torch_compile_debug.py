import time
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_8B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.few_shot_gsm8k import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="full-1-npu-a3", nightly=True)


class TestEnableTorchCompileDebugMode(CustomTestCase):
    """
    Testcase: When --enable-torch-compile-debug-mode is enabled, the overall inference duration increases compared to when it is disabled.

    [Test Category] Parameter
    [Test Target] --enable-torch-compile-debug-mode
    """

    model = QWEN3_8B_WEIGHTS_PATH
    other_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.7",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--disable-radix-cache",
        "--enforce-piecewise-cuda-graph",
    ]
    enable_args = [
        "--enable-torch-compile-debug-mode",
        "--enforce-piecewise-cuda-graph",
        "--piecewise-cuda-graph-max-tokens",
        "64",
    ]

    def setUp(self):
        self.base_url = DEFAULT_URL_FOR_TEST
        self.process = None

    def tearDown(self):
        if hasattr(self, "process") and self.process and self.process.pid:
            try:
                kill_process_tree(self.process.pid)
                self.process = None
            except Exception:
                pass

    def benchmark_gsm8k(self, args, num_runs=5):
        run_times = []
        for _ in range(num_runs):
            start_time = time.perf_counter()
            run_eval(args)
            end_time = time.perf_counter()
            elapsed_time = round(end_time - start_time, 6)
            run_times.append(elapsed_time)

        avg_time = sum(run_times) / len(run_times)
        return avg_time, run_times

    def test_enable_torch_compile_debug_mode(self):
        # Second run: with debug mode enabled
        self.process = popen_launch_server(
            self.model,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=self.other_args + self.enable_args,
        )

        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        avg_time1, all_times1 = self.benchmark_gsm8k(args, num_runs=5)

        # Clean up first process
        self.tearDown()
        time.sleep(5)

        # First run: without debug mode
        self.process = popen_launch_server(
            self.model,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=self.other_args,
        )

        avg_time2, all_times2 = self.benchmark_gsm8k(args, num_runs=5)

        print("run_gsm8k_time1:", all_times1)
        print("run_gsm8k_time2:", all_times2)
        print("run_gsm8k_avg_time1:", avg_time1)
        print("run_gsm8k_avg_time2:", avg_time2)
        # Assertion: Debug mode should be slower
        self.assertGreater(
            avg_time1,
            avg_time2,
            f"Debug mode should be slower, but measured time: "
            f"normal mode={avg_time2}s, debug mode={avg_time1}s",
        )


if __name__ == "__main__":
    unittest.main()
