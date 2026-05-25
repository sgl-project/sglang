import time
import unittest
from urllib.parse import urlparse

from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK_DATASET_DEFAULT,
    BENCHMARK_TOOL_DEFAULT,
    DEEPSEEK_R1_W8A8_MODEL_PATH,
    run_aisbench,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import kill_process_tree, popen_launch_server

register_npu_ci(
    est_time=2400,
    suite="nightly-16-npu-a3",
    nightly=True,
    disabled=False,
)

DEEPSEEK_R1_BASE_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "16",
    "HCCL_BUFFSIZE": "1600",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
}

DEEPSEEK_R1_BASE_ARGS = [
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--tp-size",
    16,
    "--dp-size",
    1,
    "--trust-remote-code",
    "--mem-fraction-static",
    0.79,
    "--chunked-prefill-size",
    64000,
    "--context-length",
    66000,
    "--max-prefill-tokens",
    66000,
    "--max-total-tokens",
    66000,
    "--disable-radix-cache",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--quantization",
    "modelslim",
]

TEST_PARAMS = {
    "max_concurrency": 64,
    "num_prompts": 256,
    "input_len": 2048,
    "output_len": 1024,
    "random_range_ratio": 1,
    "benchmark_tool": BENCHMARK_TOOL_DEFAULT,
    "aisbench_dataset_type": AISBENCHMARK_DATASET_DEFAULT,
}


class TestNPUDeepSeekR1_W8A8_MLAPO(unittest.TestCase):
    """Test enabling MLAPO optimization on the DeepSeek R1 W8A8 model reduces the TPOT
    by at least 2 milliseconds compared to the baseline performance

    [Test Category] Feature
    [Test Target] SGLANG_NPU_USE_MLAPO
    """
    baseline_tpot = None
    base_url = "http://127.0.0.1:20166"
    parsed_url = urlparse(base_url)
    host = parsed_url.hostname
    port = parsed_url.port
    process = None

    def tearDown(self):
        if self.process:
            try:
                kill_process_tree(self.process.pid)
            except Exception:
                pass
        self.process = None

    def _launch_server_and_run_test(self, extra_envs=None):

        env = DEEPSEEK_R1_BASE_ENVS.copy()
        if extra_envs:
            env.update(extra_envs)

        self.process = popen_launch_server(
            DEEPSEEK_R1_W8A8_MODEL_PATH,
            self.base_url,
            timeout=1800,
            other_args=DEEPSEEK_R1_BASE_ARGS,
            env=env,
        )

        metrics = run_aisbench(
            host=self.host,
            port=str(self.port),
            model_path=DEEPSEEK_R1_W8A8_MODEL_PATH,
            dataset_type=TEST_PARAMS["aisbench_dataset_type"],
            dataset_path=None,
            input_len=TEST_PARAMS["input_len"],
            output_len=TEST_PARAMS["output_len"],
            max_concurrency=TEST_PARAMS["max_concurrency"],
            num_prompts=TEST_PARAMS["num_prompts"],
            random_range_ratio=TEST_PARAMS["random_range_ratio"],
        )

        self.tearDown()
        return metrics

    def test_01_baseline_performance(self):
        print("\n" + "="*80)
        print("stage1：base performance line")
        print("="*80)

        metrics = self._launch_server_and_run_test()
        self.__class__.baseline_tpot = float(metrics["mean_tpot"])

        print(f"\n tpot = {self.__class__.baseline_tpot:.2f} ms")
        time.sleep(20)

    def test_02_mlapo_optimization(self):
        self.assertIsNotNone(
            self.__class__.baseline_tpot,
            "there is no base line "
        )

        print("\n" + "="*80)
        print("🔹 stage2：MLAPO performance")
        print("="*80)

        metrics = self._launch_server_and_run_test(
            extra_envs={"SGLANG_NPU_USE_MLAPO": "1"}
        )
        mlapo_tpot = float(metrics["mean_tpot"])

        tpot_reduction = self.__class__.baseline_tpot - mlapo_tpot
        print("\n" + "="*80)
        print(f"base tpot: {self.__class__.baseline_tpot:.2f} ms")
        print(f"MLAPOtpot: {mlapo_tpot:.2f} ms")
        print(f"Reduce: {tpot_reduction:.2f} ms")
        print(f"required: ≥ 2.00 ms")
        print("="*80)

        self.assertGreater(
            tpot_reduction,
            2.0,
            f"MLAPO didn't match the requirement：tpot reduced {tpot_reduction:.2f}ms，at least 2ms"
        )

        print(f"\n MLAPO match the requirement！tpot reduced{tpot_reduction:.2f} ms")


if __name__ == "__main__":
    unittest.main(verbosity=2)
