import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.nightly_utils import NightlyBenchmarkRunner
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=1800, suite="nightly-4-gpu-b200", nightly=True)

DEEPSEEK_V4_FLASH_NVFP4_MODEL_PATH = "nvidia/DeepSeek-V4-Flash-NVFP4"

SERVER_LAUNCH_TIMEOUT = 5400

GSM8K_BASELINE = 0.935

BASE_ARGS = [
    "--trust-remote-code",
    "--tp",
    "4",
    "--moe-runner-backend",
    "flashinfer_trtllm_routed",
    "--chunked-prefill-size",
    "4096",
    "--swa-full-tokens-ratio",
    "0.1",
    "--disable-flashinfer-autotune",
]

MTP_ARGS = [
    "--speculative-algorithm",
    "EAGLE",
    "--speculative-num-steps",
    "3",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "4",
]

SERVER_ARGS = BASE_ARGS + MTP_ARGS


class TestDeepseekV4FlashNvfp4(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            DEEPSEEK_V4_FLASH_NVFP4_MODEL_PATH,
            cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=SERVER_ARGS,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def test_performance(self):
        perf_runner = NightlyBenchmarkRunner(
            profile_dir="performance_profiles_deepseek_v4_flash_nvfp4",
            test_name="DeepSeek-V4-Flash-NVFP4",
            base_url=self.base_url,
        )
        perf_runner.setup_profile_directory()

        profile_path_prefix, json_output_file = perf_runner.generate_profile_filename(
            DEEPSEEK_V4_FLASH_NVFP4_MODEL_PATH, variant="TP4+MTP"
        )
        command = perf_runner.build_benchmark_command(
            DEEPSEEK_V4_FLASH_NVFP4_MODEL_PATH,
            batch_sizes=[1, 8, 16, 64],
            input_lens=(8192,),
            output_lens=(512,),
            profile_path_prefix=profile_path_prefix,
            json_output_file=json_output_file,
            extra_args=["--run-name", "TP4+MTP"],
            server_args=SERVER_ARGS,
        )
        result, success = perf_runner.run_benchmark_command(
            command, "DeepSeek-V4-Flash-NVFP4 (TP4+MTP)"
        )
        self.assertTrue(success, f"Performance benchmark failed:\n{result.stderr}")

        benchmark_results, load_success = perf_runner.load_benchmark_results(
            json_output_file, "DeepSeek-V4-Flash-NVFP4 (TP4+MTP)"
        )
        if load_success and benchmark_results:
            perf_runner.add_report(benchmark_results, variant="TP4+MTP")
        perf_runner.write_final_report()

    def test_accuracy_gsm8k(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=DEEPSEEK_V4_FLASH_NVFP4_MODEL_PATH,
            eval_name="gsm8k",
            num_examples=None,
            num_threads=1024,
        )
        metrics = run_eval(args)
        score = metrics.get("score")
        self.assertIsNotNone(score, "run_eval did not return a score")
        self.assertGreaterEqual(
            score,
            GSM8K_BASELINE,
            f"GSM8K score {score:.3f} below baseline {GSM8K_BASELINE}",
        )


if __name__ == "__main__":
    unittest.main()
