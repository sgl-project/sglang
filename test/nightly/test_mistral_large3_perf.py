import os
import unittest
from types import SimpleNamespace

from nightly_utils import NightlyBenchmarkRunner

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    _parse_int_list_env,
    popen_launch_server,
)

register_cuda_ci(est_time=600, suite="nightly-8-gpu-b200", nightly=True)

MISTRAL_LARGE3_MODEL_PATH = "mistralai/Mistral-Large-3-675B-Instruct-2512"
MISTRAL_LARGE3_EAGLE_MODEL_PATH = "mistralai/Mistral-Large-3-675B-Instruct-2512-Eagle"
PROFILE_DIR = "performance_profiles_mistral_large3"


class TestNightlyMistralLarge3Performance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Set environment variable to disable JIT DeepGemm
        os.environ["SGLANG_ENABLE_JIT_DEEPGEMM"] = "0"

        cls.model = MISTRAL_LARGE3_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.batch_sizes = [1, 1, 8, 16, 64]
        cls.input_lens = tuple(_parse_int_list_env("NIGHTLY_INPUT_LENS", "4096"))
        cls.output_lens = tuple(_parse_int_list_env("NIGHTLY_OUTPUT_LENS", "512"))

        # Mistral-Large-3-675B requires TP=8 and trtllm_mla attention backend
        cls.other_args = [
            "--tp",
            "8",
            "--attention-backend",
            "trtllm_mla",
            "--model-loader-extra-config",
            '{"enable_multithread_load": true}',
            "--chat-template",
            "mistral",
        ]

        cls.runner = NightlyBenchmarkRunner(PROFILE_DIR, cls.__name__, cls.base_url)
        cls.runner.setup_profile_directory()

    @classmethod
    def tearDownClass(cls):
        # Clean up environment variable
        if "SGLANG_ENABLE_JIT_DEEPGEMM" in os.environ:
            del os.environ["SGLANG_ENABLE_JIT_DEEPGEMM"]

    def test_bench_one_batch(self):
        results, success = self.runner.run_benchmark_for_model(
            model_path=self.model,
            batch_sizes=self.batch_sizes,
            input_lens=self.input_lens,
            output_lens=self.output_lens,
            other_args=self.other_args,
        )

        self.runner.add_report(results)
        self.runner.write_final_report()

        if not success:
            raise AssertionError(
                f"Benchmark failed for {self.model}. Check the logs for details."
            )

    def test_accuracy_mgsm(self):
        """Run MGSM accuracy evaluation for Mistral Large 3."""
        process = popen_launch_server(
            model=self.model,
            base_url=self.base_url,
            other_args=self.other_args,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        )

        try:
            args = SimpleNamespace(
                base_url=self.base_url,
                model=self.model,
                eval_name="mgsm_en",
                num_examples=None,
                num_threads=1024,
            )
            metrics = run_eval(args)
            print(f"MGSM accuracy for {self.model}: {metrics['score']}")

            # Placeholder threshold - adjust after first successful run
            expected_threshold = 0.90
            self.assertGreaterEqual(
                metrics["score"],
                expected_threshold,
                f"MGSM accuracy {metrics['score']} below threshold {expected_threshold}",
            )
        finally:
            kill_process_tree(process.pid)


class TestNightlyMistralLarge3EaglePerformance(unittest.TestCase):
    """Test Mistral Large 3 with Eagle speculative decoding."""

    @classmethod
    def setUpClass(cls):
        # Set environment variable to disable JIT DeepGemm
        os.environ["SGLANG_ENABLE_JIT_DEEPGEMM"] = "0"

        cls.model = MISTRAL_LARGE3_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.batch_sizes = [1, 1, 8, 16, 64]
        cls.input_lens = tuple(_parse_int_list_env("NIGHTLY_INPUT_LENS", "4096"))
        cls.output_lens = tuple(_parse_int_list_env("NIGHTLY_OUTPUT_LENS", "512"))

        # Mistral-Large-3 with Eagle speculative decoding
        # Eagle model is used as draft model for speculative decoding
        cls.other_args = [
            "--tp",
            "8",
            "--attention-backend",
            "trtllm_mla",
            "--speculative-algorithm",
            "EAGLE",
            "--speculative-draft-model-path",
            MISTRAL_LARGE3_EAGLE_MODEL_PATH,
            "--speculative-num-steps",
            "3",
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            "4",
            "--kv-cache-dtype",
            "auto",
            "--model-loader-extra-config",
            '{"enable_multithread_load": true}',
            "--chat-template",
            "mistral",
        ]

        cls.runner = NightlyBenchmarkRunner(
            "performance_profiles_mistral_large3_eagle", cls.__name__, cls.base_url
        )
        cls.runner.setup_profile_directory()

    @classmethod
    def tearDownClass(cls):
        # Clean up environment variable
        if "SGLANG_ENABLE_JIT_DEEPGEMM" in os.environ:
            del os.environ["SGLANG_ENABLE_JIT_DEEPGEMM"]

    def test_eagle_bench_one_batch(self):
        results, success = self.runner.run_benchmark_for_model(
            model_path=self.model,
            batch_sizes=self.batch_sizes,
            input_lens=self.input_lens,
            output_lens=self.output_lens,
            other_args=self.other_args,
        )

        self.runner.add_report(results)
        self.runner.write_final_report()

        if not success:
            raise AssertionError(
                f"Benchmark failed for {self.model} with Eagle. Check the logs for details."
            )

    def test_eagle_accuracy_mgsm(self):
        """Run MGSM accuracy evaluation for Mistral Large 3 with Eagle."""
        process = popen_launch_server(
            model=self.model,
            base_url=self.base_url,
            other_args=self.other_args,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        )

        try:
            args = SimpleNamespace(
                base_url=self.base_url,
                model=self.model,
                eval_name="mgsm_en",
                num_examples=None,
                num_threads=1024,
            )
            metrics = run_eval(args)
            print(f"MGSM accuracy for {self.model} with Eagle: {metrics['score']}")

            # Placeholder threshold - adjust after first successful run
            expected_threshold = 0.90
            self.assertGreaterEqual(
                metrics["score"],
                expected_threshold,
                f"MGSM accuracy {metrics['score']} below threshold {expected_threshold}",
            )
        finally:
            kill_process_tree(process.pid)


if __name__ == "__main__":
    unittest.main()
