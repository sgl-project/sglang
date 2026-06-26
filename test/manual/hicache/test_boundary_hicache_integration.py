import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kits.cache_hit_kit import run_multiturn_cache_hit_test
from sglang.test.kits.eval_accuracy_kit import MMLUMixin
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=450, stage="base-b", runner_config="1-gpu-large")
register_amd_ci(est_time=524, suite="stage-b-test-1-gpu-small-amd")


class TestBoundaryHiCacheIntegration(MMLUMixin, CustomTestCase):
    model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN
    base_url = DEFAULT_URL_FOR_TEST
    mmlu_score_threshold = 0.45
    mmlu_num_examples = 64
    mmlu_num_threads = 32

    @classmethod
    def setUpClass(cls):
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--enable-hierarchical-cache",
                "--radix-cache-backend",
                "boundary_hicache",
                "--mem-fraction-static",
                "0.7",
                "--max-total-tokens",
                "200000",
                "--hicache-ratio",
                "0.25",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_multiturn_cache_hit(self):
        metrics = run_multiturn_cache_hit_test(
            base_url=self.base_url,
            model_path=self.model,
            num_clients=8,
            num_rounds=6,
            request_length=289,
            output_length=367,
        )
        self.assertGreater(metrics["overall"]["cache_hit_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
