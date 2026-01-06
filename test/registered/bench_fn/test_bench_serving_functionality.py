import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase, run_bench_serving

register_cuda_ci(est_time=300, suite="nightly-1-gpu", nightly=True)

MODEL = "Qwen/Qwen3-0.6B"


class TestBenchServingFunctionality(CustomTestCase):
    def test_gsp_basic(self):
        res = run_bench_serving(
            model=MODEL,
            num_prompts=16,
            request_rate=float("inf"),
            other_server_args=["--mem-fraction-static", "0.7"],
            dataset_name="generated-shared-prefix",
            gsp_num_groups=4,
            gsp_prompts_per_group=4,
            gsp_system_prompt_len=128,
            gsp_question_len=32,
            gsp_output_len=32,
        )
        self.assertGreater(res["output_throughput"], 0)

    def test_gsp_multi_turn(self):
        res = run_bench_serving(
            model=MODEL,
            num_prompts=8,
            request_rate=float("inf"),
            other_server_args=["--mem-fraction-static", "0.7"],
            dataset_name="generated-shared-prefix",
            disable_ignore_eos=True,
            gsp_num_groups=2,
            gsp_prompts_per_group=4,
            gsp_system_prompt_len=64,
            gsp_question_len=16,
            gsp_output_len=16,
            gsp_num_turns=3,
        )
        self.assertEqual(res["completed"], 8 * 3)
        self.assertGreater(res["output_throughput"], 0)


if __name__ == "__main__":
    unittest.main()
