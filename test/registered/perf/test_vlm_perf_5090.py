"""
VLM Performance tests that work on 5090 (32GB) - VLM offline throughput and online latency tests.
"""

import os
import unittest

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_VLM_MODEL_NAME_FOR_TEST,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    auto_config_device,
    get_benchmark_args,
    is_in_ci,
    run_bench_serving_multi,
    write_github_step_summary,
)

register_cuda_ci(est_time=180, stage="extra-a", runner_config="1-gpu-small")
register_amd_ci(est_time=300, suite="stage-b-test-1-gpu-small-amd")


def _local_tokenizer_path():
    # Prefer the local snapshot so the benchmark client's AutoTokenizer does
    # not call the HF Hub API, which can stall for minutes in CI.
    try:
        from sglang.srt.utils import find_local_repo_dir

        local_dir = find_local_repo_dir(
            DEFAULT_SMALL_VLM_MODEL_NAME_FOR_TEST, revision=None
        )
        if local_dir and os.path.isdir(local_dir):
            return local_dir
    except Exception:
        pass
    return None


class TestVLMPerf5090(CustomTestCase):
    def test_vlm_perf(self):
        common = dict(
            base_url=DEFAULT_URL_FOR_TEST,
            dataset_name="mmmu",
            dataset_path="",
            tokenizer=_local_tokenizer_path(),
            random_input_len=4096,
            random_output_len=2048,
            sharegpt_context_len=None,
            disable_stream=False,
            disable_ignore_eos=False,
            seed=0,
            device=auto_config_device(),
            lora_name=None,
        )
        offline = get_benchmark_args(
            num_prompts=200, request_rate=float("inf"), **common
        )
        # 50 prompts at 1 req/s keeps the online phase ~1 min; medians are
        # stable at this sample size and the thresholds are loose ceilings.
        online = get_benchmark_args(num_prompts=50, request_rate=1, **common)

        (_, res_offline), (_, res_online) = run_bench_serving_multi(
            DEFAULT_SMALL_VLM_MODEL_NAME_FOR_TEST,
            DEFAULT_URL_FOR_TEST,
            other_server_args=["--mem-fraction-static", "0.7"],
            benchmark_args=[offline, online],
        )

        if is_in_ci():
            write_github_step_summary(
                f"### test_vlm_perf (5090)\n"
                f"Output throughput: {res_offline['output_throughput']:.2f} token/s\n"
                f"median_e2e_latency_ms: {res_online['median_e2e_latency_ms']:.2f} ms\n"
            )
            self.assertGreater(res_offline["output_throughput"], 2000)
            self.assertLess(res_online["median_e2e_latency_ms"], 16500)
            self.assertLess(res_online["median_ttft_ms"], 150)
            self.assertLess(res_online["median_itl_ms"], 8)


if __name__ == "__main__":
    unittest.main()
