import unittest
from urllib.parse import urlparse

import requests
import torch

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import MMLUMixin
from sglang.test.send_one import BenchArgs, send_one_prompt
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    auto_config_device,
    get_benchmark_args,
    is_in_amd_ci,
    is_in_ci,
    popen_launch_server,
    run_benchmark,
    write_github_step_summary,
)

register_cuda_ci(est_time=211, suite="base-b-test-1-gpu-large")
register_amd_ci(est_time=345, suite="stage-b-test-1-gpu-small-amd")

# dp_rank detokenizer router integration: one router per DP rank (requires dp_size > 2).
DP_SIZE_FOR_DP_RANK_ROUTER_TEST = 3
_MIN_GPUS_FOR_DP_RANK_ROUTER_TEST = DP_SIZE_FOR_DP_RANK_ROUTER_TEST

register_cuda_ci(
    est_time=420,
    stage="extra-b",
    runner_config="4-gpu-h100",
)
register_amd_ci(est_time=600, suite="stage-c-test-4-gpu-amd")


class TestMultiDetokenizer(CustomTestCase, MMLUMixin):
    mmlu_score_threshold = 0.65
    mmlu_num_examples = 64
    mmlu_num_threads = 32

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tokenizer-worker-num",
                8,
                "--detokenizer-worker-num",
                4,
                # Explicit default: one router fans out to all detokenizer workers.
                "--detokenizer-router-sharding",
                "single",
                "--mem-fraction-static",
                0.7,
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_multi_detokenizer_ttft(self):
        args = get_benchmark_args(
            base_url=self.base_url,
            dataset_name="random",
            dataset_path="",
            tokenizer=None,
            num_prompts=100,
            random_input_len=4096,
            random_output_len=2048,
            sharegpt_context_len=None,
            request_rate=1,
            disable_stream=False,
            disable_ignore_eos=False,
            seed=0,
            device=auto_config_device(),
            lora_name=None,
        )
        res = run_benchmark(args)
        if is_in_ci():
            write_github_step_summary(
                f"### test_multi_detokenizer_ttft\n"
                f"median_e2e_latency_ms: {res['median_e2e_latency_ms']:.2f} ms\n"
            )
            self.assertLess(res["median_e2e_latency_ms"], 11000)
            self.assertLess(res["median_ttft_ms"], 130 if is_in_amd_ci() else 86)
            self.assertLess(res["median_itl_ms"], 10)


@unittest.skipUnless(
    torch.cuda.is_available()
    and torch.cuda.device_count() >= _MIN_GPUS_FOR_DP_RANK_ROUTER_TEST,
    f"needs >= {_MIN_GPUS_FOR_DP_RANK_ROUTER_TEST} GPUs for dp_size="
    f"{DP_SIZE_FOR_DP_RANK_ROUTER_TEST} dp_rank router test",
)
class TestMultiDetokenizerDpRankSharding(CustomTestCase):
    """End-to-end: multi detokenizer workers with per-DP-rank routers (dp_size > 2)."""

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--dp-size",
                str(DP_SIZE_FOR_DP_RANK_ROUTER_TEST),
                "--tokenizer-worker-num",
                "4",
                "--detokenizer-worker-num",
                "4",
                "--detokenizer-router-sharding",
                "dp_rank",
                "--mem-fraction-static",
                "0.6",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_server_reports_expected_dp_size(self):
        info = requests.get(self.base_url + "/server_info", timeout=30).json()
        self.assertEqual(info["dp_size"], DP_SIZE_FOR_DP_RANK_ROUTER_TEST)
        self.assertEqual(info["detokenizer_worker_num"], 4)
        self.assertEqual(info["detokenizer_router_sharding"], "dp_rank")

    def test_dp_rank_router_generation(self):
        parsed = urlparse(self.base_url)
        port = parsed.port or 30000
        args = BenchArgs(
            host=parsed.hostname or "127.0.0.1",
            port=port,
            max_new_tokens=32,
            temperature=0.0,
        )
        _, speed = send_one_prompt(args)
        self.assertGreater(speed, 0)

    def test_dp_rank_router_ttft(self):
        args = get_benchmark_args(
            base_url=self.base_url,
            dataset_name="random",
            dataset_path="",
            tokenizer=None,
            num_prompts=32,
            random_input_len=512,
            random_output_len=128,
            sharegpt_context_len=None,
            request_rate=2,
            disable_stream=False,
            disable_ignore_eos=False,
            seed=0,
            device=auto_config_device(),
            lora_name=None,
        )
        res = run_benchmark(args)
        if is_in_ci():
            write_github_step_summary(
                f"### test_dp_rank_router_ttft (dp_size={DP_SIZE_FOR_DP_RANK_ROUTER_TEST})\n"
                f"median_e2e_latency_ms: {res['median_e2e_latency_ms']:.2f} ms\n"
                f"median_ttft_ms: {res['median_ttft_ms']:.2f} ms\n"
            )
            self.assertLess(res["median_e2e_latency_ms"], 15000)
            self.assertLess(res["median_ttft_ms"], 200 if is_in_amd_ci() else 150)


if __name__ == "__main__":
    unittest.main()
