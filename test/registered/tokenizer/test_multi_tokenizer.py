import concurrent.futures
import time
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import MMLUMixin
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    auto_config_device,
    get_benchmark_args,
    is_in_ci,
    popen_launch_server,
    run_benchmark,
    write_github_step_summary,
)

register_cuda_ci(est_time=204, suite="stage-b-test-1-gpu-large")
register_amd_ci(est_time=345, suite="stage-b-test-1-gpu-small-amd")


class TestMultiTokenizer(CustomTestCase, MMLUMixin):
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
                "--mem-fraction-static",
                0.7,
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_multi_tokenizer_ttft(self):
        # from test_bench_serving.py run_bench_serving
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
                f"### test_multi_tokenizer_ttft\n"
                f"median_e2e_latency_ms: {res['median_e2e_latency_ms']:.2f} ms\n"
            )
            self.assertLess(res["median_e2e_latency_ms"], 11000)
            self.assertLess(res["median_ttft_ms"], 86)
            self.assertLess(res["median_itl_ms"], 10)

    def test_pause_continue_generation(self):
        """Test that pause/continue works across all tokenizer workers."""

        def generate(timeout=30):
            return requests.post(
                self.base_url + "/generate",
                json={
                    "text": "The capital of France is",
                    "sampling_params": {"temperature": 0, "max_new_tokens": 8},
                },
                timeout=timeout,
            )

        requests.post(
            self.base_url + "/pause_generation",
            json={"mode": "in_place"},
            timeout=30,
        ).raise_for_status()

        num_requests = 16
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as pool:
            futures = [pool.submit(generate, timeout=60) for _ in range(num_requests)]

            time.sleep(2)

            done = [f for f in futures if f.done()]
            self.assertEqual(
                len(done),
                0,
                f"{len(done)}/{num_requests} requests completed while paused",
            )

            requests.post(
                self.base_url + "/continue_generation", json={}
            ).raise_for_status()

            for f in concurrent.futures.as_completed(futures, timeout=60):
                resp = f.result()
                self.assertEqual(resp.status_code, 200)


if __name__ == "__main__":
    unittest.main()
