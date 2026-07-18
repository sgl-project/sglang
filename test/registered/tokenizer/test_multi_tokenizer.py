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
    is_in_amd_ci,
    is_in_ci,
    popen_launch_server,
    run_benchmark,
    write_github_step_summary,
)

register_cuda_ci(est_time=220, stage="base-b", runner_config="1-gpu-large")
register_amd_ci(est_time=355, suite="stage-b-test-1-gpu-small-amd")

# --- KV_SIZE_THRES begin (auto; update_memory_thresholds.py) ---
# gpu=h100 updated=2026-07-18
KV_SIZE_THRES = 40428.7
# --- KV_SIZE_THRES end ---


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
            # relax for mi300x
            self.assertLess(res["median_ttft_ms"], 130 if is_in_amd_ci() else 86)
            self.assertLess(res["median_itl_ms"], 10)

    def test_batch_input_ids_routing(self):
        # Regression guard for sgl-project/sglang#29878 (introduced by #29214).
        #
        # A batch of pre-tokenized `input_ids` (no text / multimodal) is the one
        # case that takes the batch-tokenization path (_send_batch_request ->
        # BatchTokenizedGenerateReqInput). In multi-tokenizer mode this batch
        # must stamp each sub-request's `http_worker_ipc` so the scheduler can
        # route every reply back to its owning tokenizer worker. If it is missing,
        # the requests hang forever.
        #
        # The existing ttft test only sends *text*, so it never exercises this
        # path — this case does, and uses a short timeout so a routing hang
        # fails fast instead of stalling until the server launch timeout.
        batch_input_ids = [
            [1, 2, 3, 4, 5],
            [10, 11, 12, 13, 14],
            [20, 21, 22, 23, 24],
            [30, 31, 32, 33, 34],
        ]
        response = requests.post(
            self.base_url + "/generate",
            json={
                "input_ids": batch_input_ids,
                "sampling_params": {"max_new_tokens": 8, "temperature": 0},
            },
            timeout=60,
        )
        self.assertEqual(response.status_code, 200, response.text)
        results = response.json()
        # Every batched request must get its reply routed back — not hang.
        self.assertEqual(len(results), len(batch_input_ids))
        for result in results:
            self.assertIn("text", result)


if __name__ == "__main__":
    unittest.main()
