import unittest
from types import SimpleNamespace

import sglang.srt.managers.io_struct as io_struct
from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
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


class TestMultiTokenizer(CustomTestCase):
    # from test_hicache.py
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

    def test_mmlu(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=64,
            num_threads=32,
        )
        metrics = run_eval(args)
        self.assertGreaterEqual(metrics["score"], 0.65)

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


if __name__ == "__main__":
    unittest.main()
