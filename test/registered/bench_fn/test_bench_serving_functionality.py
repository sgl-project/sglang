import copy
import unittest

from sglang.bench_serving import DatasetRow, run_benchmark
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    get_benchmark_args,
    popen_launch_server,
)

register_cuda_ci(est_time=300, suite="nightly-1-gpu", nightly=True)


class TestBenchServingFunctionality(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--mem-fraction-static", "0.7"],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_basic_smoke(self):
        args = get_benchmark_args(
            base_url=self.base_url,
            dataset_name="random",
            num_prompts=10,
            random_input_len=128,
            random_output_len=32,
            request_rate=float("inf"),
        )
        res = run_benchmark(args)
        self.assertEqual(res["completed"], 10)
        self.assertGreater(res["output_throughput"], 0)

    def test_max_concurrency(self):
        args = get_benchmark_args(
            base_url=self.base_url,
            dataset_name="random",
            num_prompts=20,
            random_input_len=128,
            random_output_len=32,
            request_rate=float("inf"),
        )
        args.max_concurrency = 4
        res = run_benchmark(args)
        self.assertEqual(res["completed"], 20)
        self.assertGreater(res["output_throughput"], 0)

    def test_multi_turn_functionality(self):
        args = get_benchmark_args(
            base_url=self.base_url,
            dataset_name="random",
            num_prompts=5,
            random_input_len=64,
            random_output_len=32,
            request_rate=float("inf"),
            disable_ignore_eos=True,
        )

        multi_turn_requests = []
        for i in range(5):
            multi_turn_requests.append(
                DatasetRow(
                    prompt=[
                        f"Hello, this is turn 1 of conversation {i}. Please respond briefly.",
                        f"This is turn 2 of conversation {i}. What did you say before?",
                        f"This is turn 3 of conversation {i}. Please summarize.",
                    ],
                    prompt_len=64,
                    output_len=32,
                )
            )

        from sglang.bench_serving import benchmark
        import asyncio

        async def run_multi_turn():
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self.model)
            result = await benchmark(
                backend="sglang",
                api_url=f"{self.base_url}/v1/completions",
                base_url=self.base_url,
                model_id=self.model,
                tokenizer=tokenizer,
                input_requests=multi_turn_requests,
                request_rate=float("inf"),
                max_concurrency=2,
                disable_tqdm=True,
                lora_names=None,
                lora_request_distribution=None,
                lora_zipf_alpha=None,
                extra_request_body={},
                profile=False,
            )
            return result

        res = asyncio.run(run_multi_turn())

        total_turns = 5 * 3
        self.assertEqual(res["completed"], total_turns)

        for output_metadata_list in res.get("output_metadata", []):
            if output_metadata_list:
                for metadata in output_metadata_list:
                    if metadata:
                        self.assertIn("multi_round_index", metadata)
                        self.assertIn("multi_round_len", metadata)
                        self.assertEqual(metadata["multi_round_len"], 3)


if __name__ == "__main__":
    unittest.main()

