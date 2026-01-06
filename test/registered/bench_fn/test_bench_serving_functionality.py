import asyncio
import unittest
from types import SimpleNamespace

from sglang.bench_serving import DatasetRow, benchmark, run_benchmark
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
import sglang.bench_serving as bench_serving_module
from transformers import AutoTokenizer


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

    def test_multi_turn_functionality(self):
        multi_turn_requests = []
        for i in range(3):
            multi_turn_requests.append(
                DatasetRow(
                    prompt=[
                        f"Hello, this is turn 1 of conversation {i}. Say hi.",
                        f"Turn 2 of conversation {i}. What is 1+1?",
                    ],
                    prompt_len=32,
                    output_len=16,
                )
            )

        mock_args = SimpleNamespace(
            disable_ignore_eos=True,
            disable_stream=False,
            output_file=None,
            output_details=False,
            dataset_name="custom",
            warmup_requests=1,
            plot_throughput=False,
        )
        bench_serving_module.args = mock_args

        tokenizer = AutoTokenizer.from_pretrained(self.model)

        async def run_multi_turn():
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

        total_turns = 3 * 2
        self.assertEqual(res["completed"], total_turns)
        self.assertGreater(res["output_throughput"], 0)


if __name__ == "__main__":
    unittest.main()
