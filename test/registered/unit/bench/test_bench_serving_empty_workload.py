import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import Mock, patch

from sglang.benchmark import serving
from sglang.benchmark.datasets.sharegpt import sample_sharegpt_requests
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestBenchServingEmptyWorkload(unittest.IsolatedAsyncioTestCase):
    async def test_empty_workload_reports_the_dataset(self):
        tokenizer = Mock()
        tokenizer.encode.return_value = [1, 2]
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "sharegpt.json"
            dataset.write_text(
                json.dumps([{"conversations": [{"value": "Hi"}, {"value": "Hello"}]}]),
                encoding="utf-8",
            )
            requests = sample_sharegpt_requests(
                str(dataset), num_requests=1, tokenizer=tokenizer, context_len=3
            )
        self.assertEqual(requests, [])

        for dataset_name in ("sharegpt", "mooncake"):
            with (
                self.subTest(dataset=dataset_name),
                patch.object(
                    serving, "args", Namespace(dataset_name=dataset_name), create=True
                ),
                self.assertRaisesRegex(ValueError, f"No requests.*{dataset_name}"),
            ):
                await serving.benchmark(
                    backend="sglang",
                    api_url="",
                    base_url="",
                    model_id="dummy",
                    tokenizer=tokenizer,
                    input_requests=requests,
                    request_rate=float("inf"),
                    max_concurrency=None,
                    disable_tqdm=True,
                    lora_names=[],
                    lora_request_distribution=None,
                    lora_zipf_alpha=None,
                    extra_request_body={},
                    profile=False,
                    warmup_requests=0,
                )


if __name__ == "__main__":
    unittest.main()
