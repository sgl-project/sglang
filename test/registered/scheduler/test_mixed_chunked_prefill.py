import unittest
from concurrent.futures import ThreadPoolExecutor

import requests

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=184, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=180, suite="stage-b-test-1-gpu-small-amd")


class TestMixedChunkedPrefill(GSM8KMixin, CustomTestCase):
    model = DEFAULT_MODEL_NAME_FOR_TEST
    base_url = DEFAULT_URL_FOR_TEST
    gsm8k_accuracy_thres = 0.62

    extra_args = [
        "--enable-mixed-chunk",
        "--chunked-prefill-size",
        "32",
    ]

    @classmethod
    def setUpClass(cls):
        with envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.override(1):
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=cls.extra_args,
            )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _generate_with_logprobs(self, idx: int) -> dict:
        response = requests.post(
            self.base_url + "/generate",
            json={
                # Prompts longer than the chunk size prefill while earlier
                # requests decode, so they run in mixed batches.
                "text": f"Request {idx}: " + "I have a very good idea on this. " * idx,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                    "ignore_eos": True,
                },
                "return_logprob": True,
                "logprob_start_len": 0,
                "top_logprobs_num": 5,
            },
            timeout=600,
        )
        response.raise_for_status()
        return response.json()

    def test_logprobs(self):
        with ThreadPoolExecutor(max_workers=16) as pool:
            results = list(pool.map(self._generate_with_logprobs, range(32)))
        for res in results:
            meta = res["meta_info"]
            self.assertEqual(len(meta["input_token_logprobs"]), meta["prompt_tokens"])
            self.assertEqual(len(meta["input_top_logprobs"]), meta["prompt_tokens"])
            self.assertEqual(meta["completion_tokens"], 32)
            self.assertEqual(len(meta["output_token_logprobs"]), 32)
            self.assertEqual(len(meta["output_top_logprobs"]), 32)


class TestMixedChunkedPrefillNoRadixCache(TestMixedChunkedPrefill):
    extra_args = [
        "--enable-mixed-chunk",
        "--chunked-prefill-size",
        "32",
        "--disable-radix-cache",
    ]


if __name__ == "__main__":
    unittest.main()
