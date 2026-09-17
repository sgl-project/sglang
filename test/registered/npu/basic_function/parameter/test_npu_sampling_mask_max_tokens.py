import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="base-b-test-1-npu-a3")

_MAX_NEW_TOKENS = 4
_TOP_K = 8
_TOP_P = 0.9
_SAMPLING_MASK_MAX_TOKENS = 64


class TestNpuSamplingMaskMaxTokens(CustomTestCase):
    """Testcase: Verify `--sampling-mask-max-tokens` bounds the returned sampling
    mask and rejects requests whose top_k exceeds the cap on the NPU backend.

    [Test Category] Parameter
    [Test Target] --sampling-mask-max-tokens
    """

    @classmethod
    def setUpClass(cls):
        cls.model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--disable-radix-cache",
                "--sampling-mask-max-tokens",
                str(_SAMPLING_MASK_MAX_TOKENS),
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _generate(self, sampling_params):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 1.0,
                    "max_new_tokens": _MAX_NEW_TOKENS,
                    "ignore_eos": True,
                    **sampling_params,
                },
                "return_sampling_mask": True,
            },
            timeout=60,
        )
        return response

    def test_greedy_returns_singleton_mask(self):
        response = self._generate({"temperature": 0.0})
        self.assertEqual(response.status_code, 200, response.text)

        output = response.json()
        output_ids = output["output_ids"]
        meta = output["meta_info"]
        masks = meta["output_token_sampling_mask"]

        self.assertEqual(len(output_ids), _MAX_NEW_TOKENS)
        self.assertEqual(len(masks), _MAX_NEW_TOKENS)
        self.assertTrue(all(len(mask) == 1 for mask in masks))
        for token_id, mask in zip(output_ids, masks):
            self.assertIn(token_id, mask)

    def test_non_greedy_returns_bounded_mask(self):
        response = self._generate({"temperature": 1.0, "top_k": _TOP_K, "top_p": _TOP_P})
        self.assertEqual(response.status_code, 200, response.text)

        output = response.json()
        output_ids = output["output_ids"]
        meta = output["meta_info"]
        masks = meta["output_token_sampling_mask"]

        self.assertEqual(len(output_ids), _MAX_NEW_TOKENS)
        self.assertEqual(meta["output_token_sampling_mask_length"], _MAX_NEW_TOKENS)
        self.assertEqual(len(masks), _MAX_NEW_TOKENS)
        for token_id, mask in zip(output_ids, masks):
            self.assertIn(token_id, mask)
            self.assertEqual(len(mask), len(set(mask)))
            self.assertLessEqual(len(mask), _SAMPLING_MASK_MAX_TOKENS)

    def test_rejects_top_k_exceeding_cap(self):
        response = self._generate({"top_k": _SAMPLING_MASK_MAX_TOKENS + 1})
        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn(
            "return_sampling_mask requires top_k=1 for greedy sampling",
            response.text,
        )


if __name__ == "__main__":
    unittest.main()