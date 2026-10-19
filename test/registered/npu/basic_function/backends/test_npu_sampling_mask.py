import math
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
register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)

_SAMPLING_MASK_MAX_TOKENS = 64
_TOP_K = 10
_MAX_NEW_TOKENS = 4


class TestAscendSamplingMask(CustomTestCase):
    """Verify Ascend sampling-mask output.

    [Test Category] Backends
    [Test Target] --sampling-backend ascend, return_sampling_mask
    """

    model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST

    @classmethod
    def setUpClass(cls):
        other_args = [
            "--sampling-backend",
            "ascend",
            "--sampling-mask-max-tokens",
            str(_SAMPLING_MASK_MAX_TOKENS),
            "--disable-radix-cache",
            "--disable-cuda-graph",
            "--mem-fraction-static",
            "0.85",
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _generate(self, sampling_params, *, sampling_logprobs_mode="support"):
        payload = {
            "text": "The capital of France is",
            "sampling_params": {
                "temperature": 1.0,
                "max_new_tokens": _MAX_NEW_TOKENS,
                "ignore_eos": True,
                **sampling_params,
            },
            "return_sampling_mask": True,
        }
        if sampling_logprobs_mode is not None:
            payload["sampling_logprobs_mode"] = sampling_logprobs_mode
        response = requests.post(self.base_url + "/generate", json=payload, timeout=60)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_top_k_sampling_mask(self):
        body = self._generate({"top_k": _TOP_K, "top_p": 1.0})
        output_ids = body["output_ids"]
        masks = body["meta_info"]["output_token_sampling_mask"]
        sampling_logprobs = body["meta_info"]["output_token_sampling_logprobs"]

        self.assertEqual(len(masks), len(output_ids))
        self.assertEqual(len(sampling_logprobs), len(output_ids))
        for token_id, mask, logprobs in zip(
            output_ids, masks, sampling_logprobs, strict=True
        ):
            self.assertIn(token_id, mask)
            self.assertEqual(len(mask), len(set(mask)))
            # Cutoff ties can make support larger than top_k.
            self.assertLessEqual(len(mask), _SAMPLING_MASK_MAX_TOKENS)
            self.assertEqual(len(mask), len(logprobs))
            self.assertTrue(all(math.isfinite(logprob) for logprob in logprobs))
            if len(mask) > _TOP_K:
                self.assertTrue(
                    all(
                        logprob == logprobs[_TOP_K - 1] for logprob in logprobs[_TOP_K:]
                    )
                )

    def test_min_p_sampling_mask(self):
        body = self._generate({"top_k": _TOP_K, "top_p": 1.0, "min_p": 0.1})
        output_ids = body["output_ids"]
        masks = body["meta_info"]["output_token_sampling_mask"]

        self.assertEqual(len(masks), len(output_ids))
        for token_id, mask in zip(output_ids, masks, strict=True):
            self.assertIn(token_id, mask)
            self.assertGreaterEqual(len(mask), 1)

    def test_greedy_sampling_mask(self):
        body = self._generate({"temperature": 0.0, "top_k": 1})
        output_ids = body["output_ids"]
        masks = body["meta_info"]["output_token_sampling_mask"]

        self.assertEqual(len(masks), len(output_ids))
        for token_id, mask in zip(output_ids, masks, strict=True):
            self.assertEqual(list(mask), [token_id])


if __name__ == "__main__":
    unittest.main()
