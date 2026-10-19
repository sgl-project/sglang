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

_SAMPLING_MASK_MAX_TOKENS = 2048
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
            self.assertTrue(
                math.isclose(
                    math.fsum(math.exp(logprob) for logprob in logprobs),
                    1.0,
                    rel_tol=1e-4,
                    abs_tol=1e-4,
                )
            )
            if len(mask) > _TOP_K:
                self.assertTrue(
                    all(
                        logprob == logprobs[_TOP_K - 1] for logprob in logprobs[_TOP_K:]
                    )
                )

    def test_top_p_sampling_mask(self):
        top_k_body = self._generate({"top_k": _TOP_K, "top_p": 1.0})
        body = self._generate({"top_k": _TOP_K, "top_p": 0.8})
        masks = body["meta_info"]["output_token_sampling_mask"]
        logprobs = body["meta_info"]["output_token_sampling_logprobs"]

        self.assertTrue(
            set(masks[0]).issubset(
                top_k_body["meta_info"]["output_token_sampling_mask"][0]
            )
        )
        for token_id, mask, row_logprobs in zip(
            body["output_ids"], masks, logprobs, strict=True
        ):
            self.assertIn(token_id, mask)
            self.assertEqual(len(mask), len(row_logprobs))
            self.assertTrue(
                math.isclose(
                    math.fsum(math.exp(value) for value in row_logprobs),
                    1.0,
                    rel_tol=1e-4,
                    abs_tol=1e-4,
                )
            )

    def test_torch_fallback_sampling_mask(self):
        top_k = 1025  # Above the fused NPU operator's limit.
        body = self._generate({"top_k": top_k, "top_p": 1.0})
        masks = body["meta_info"]["output_token_sampling_mask"]
        logprobs = body["meta_info"]["output_token_sampling_logprobs"]

        for token_id, mask, row_logprobs in zip(
            body["output_ids"], masks, logprobs, strict=True
        ):
            self.assertIn(token_id, mask)
            self.assertEqual(len(mask), len(set(mask)))
            self.assertGreaterEqual(len(mask), 1)
            self.assertLessEqual(len(mask), top_k)
            self.assertEqual(len(mask), len(row_logprobs))
            self.assertTrue(all(math.isfinite(value) for value in row_logprobs))
            self.assertTrue(
                math.isclose(
                    math.fsum(math.exp(value) for value in row_logprobs),
                    1.0,
                    rel_tol=1e-4,
                    abs_tol=1e-4,
                )
            )

    def test_mixed_batch_sampling_mask_modes(self):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": [
                    "The capital of France is",
                    "The capital of Germany is",
                    "The capital of Italy is",
                ],
                "sampling_params": {
                    "temperature": 1.0,
                    "top_k": _TOP_K,
                    "top_p": 1.0,
                    "max_new_tokens": _MAX_NEW_TOKENS,
                    "ignore_eos": True,
                },
                "return_sampling_mask": [True, False, True],
                "sampling_logprobs_mode": ["support", None, "selected"],
            },
            timeout=60,
        )
        self.assertEqual(response.status_code, 200, response.text)
        bodies = response.json()
        self.assertEqual(len(bodies), 3)
        self.assertNotIn("output_token_sampling_mask", bodies[1]["meta_info"])
        self.assertNotIn("output_token_sampling_logprobs", bodies[1]["meta_info"])

        for row, mode in ((0, "support"), (2, "selected")):
            output_ids = bodies[row]["output_ids"]
            masks = bodies[row]["meta_info"]["output_token_sampling_mask"]
            logprobs = bodies[row]["meta_info"]["output_token_sampling_logprobs"]
            self.assertEqual(len(masks), len(output_ids))
            self.assertEqual(len(logprobs), len(output_ids))
            for token_id, mask, row_logprobs in zip(
                output_ids, masks, logprobs, strict=True
            ):
                self.assertIn(token_id, mask)
                if mode == "support":
                    self.assertEqual(len(row_logprobs), len(mask))
                    self.assertTrue(all(math.isfinite(value) for value in row_logprobs))
                else:
                    self.assertIsInstance(row_logprobs, float)
                    self.assertTrue(math.isfinite(row_logprobs))

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
