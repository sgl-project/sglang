import math
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.mock_model.utils import MOCK_MODEL_PATH
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=80, suite="stage-b-test-1-gpu-small-amd")

_MAX_NEW_TOKENS = 4
_TOP_P = 0.99
_TOP_K = 100


class TestSamplingMask(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = MOCK_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=(
                "--load-format",
                "dummy",
                "--mem-fraction-static",
                "0.7",
            ),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_generate_returns_sampling_mask(self):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 1.0,
                    "top_k": _TOP_K,
                    "top_p": _TOP_P,
                    "max_new_tokens": _MAX_NEW_TOKENS,
                    "ignore_eos": True,
                },
                "return_sampling_mask": True,
            },
            timeout=60,
        )
        self.assertEqual(response.status_code, 200, response.text)

        output = response.json()
        meta_info = output["meta_info"]
        output_ids = output["output_ids"]
        sampling_masks = meta_info["output_token_sampling_mask"]
        sampling_logprobs = meta_info["output_token_sampling_logprobs"]

        self.assertEqual(len(output_ids), _MAX_NEW_TOKENS)
        self.assertEqual(meta_info["completion_tokens"], len(output_ids))
        self.assertEqual(meta_info["output_token_sampling_mask_length"], len(output_ids))
        self.assertEqual(len(sampling_masks), len(output_ids))
        self.assertEqual(len(sampling_logprobs), len(output_ids))

        for output_id, sampling_mask, sampling_logprob in zip(
            output_ids, sampling_masks, sampling_logprobs, strict=True
        ):
            self.assertIsInstance(sampling_mask, list)
            self.assertGreater(len(sampling_mask), 0)
            self.assertIn(output_id, sampling_mask)
            self.assertTrue(all(isinstance(token_id, int) for token_id in sampling_mask))
            self.assertIsInstance(sampling_logprob, float)
            self.assertTrue(math.isfinite(sampling_logprob))
            self.assertLessEqual(sampling_logprob, 0.0)


if __name__ == "__main__":
    unittest.main()
