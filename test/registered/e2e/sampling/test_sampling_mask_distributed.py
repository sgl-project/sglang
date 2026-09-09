import math
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=180, stage="base-b", runner_config="2-gpu-large")


class TestDistributedSamplingMask(CustomTestCase):
    def _check_parallel_config(self, *, tp_size, pp_size):
        process = None
        try:
            process = popen_launch_server(
                "Qwen/Qwen2.5-0.5B-Instruct",
                DEFAULT_URL_FOR_TEST,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--tp-size",
                    str(tp_size),
                    "--pp-size",
                    str(pp_size),
                    "--sampling-mask-max-tokens",
                    "64",
                    "--mem-fraction-static",
                    "0.5",
                    "--max-running-requests",
                    "8",
                    "--cuda-graph-max-bs-decode",
                    "8",
                ],
            )
            for return_logprob in (False, True):
                with self.subTest(return_logprob=return_logprob):
                    output = self._generate(
                        return_sampling_mask=True, return_logprob=return_logprob
                    )
                    token_ids = output["output_ids"]
                    meta = output["meta_info"]
                    masks = meta["output_token_sampling_mask"]
                    logprobs = meta["output_token_sampling_logprobs"]
                    self.assertEqual(len(token_ids), 4)
                    self.assertEqual(meta["output_token_sampling_mask_length"], 4)
                    self.assertEqual(len(masks), 4)
                    self.assertEqual(len(logprobs), 4)
                    for token_id, mask, logprob in zip(token_ids, masks, logprobs):
                        self.assertIn(token_id, mask)
                        self.assertEqual(len(mask), len(set(mask)))
                        self.assertLessEqual(len(mask), 64)
                        self.assertTrue(math.isfinite(logprob))
                        self.assertLessEqual(logprob, 0.0)
                    if return_logprob:
                        self.assertEqual(len(meta["output_token_logprobs"]), 4)

            ordinary = self._generate(return_sampling_mask=False, return_logprob=False)
            self.assertEqual(len(ordinary["output_ids"]), 4)
            self.assertNotIn("output_token_sampling_mask", ordinary["meta_info"])
        finally:
            if process is not None:
                kill_process_tree(process.pid)
                process.wait(timeout=30)

    def _generate(self, *, return_sampling_mask, return_logprob):
        response = requests.post(
            DEFAULT_URL_FOR_TEST + "/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0.8,
                    "top_k": 8,
                    "top_p": 0.9,
                    "max_new_tokens": 4,
                    "ignore_eos": True,
                },
                "return_sampling_mask": return_sampling_mask,
                "return_logprob": return_logprob,
            },
            timeout=120,
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_tp2_sampling_mask(self):
        """Exercise status synchronization across two tensor-parallel ranks."""
        self._check_parallel_config(tp_size=2, pp_size=1)

    def test_pp2_sampling_mask(self):
        """Exercise mask transport between two live pipeline stages."""
        self._check_parallel_config(tp_size=1, pp_size=2)


if __name__ == "__main__":
    unittest.main()
