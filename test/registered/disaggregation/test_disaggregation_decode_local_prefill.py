import math
import random
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor

import requests

from sglang.benchmark.datasets.random import sample_random_requests
from sglang.benchmark.utils import get_tokenizer
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
    assert_process_healthy,
)
from sglang.test.test_utils import DEFAULT_MODEL_NAME_FOR_TEST, try_cached_model

register_cuda_ci(est_time=150, stage="base-b", runner_config="2-gpu-large")

CHUNK_SIZE = 128


class TestDisaggregationDecodeLocalPrefill(PDDisaggregationServerBase):
    extra_decode_args = [
        "--disaggregation-decode-enable-radix-cache",
        "--disaggregation-decode-enable-local-prefill",
        "--chunked-prefill-size",
        str(CHUNK_SIZE),
    ]

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = try_cached_model(DEFAULT_MODEL_NAME_FOR_TEST)
        cls.launch_all()
        cls.tokenizer = get_tokenizer(cls.model)

    def _token_ids(self, input_len, num_prompts):
        return [
            list(request.prompt)
            for request in sample_random_requests(
                input_len=input_len,
                output_len=1,
                num_prompts=num_prompts,
                range_ratio=1.0,
                tokenizer=self.tokenizer,
                dataset_path="",
                return_text=False,
            )
        ]

    def _generate(self, url, input_ids, max_new_tokens, **fields):
        body = {
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": max_new_tokens,
                "ignore_eos": True,
            },
            "return_logprob": True,
            "top_logprobs_num": 5,
            **fields,
        }
        response = requests.post(f"{url}/generate", json=body, timeout=300)
        response.raise_for_status()
        return response.json()

    def _pd(self, input_ids, max_new_tokens, **fields):
        return self._generate(self.lb_url, input_ids, max_new_tokens, **fields)

    def _local(self, input_ids, max_new_tokens):
        return self._generate(
            self.decode_url, input_ids, max_new_tokens, disagg_local_prefill=True
        )

    @staticmethod
    def _first_token_kl(ref, got):
        p = {t[1]: t[0] for t in ref["meta_info"]["output_top_logprobs"][0]}
        q = {t[1]: t[0] for t in got["meta_info"]["output_top_logprobs"][0]}
        return sum(math.exp(lp) * (lp - q.get(t, -30.0)) for t, lp in p.items())

    def test_local_prefill_reuses_the_decode_radix_cache(self):
        history_len, suffix_len, output_len = 1024, 3 * CHUNK_SIZE + 17, 32
        histories = self._token_ids(history_len, 4)
        suffixes = self._token_ids(suffix_len, 4)
        for history, suffix in zip(histories, suffixes):
            turn1 = self._pd(history, output_len)
            prompt = history + turn1["output_ids"] + suffix

            local = self._local(prompt, output_len)
            reference = self._pd(prompt, output_len, cache_salt=uuid.uuid4().hex)

            # The decode server inserted turn 1's prompt and output into its tree.
            self.assertGreaterEqual(local["meta_info"]["cached_tokens"], history_len)
            self.assertEqual(len(local["output_ids"]), output_len)
            self.assertLess(self._first_token_kl(reference, local), 0.05)

    def test_local_and_pd_requests_share_the_decode_server(self):
        rnd = random.Random(0)
        histories = self._token_ids(512, 8)
        for history in histories:
            self._pd(history, 8)
        suffixes = self._token_ids(4 * CHUNK_SIZE, 32)

        def run(i):
            prompt = histories[i % len(histories)] + suffixes[i]
            max_new_tokens = rnd.choice([1, 16, 64])
            if i % 2:
                return self._local(prompt, max_new_tokens)
            return self._pd(prompt, max_new_tokens)

        with ThreadPoolExecutor(16) as pool:
            outputs = list(pool.map(run, range(len(suffixes))))

        for output in outputs:
            self.assertEqual(output["meta_info"]["finish_reason"]["type"], "length")
        assert_process_healthy(self, "prefill", self.process_prefill, self.prefill_url)
        assert_process_healthy(self, "decode", self.process_decode, self.decode_url)

    def test_prefill_server_rejects_local_prefill(self):
        response = requests.post(
            f"{self.prefill_url}/generate",
            json={
                "input_ids": [1, 2, 3],
                "sampling_params": {"max_new_tokens": 1},
                "disagg_local_prefill": True,
            },
            timeout=60,
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("--disaggregation-decode-enable-local-prefill", response.text)


if __name__ == "__main__":
    unittest.main()
