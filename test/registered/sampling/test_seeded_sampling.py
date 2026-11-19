"""Per-request sampling seeds, in the torch sampler and through the server."""

import unittest

import requests
import torch

from sglang.srt.layers.sampler import top_k_top_p_min_p_sampling_from_probs_torch
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_amd_ci,
    popen_launch_server,
)

register_cuda_ci(est_time=70, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=10, suite="base-b-test-1-gpu-small-amd")

VOCAB_SIZE = 4096
BATCH_SIZE = 8


class TestSeededSampling(CustomTestCase):
    def test_seeded_min_p_matches_renormalized_probs(self):
        """Seeded Gumbel-max sampling is invariant to a row's scale, so min-p's
        unnormalized weights draw what their renormalized distribution draws."""
        gen = torch.Generator(device="cuda").manual_seed(0)
        probs = torch.softmax(
            torch.randn(BATCH_SIZE, VOCAB_SIZE, generator=gen, device="cuda") * 3.0,
            dim=-1,
        )
        min_ps = torch.tensor([0.0, 0.01, 0.05, 0.1] * 2, device="cuda")
        seeds = torch.arange(BATCH_SIZE, dtype=torch.int64, device="cuda") * 7919
        positions = torch.arange(BATCH_SIZE, dtype=torch.int64, device="cuda") + 100
        no_top_k = torch.full(
            (BATCH_SIZE,), VOCAB_SIZE, dtype=torch.int32, device="cuda"
        )
        no_top_p = torch.ones(BATCH_SIZE, device="cuda")

        min_p_ids = top_k_top_p_min_p_sampling_from_probs_torch(
            probs.clone(),
            no_top_k,
            no_top_p,
            min_ps,
            True,
            seeds,
            positions,
        )
        threshold = probs.max(dim=-1, keepdim=True).values * min_ps[:, None]
        kept = probs * (probs >= threshold)
        renormalized_ids = top_k_top_p_min_p_sampling_from_probs_torch(
            kept / kept.sum(dim=-1, keepdim=True),
            no_top_k,
            no_top_p,
            torch.zeros(BATCH_SIZE, device="cuda"),
            False,
            seeds,
            positions,
        )
        torch.testing.assert_close(min_p_ids, renormalized_ids, rtol=0, atol=0)


@unittest.skipIf(
    is_in_amd_ci(), "Decoding is not bit-exact across runs on the AMD runners."
)
class TestSeededGeneration(CustomTestCase):
    """A seed is honored without --enable-deterministic-inference."""

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            # A cached prefix recomputes the last prompt token through another path,
            # which can move its logits enough to flip a near-tie draw.
            other_args=["--disable-radix-cache"],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _generate(self, seed, n=1):
        # One request at a time, so every draw sees the same batch and logits.
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": "Write a short story about a lighthouse.",
                "sampling_params": {
                    "temperature": 1.0,
                    "max_new_tokens": 32,
                    "sampling_seed": seed,
                    "n": n,
                },
            },
        )
        response.raise_for_status()
        body = response.json()
        return body["text"] if n == 1 else [choice["text"] for choice in body]

    def test_seed_repeats_output(self):
        self.assertEqual(self._generate(1234), self._generate(1234))
        self.assertNotEqual(self._generate(1234), self._generate(5678))

    def test_samples_of_a_seeded_request_differ(self):
        first, second = self._generate(1234, n=2)
        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
