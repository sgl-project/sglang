"""The bounded top-k sort in the torch sampler must match the full vocabulary sort."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from unittest import mock

import torch

from sglang.srt.layers.sampler import top_k_top_p_min_p_sampling_from_probs_torch
from sglang.test.test_utils import CustomTestCase

VOCAB_SIZE = 4096
BATCH_SIZE = 8


def _probs(seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.softmax(
        torch.randn(BATCH_SIZE, VOCAB_SIZE, generator=gen) * 3.0, dim=-1
    )


def _run(probs, top_ks, top_ps, min_ps, need_min_p, max_top_k):
    """Return (token ids, the weights handed to multinomial).

    multinomial is replaced by argmax so both sorts can be compared on the same draw.
    """
    seen = {}

    def _argmax(weights, num_samples):
        seen["weights"] = weights.clone()
        return weights.argmax(dim=-1, keepdim=True)

    with mock.patch("torch.multinomial", side_effect=_argmax):
        token_ids = top_k_top_p_min_p_sampling_from_probs_torch(
            probs.clone(),
            top_ks,
            top_ps,
            min_ps,
            need_min_p,
            None,
            torch.zeros(BATCH_SIZE, dtype=torch.int64),
            max_top_k=max_top_k,
        )
    return token_ids, seen["weights"]


class TestTorchSamplingTopKBound(CustomTestCase):
    def test_bounded_sort_matches_full_sort(self):
        cases = (
            # (top_ks, top_ps, min_ps, need_min_p)
            ([20] * BATCH_SIZE, [1.0] * BATCH_SIZE, [0.0] * BATCH_SIZE, False),
            (
                [1, 2, 5, 20, 50, 100, 7, 64],
                [1.0] * BATCH_SIZE,
                [0.0] * BATCH_SIZE,
                False,
            ),
            ([20] * BATCH_SIZE, [0.95, 0.5, 0.8, 1.0] * 2, [0.0] * BATCH_SIZE, False),
            ([40] * BATCH_SIZE, [0.9] * BATCH_SIZE, [0.0, 0.05] * 4, True),
        )
        for i, (top_ks, top_ps, min_ps, need_min_p) in enumerate(cases):
            with self.subTest(case=i):
                probs = _probs(seed=i)
                top_ks = torch.tensor(top_ks, dtype=torch.int32)
                top_ps = torch.tensor(top_ps, dtype=torch.float32)
                min_ps = torch.tensor(min_ps, dtype=torch.float32)
                max_top_k = int(top_ks.max())

                full_ids, full_weights = _run(
                    probs, top_ks, top_ps, min_ps, need_min_p, max_top_k=None
                )
                bounded_ids, bounded_weights = _run(
                    probs, top_ks, top_ps, min_ps, need_min_p, max_top_k=max_top_k
                )

                self.assertEqual(full_weights.shape[-1], VOCAB_SIZE)
                self.assertEqual(bounded_weights.shape[-1], max_top_k)
                self.assertTrue(torch.all(full_weights[:, max_top_k:] == 0))
                torch.testing.assert_close(
                    bounded_weights, full_weights[:, :max_top_k], rtol=0, atol=0
                )
                torch.testing.assert_close(bounded_ids, full_ids, rtol=0, atol=0)

    def test_return_filtered_probs_keeps_full_sort(self):
        probs = _probs(seed=0)
        top_ks = torch.full((BATCH_SIZE,), 20, dtype=torch.int32)

        _, filtered_probs, token_ids, _ = top_k_top_p_min_p_sampling_from_probs_torch(
            probs.clone(),
            top_ks,
            torch.ones(BATCH_SIZE),
            torch.zeros(BATCH_SIZE),
            False,
            None,
            torch.zeros(BATCH_SIZE, dtype=torch.int64),
            return_filtered_probs=True,
            max_top_k=20,
        )
        self.assertEqual(filtered_probs.shape[-1], VOCAB_SIZE)
        self.assertEqual(token_ids.shape[-1], VOCAB_SIZE)

    def test_bound_at_or_above_vocab_keeps_full_sort(self):
        probs = _probs(seed=0)
        top_ks = torch.full((BATCH_SIZE,), VOCAB_SIZE, dtype=torch.int32)
        _, weights = _run(
            probs,
            top_ks,
            torch.ones(BATCH_SIZE),
            torch.zeros(BATCH_SIZE),
            False,
            max_top_k=VOCAB_SIZE,
        )
        self.assertEqual(weights.shape[-1], VOCAB_SIZE)


if __name__ == "__main__":
    unittest.main()
