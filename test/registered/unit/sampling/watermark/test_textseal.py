import unittest

import torch

from sglang.srt.sampling.watermark import (
    context_from_token_ids,
    deterministic_key_a_mask,
    prf_dual,
    prf_uniform,
    request_nonce,
    select_textseal_tokens,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestTextSealCore(CustomTestCase):
    def test_pinned_upstream_dual_key_vectors(self):
        contexts = torch.tensor([[1, 2], [2, 1], [0, 7]], dtype=torch.long)
        token_ids = torch.tensor(
            [[0, 1, 17, 8191], [0, 1, 17, 8191], [0, 1, 17, 8191]],
            dtype=torch.long,
        )
        expected_a = torch.tensor(
            [
                [0.9306555986, 0.1497985572, 0.1714076400, 0.1441826373],
                [0.1108533740, 0.1986326426, 0.6988157630, 0.2518618107],
                [0.4537907541, 0.2308631390, 0.2098644823, 0.0665364414],
            ]
        )
        expected_b = torch.tensor(
            [
                [0.3291417360, 0.7209131718, 0.0850933939, 0.9232084155],
                [0.8423879743, 0.9981687069, 0.7058967352, 0.0708094239],
                [0.3026492596, 0.4627029598, 0.1153705269, 0.8742522001],
            ]
        )

        actual_a, actual_b = prf_dual(contexts, token_ids, 741852963, 963852741)
        torch.testing.assert_close(actual_a, expected_a, rtol=0, atol=1e-7)
        torch.testing.assert_close(actual_b, expected_b, rtol=0, atol=1e-7)

    def test_pinned_upstream_signed_overflow_vectors(self):
        cases = [
            ([1], 0, 0, 0.3553900719),
            (
                [9223372036854775807],
                2147483647,
                9223372036854775807,
                0.4679526389,
            ),
            ([-1, 2], -7, -9223372036854775808, 0.9518983960),
        ]
        for context, token_id, key, expected in cases:
            with self.subTest(context=context, token_id=token_id, key=key):
                actual = prf_uniform(
                    torch.tensor([context]), torch.tensor([token_id]), key
                )
                self.assertAlmostEqual(actual.item(), expected, places=7)

    def test_selector_matches_pinned_upstream_expression(self):
        contexts = torch.tensor([[1, 2], [2, 1]], dtype=torch.long)
        probs = torch.tensor([[0.55, 0.25, 0.15, 0.05], [0.6, 0.0, 0.3, 0.1]])
        key_a = torch.tensor([741852963, 741852963])
        key_b = torch.tensor([963852741, 963852741])

        selected_a = select_textseal_tokens(
            probs, contexts, key_a, key_b, torch.tensor([True, True])
        )
        selected_b = select_textseal_tokens(
            probs, contexts, key_a, key_b, torch.tensor([False, False])
        )
        self.assertEqual(selected_a.tolist(), [0, 2])
        self.assertEqual(selected_b.tolist(), [1, 0])

    def test_mixed_ngram_selector_matches_independent_rows(self):
        contexts = torch.tensor([[0, 0, 1, 2], [3, 4, 5, 6]])
        probs = torch.tensor([[0.55, 0.25, 0.15, 0.05], [0.6, 0.0, 0.3, 0.1]])
        key_a = torch.tensor([741852963, 741852963])
        key_b = torch.tensor([963852741, 963852741])
        use_key_a = torch.tensor([True, False])

        mixed = select_textseal_tokens(
            probs,
            contexts,
            key_a,
            key_b,
            use_key_a,
            ngrams=torch.tensor([2, 4]),
        )
        independent = torch.cat(
            [
                select_textseal_tokens(
                    probs[row : row + 1],
                    contexts[row : row + 1, -ngram:],
                    key_a[row : row + 1],
                    key_b[row : row + 1],
                    use_key_a[row : row + 1],
                )
                for row, ngram in enumerate((2, 4))
            ]
        )

        torch.testing.assert_close(mixed, independent)

    def test_context_padding_and_deterministic_key_choice(self):
        self.assertEqual(
            context_from_token_ids([7], 3, device="cpu").tolist(), [0, 0, 7]
        )
        self.assertEqual(
            context_from_token_ids([1, 2, 3, 4], 3, device="cpu").tolist(),
            [2, 3, 4],
        )
        nonces = torch.tensor([request_nonce("a"), request_nonce("b")])
        positions = torch.tensor([11, 11])
        probabilities = torch.tensor([0.5, 0.5])
        first = deterministic_key_a_mask(nonces, positions, probabilities)
        second = deterministic_key_a_mask(nonces, positions, probabilities)
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(
            deterministic_key_a_mask(nonces, positions, torch.zeros(2)).any()
        )
        self.assertTrue(
            deterministic_key_a_mask(nonces, positions, torch.ones(2)).all()
        )


if __name__ == "__main__":
    unittest.main()
