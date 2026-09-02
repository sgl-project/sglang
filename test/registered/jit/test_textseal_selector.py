from __future__ import annotations

import unittest

import torch

from sglang.kernels.ops.sampling.textseal_selector import (
    select_textseal_tokens_triton,
)
from sglang.srt.sampling.watermark.textseal import (
    _weighted_sum_by_ngram,
    select_textseal_tokens,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class TestTextSealSelector(CustomTestCase):
    def test_matches_reference_across_split_boundaries(self) -> None:
        """Blockwise selection must preserve dense hash-score argmax semantics."""
        generator = torch.Generator().manual_seed(7)
        batch_size = 5
        vocab_size = 16_397
        probs = torch.rand(batch_size, vocab_size, generator=generator)
        probs, token_ids = probs.sort(dim=-1, descending=True)
        support_lengths = torch.tensor([1, 8_191, 8_192, 8_193, vocab_size])
        is_candidate = torch.arange(vocab_size).view(1, -1) < support_lengths.view(
            -1, 1
        )
        probs = torch.where(is_candidate, probs, 0.0)
        probs = probs / probs.sum(dim=-1, keepdim=True)

        contexts = torch.tensor(
            [
                [0, 0, 0, 0],
                [1, -2, 3, -4],
                [2**31 - 1, -(2**31), 17, -29],
                [10**12, -(10**12), 97, -101],
                [-(2**40), 2**39, -(2**38), 2**37],
            ],
            dtype=torch.int64,
        )
        ngrams = torch.tensor([1, 2, 3, 4, 4], dtype=torch.int32)
        key_a = torch.tensor([0, 741852963, -(2**40), 2**39, -17])
        key_b = torch.tensor([963852741, -31, 2**40, -(2**39), 19])
        use_key_a = torch.tensor([True, False, True, False, True])

        expected = select_textseal_tokens(
            probs,
            contexts,
            key_a,
            key_b,
            use_key_a,
            token_ids=token_ids,
            ngrams=ngrams,
        )
        weighted_contexts = _weighted_sum_by_ngram(contexts, ngrams)
        keys = torch.where(use_key_a, key_a, key_b)
        actual = select_textseal_tokens_triton(
            probs.cuda(),
            token_ids.cuda(),
            weighted_contexts.cuda(),
            keys.cuda(),
        ).cpu()

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
