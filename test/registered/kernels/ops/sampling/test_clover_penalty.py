# SPDX-License-Identifier: Apache-2.0
"""Fused Clover penalties must preserve the original operation rounding."""

import unittest

import torch

from sglang.kernels.ops.sampling.clover_penalty import clover_penalty
from sglang.srt.sampling.penaltylib.repetition_penalty import apply_scaling_penalties
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-small")


class TestCloverPenalty(CustomTestCase):
    def test_exact_committed_history_reference_with_strides(self):
        torch.manual_seed(27)
        for width, vocab, frequency, presence, repetition in (
            (16, 248320, 0.7, -0.3, 0.8),
            (16, 248320, -0.5, 0, 2),
            (16, 248320, 0, 1, 1),
            (3, 129, 0.7, -0.3, 0.8),
            (19, 1025, 0.7, -0.3, 0.8),
        ):
            with self.subTest(
                shape=(width, vocab), params=(frequency, presence, repetition)
            ):
                storage = torch.randn(width, vocab + 7, device="cuda")
                actual = storage[:, :vocab]
                expected = actual.clone()
                guard = storage[:, vocab:].clone()
                counts = torch.randint(0, 4, (vocab,), device="cuda").float()
                original_counts = counts.clone()
                seen = counts > 0
                if frequency:
                    expected.sub_(counts * frequency)
                if presence:
                    expected.sub_(seen.float() * presence)
                if repetition != 1:
                    scale = torch.where(seen, repetition, 1.0)
                    apply_scaling_penalties(expected, scale[None, :])
                params = torch.tensor(
                    [frequency, presence, repetition],
                    device="cuda",
                    dtype=torch.float32,
                )
                clover_penalty(
                    actual,
                    counts,
                    params,
                    frequency=bool(frequency),
                    presence=bool(presence),
                    repetition=repetition != 1,
                )
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(counts, original_counts, rtol=0, atol=0)
                torch.testing.assert_close(storage[:, vocab:], guard, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
