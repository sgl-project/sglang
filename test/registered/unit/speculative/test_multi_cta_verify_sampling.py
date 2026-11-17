"""Multi-CTA verify tree sampling: the shapes it takes from the stock tree sampler (CPU)."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.kernels.ops.speculative import multi_cta_verify_sampling as mcta
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _probs(bs=1, draft=4, vocab=248320, dtype=torch.float32, is_cuda=True):
    return SimpleNamespace(
        is_cuda=is_cuda, dtype=dtype, shape=(bs, draft, vocab), dim=lambda: 3
    )


class TestApplies(CustomTestCase):
    def test_only_the_shapes_that_gain(self):
        # up to the scratch's 1024 rows of fp32 rows of 128000+ entries on CUDA
        with mock.patch.object(torch.version, "hip", None):
            self.assertTrue(mcta.applies(_probs()))
            self.assertTrue(mcta.applies(_probs(bs=256, vocab=128000)))
            for rejected in (
                _probs(bs=0),
                _probs(bs=257),
                _probs(vocab=127999),
                _probs(dtype=torch.bfloat16),
                _probs(is_cuda=False),
            ):
                self.assertFalse(mcta.applies(rejected), rejected)
        with mock.patch.object(torch.version, "hip", "6.4"):
            self.assertFalse(mcta.applies(_probs()))

    def test_scratch_bounds_raise(self):
        probs = torch.zeros(1025, 1, 8)
        with self.assertRaisesRegex(ValueError, "past the scratch"):
            mcta.tree_speculative_sampling_target_only(*([None] * 9), probs, probs)


if __name__ == "__main__":
    unittest.main()
