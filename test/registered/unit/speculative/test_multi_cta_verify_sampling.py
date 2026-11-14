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
    """The attributes applies() reads, for shapes no CPU tensor can stand in for."""
    return SimpleNamespace(
        is_cuda=is_cuda, dtype=dtype, shape=(bs, draft, vocab), dim=lambda: 3
    )


class TestApplies(CustomTestCase):
    def test_off_without_the_runtime_master(self):
        with mock.patch.object(mcta, "ENABLED", False):
            self.assertFalse(mcta.applies(_probs()))

    def test_only_the_shapes_checked_bitwise(self):
        """Up to the scratch's 64 rows (16 requests at 4 draft tokens) of long fp32 rows on CUDA."""
        with (
            mock.patch.object(mcta, "ENABLED", True),
            mock.patch.object(torch.version, "hip", None),
        ):
            self.assertTrue(mcta.applies(_probs()))
            self.assertTrue(mcta.applies(_probs(bs=16, vocab=65536)))
            for rejected in (
                _probs(bs=0),
                _probs(bs=17),
                _probs(vocab=65532),
                _probs(dtype=torch.bfloat16),
                _probs(is_cuda=False),
            ):
                self.assertFalse(mcta.applies(rejected), rejected)
        with (
            mock.patch.object(mcta, "ENABLED", True),
            mock.patch.object(torch.version, "hip", "6.4"),
        ):
            self.assertFalse(mcta.applies(_probs()))

    def test_scratch_bounds_raise(self):
        probs = torch.zeros(65, 1, 8)
        with self.assertRaisesRegex(ValueError, "past the scratch"):
            mcta.tree_speculative_sampling_target_only(*([None] * 9), probs, probs)


if __name__ == "__main__":
    unittest.main()
