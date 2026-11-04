"""Multi-CTA verify sampling: the shapes it takes from eagle_sample's stock softmax (CPU)."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.kernels.ops.speculative import multi_cta_verify_sampling as mcta
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _logits(rows=4, vocab=248320, dtype=torch.float32, is_cuda=True, device_index=0):
    """The attributes applies() reads, for shapes no CPU tensor can stand in for."""
    return SimpleNamespace(
        is_cuda=is_cuda,
        dtype=dtype,
        shape=(rows, vocab),
        dim=lambda: 2,
        device=SimpleNamespace(index=device_index),
    )


class TestApplies(CustomTestCase):
    def test_off_by_default(self):
        with mock.patch.object(mcta, "_cake_capable", lambda index: False):
            self.assertFalse(mcta.applies(_logits()))

    def test_only_the_reproduced_softmax_launch(self):
        """The kernels reproduce torch's one-CTA-per-row fp32 launch for long rows, up to the scratch's 64 rows;
        anything else, and SM103 (where the stock softmax is the Cake kernel), keeps the stock path."""
        cake = {0: False, 1: True}
        with (
            mock.patch.object(mcta, "ENABLED", True),
            mock.patch.object(mcta, "_cake_capable", lambda index: cake[index]),
            mock.patch.object(torch.version, "hip", None),
        ):
            self.assertTrue(mcta.applies(_logits()))
            self.assertTrue(mcta.applies(_logits(rows=64, vocab=65536)))
            for rejected in (
                _logits(rows=0),
                _logits(rows=65),
                _logits(vocab=65532),
                _logits(vocab=100002),
                _logits(dtype=torch.bfloat16),
                _logits(is_cuda=False),
                _logits(device_index=1),
            ):
                self.assertFalse(mcta.applies(rejected), rejected)


if __name__ == "__main__":
    unittest.main()
