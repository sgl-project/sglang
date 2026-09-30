"""Unit tests for the expert distribution recorder buffers -- no server."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

import torch

from sglang.srt.eplb.expert_distribution import _InfiniteBuffer
from sglang.test.test_utils import CustomTestCase


class TestInfiniteBuffer(CustomTestCase):
    def test_growth_inside_inference_mode_accepts_later_appends(self):
        """A buffer grown under inference_mode must still accept appends outside it."""
        buffer = _InfiniteBuffer(item_shape=(2,), dtype=torch.int32, device="cpu")
        with torch.inference_mode():
            for i in range(129):
                buffer.append(torch.full((2,), i, dtype=torch.int32))
        with torch.no_grad():
            buffer.append(torch.full((2,), 129, dtype=torch.int32))
        self.assertEqual(buffer.get_all()[:, 0].tolist(), list(range(130)))


if __name__ == "__main__":
    unittest.main()
