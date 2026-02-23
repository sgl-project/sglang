import tempfile
import unittest
from pathlib import Path

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="default", nightly=True)


class TestUtils(CustomTestCase):
    def test_calc_rel_diff(self):
        from sglang.srt.debug_utils.comparator.utils import calc_rel_diff

        x = torch.randn(10, 10)
        self.assertAlmostEqual(calc_rel_diff(x, x).item(), 0.0, places=5)
        self.assertAlmostEqual(
            calc_rel_diff(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])).item(),
            1.0,
            places=5,
        )

    def test_argmax_coord(self):
        from sglang.srt.debug_utils.comparator.utils import argmax_coord

        x = torch.zeros(2, 3, 4)
        x[1, 2, 3] = 10.0
        self.assertEqual(argmax_coord(x), (1, 2, 3))

    def test_try_unify_shape(self):
        from sglang.srt.debug_utils.comparator.utils import try_unify_shape

        target = torch.Size([3, 4])
        self.assertEqual(try_unify_shape(torch.randn(1, 1, 3, 4), target).shape, target)
        self.assertEqual(try_unify_shape(torch.randn(2, 3, 4), target).shape, (2, 3, 4))

    def test_compute_smaller_dtype(self):
        from sglang.srt.debug_utils.comparator.utils import compute_smaller_dtype

        self.assertEqual(
            compute_smaller_dtype(torch.float32, torch.bfloat16), torch.bfloat16
        )
        self.assertEqual(
            compute_smaller_dtype(torch.bfloat16, torch.float32), torch.bfloat16
        )
        self.assertIsNone(compute_smaller_dtype(torch.float32, torch.float32))

    def test_load_object(self):
        from sglang.srt.debug_utils.comparator.utils import load_object

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tensor.pt"
            torch.save(torch.randn(5, 5), path)
            self.assertEqual(load_object(path).shape, (5, 5))

            torch.save({"dict": 1}, path)
            self.assertIsNone(load_object(path))

        self.assertIsNone(load_object(Path("/nonexistent.pt")))


if __name__ == "__main__":
    unittest.main()
