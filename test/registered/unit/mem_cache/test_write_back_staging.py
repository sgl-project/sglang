import unittest
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.write_back_staging import WriteBackStaging
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestWriteBackStaging(unittest.TestCase):
    def test_smaller_host_pool_reuses_prepared_storage(self):
        for v_dim in (8, 16):
            with self.subTest(v_dim=v_dim):
                shapes = ((64, 3, 2, 8), (64, 3, 2, v_dim))
                staging = WriteBackStaging.allocate(
                    shapes, dtype=torch.bfloat16, device="cpu"
                )
                expected_bytes = (
                    sum(torch.tensor(shape).prod().item() for shape in shapes) * 2
                )
                self.assertEqual(staging.nbytes, expected_bytes)
                smaller = tuple((7, *shape[1:]) for shape in shapes)
                with patch.object(
                    torch, "empty", side_effect=AssertionError("allocated twice")
                ):
                    views = staging.views(smaller, dtype=torch.bfloat16)
                for buffer, view, shape in zip(staging.buffers, views, smaller):
                    self.assertEqual(buffer.data_ptr(), view.data_ptr())
                    self.assertEqual(tuple(view.shape), shape)
                    view.fill_(3)
                    self.assertTrue(torch.equal(buffer[:7], view))

    def test_geometry_changes_fail_without_allocating(self):
        staging = WriteBackStaging.allocate(
            ((64, 3, 2, 8),), dtype=torch.bfloat16, device="cpu"
        )
        for shapes, dtype in (
            (((65, 3, 2, 8),), torch.bfloat16),
            (((64, 4, 2, 8),), torch.bfloat16),
            (((64, 3, 2, 8),), torch.float32),
            ((), torch.bfloat16),
        ):
            with self.subTest(shapes=shapes, dtype=dtype):
                with self.assertRaisesRegex(ValueError, "changed after preparation"):
                    staging.views(shapes, dtype=dtype)


if __name__ == "__main__":
    unittest.main()
