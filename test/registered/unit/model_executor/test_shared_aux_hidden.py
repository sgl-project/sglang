# Shared graph output ownership across shapes, streams and recaptures.
import unittest

import torch

from sglang.srt.model_executor.shared_aux_hidden import SharedAuxHiddenBuffers
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestSharedAuxHidden(CustomTestCase):
    def get(self, pool, rows, stream=None, device="cpu"):
        return pool.get(
            stream,
            rows=rows,
            max_rows=64,
            width=12,
            dtype=torch.bfloat16,
            device=device,
        )

    def test_shapes_share_one_storage_and_preserve_values(self):
        pool = SharedAuxHiddenBuffers()
        big = self.get(pool, 64)
        big.fill_(3)
        for rows in range(1, 65):
            view = self.get(pool, rows)
            self.assertEqual(view.shape, (rows, 12))
            self.assertEqual(view.data_ptr(), big.data_ptr())
            torch.testing.assert_close(view, torch.full_like(view, 3))
        self.assertEqual(len(pool._buffers), 1)

    def test_streams_and_runners_do_not_alias(self):
        pool = SharedAuxHiddenBuffers()
        a, b = self.get(pool, 8, 0), self.get(pool, 8, 1)
        c = self.get(SharedAuxHiddenBuffers(), 8, 0)
        self.assertNotEqual(a.data_ptr(), b.data_ptr())
        self.assertNotEqual(a.data_ptr(), c.data_ptr())
        a.fill_(1)
        b.fill_(2)
        torch.testing.assert_close(a, torch.ones_like(a))

    def test_capacity_and_recapture(self):
        pool = SharedAuxHiddenBuffers()
        with self.assertRaises(ValueError):
            self.get(pool, 65)
        old = self.get(pool, 64)
        new = pool.get(
            None, rows=128, max_rows=128, width=12, dtype=torch.bfloat16, device="cpu"
        )
        self.assertNotEqual(old.data_ptr(), new.data_ptr())
        self.assertEqual(old.shape, (64, 12))


if __name__ == "__main__":
    unittest.main()
