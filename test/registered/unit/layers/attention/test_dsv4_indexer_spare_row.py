import unittest

import torch

from sglang.srt.layers.attention.dsv4.metadata import _with_spare_row
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDSV4IndexerSpareRow(unittest.TestCase):
    def test_lengths_keep_one_row_behind_them_in_storage(self):
        for shape in ((128,), (128, 1), (1,)):
            lens = torch.arange(shape[0], dtype=torch.int32).reshape(shape)
            spare = _with_spare_row(lens)

            self.assertEqual(spare.shape, lens.shape)
            self.assertEqual(spare.dtype, lens.dtype)
            self.assertTrue(torch.equal(spare, lens))
            row_bytes = lens[0].numel() * lens.element_size()
            self.assertEqual(spare.storage_offset(), 0)
            self.assertEqual(
                spare.untyped_storage().nbytes(), (lens.shape[0] + 1) * row_bytes
            )


if __name__ == "__main__":
    unittest.main()
