"""Compaction parity for ``get_valid_kv_indices``.

``tl.arange`` needs a power-of-two extent, and the page-table width is one
only at ``index_kpool == 1``. Non-power-of-two widths are the regression.
"""

import unittest

import torch

from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=8, suite="stage-b-test-1-gpu-small-amd-mi35x")


def _run(page_table_1):
    from sglang.kernels.ops.attention.dsa.triton_kernel import get_valid_kv_indices

    bs, topk = page_table_1.shape
    device = page_table_1.device
    counts = (page_table_1 != -1).sum(dim=1)
    kv_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.cumsum(counts, dim=0)
    kv_indices = torch.zeros(bs * topk, dtype=torch.int32, device=device)
    get_valid_kv_indices(
        page_table_1=page_table_1,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        bs=bs,
    )
    return kv_indptr, kv_indices, counts


def _check(test, page_table_1):
    kv_indptr, kv_indices, counts = _run(page_table_1)
    total = int(counts.sum().item())
    for b in range(page_table_1.shape[0]):
        lo, hi = int(kv_indptr[b]), int(kv_indptr[b + 1])
        want = [v for v in page_table_1[b].tolist() if v != -1]
        test.assertEqual(kv_indices[lo:hi].tolist(), want)
    return kv_indices, total


def _table(bs, topk, *, valid_per_row, seed=0, device="cuda:0"):
    gen = torch.Generator().manual_seed(seed)
    vals = torch.randint(0, 100000, (bs, topk), generator=gen, dtype=torch.int32)
    table = torch.full((bs, topk), -1, dtype=torch.int32)
    for b in range(bs):
        n = valid_per_row[b]
        table[b, :n] = vals[b, :n]
    return table.to(device)


class TestGetValidKvIndices(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not is_hip():
            raise unittest.SkipTest("get_valid_kv_indices is the ROCm DSA path")

    def test_non_power_of_two_topk(self):
        """kpool > 1 widths. These used to raise inside tl.arange(0, topk)."""
        for topk in (96, 100, 320, 3072, 2176):
            with self.subTest(topk=topk):
                self.assertNotEqual(topk & (topk - 1), 0)
                valid = [0, 1, topk // 3, topk // 2, topk - 1, topk, 5, topk // 4]
                _check(self, _table(8, topk, valid_per_row=valid, seed=topk))

    def test_power_of_two_topk_still_compacts(self):
        for topk in (64, 256, 2048):
            with self.subTest(topk=topk):
                valid = [0, 1, topk // 3, topk // 2, topk - 1, topk, 5, topk // 4]
                _check(self, _table(8, topk, valid_per_row=valid, seed=topk))

    def test_all_entries_invalid_leaves_the_buffer_untouched(self):
        table = torch.full((4, 320), -1, dtype=torch.int32, device="cuda:0")
        kv_indptr, kv_indices, counts = _run(table)
        self.assertEqual(int(counts.sum().item()), 0)
        self.assertEqual(int(kv_indptr[-1].item()), 0)
        self.assertTrue(kv_indices.eq(0).all())

    def test_padding_overhang_is_never_written(self):
        """2049 pads to 4096; the sentinel lanes must not reach the buffer."""
        topk = 2049
        kv_indices, total = _check(
            self, _table(2, topk, valid_per_row=[7, topk], seed=9)
        )
        self.assertTrue(kv_indices[total:].eq(0).all())

    def test_interleaved_invalid_entries(self):
        topk = 192
        gen = torch.Generator().manual_seed(21)
        table = torch.randint(0, 100000, (6, topk), generator=gen, dtype=torch.int32)
        holes = torch.rand(6, topk, generator=gen) < 0.5
        table[holes] = -1
        _check(self, table.to("cuda:0"))


if __name__ == "__main__":
    unittest.main()
