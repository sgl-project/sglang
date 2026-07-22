"""
Unit tests for mem_cache/common.py page helpers - no server, no model loading needed.
"""

import unittest

import numpy as np
import torch

from sglang.srt.mem_cache.common import (
    kv_to_page_indices,
    kv_to_page_num,
    page_align_floor,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestKVToPageIndices(CustomTestCase):
    def test_maps_contiguous_rows_to_page_ids(self):
        kv_indices = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
        page_indices = kv_to_page_indices(kv_indices, page_size=4)
        np.testing.assert_array_equal(page_indices, np.array([0, 1]))

    def test_maps_offset_rows_to_page_ids(self):
        kv_indices = torch.tensor([8, 9, 10, 11, 12, 13, 14, 15])
        page_indices = kv_to_page_indices(kv_indices, page_size=4)
        np.testing.assert_array_equal(page_indices, np.array([2, 3]))

    def test_page_size_one_keeps_row_indices(self):
        kv_indices = torch.tensor([0, 1, 2, 3])
        page_indices = kv_to_page_indices(kv_indices, page_size=1)
        np.testing.assert_array_equal(page_indices, np.array([0, 1, 2, 3]))


class TestKVToPageNum(CustomTestCase):
    def test_ceil_division(self):
        self.assertEqual(kv_to_page_num(17, 4), 5)
        self.assertEqual(kv_to_page_num(16, 4), 4)
        self.assertEqual(kv_to_page_num(1, 4), 1)
        self.assertEqual(kv_to_page_num(0, 4), 0)


class TestPageAlignFloor(CustomTestCase):
    def test_floor_align(self):
        self.assertEqual(page_align_floor(17, 4), 16)
        self.assertEqual(page_align_floor(16, 4), 16)
        self.assertEqual(page_align_floor(15, 4), 12)
        self.assertEqual(page_align_floor(3, 4), 0)
        self.assertEqual(page_align_floor(0, 4), 0)


if __name__ == "__main__":
    unittest.main()
