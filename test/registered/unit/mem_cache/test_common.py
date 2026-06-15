"""Unit tests for mem_cache/common.py — no server, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")
register_cpu_ci(est_time=7, suite="base-b-test-cpu")

import unittest

import numpy as np

from sglang.srt.mem_cache.common import (
    get_alloc_len_per_decode,
    get_alloc_reserve_per_decode,
    get_req_to_token_extra_context_len,
    kv_to_page_indices,
    kv_to_page_num,
    page_align_floor,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.test_utils import CustomTestCase


class TestKvToPageIndices(CustomTestCase):
    def test_page_size_1_returns_unchanged(self):
        kv = np.array([0, 1, 2, 3, 4, 5])
        result = kv_to_page_indices(kv, 1)
        np.testing.assert_array_equal(result, kv)

    def test_page_size_16(self):
        kv = np.arange(48)
        result = kv_to_page_indices(kv, 16)
        np.testing.assert_array_equal(result, np.array([0, 1, 2]))

    def test_full_pages_only(self):
        kv = np.arange(32)
        result = kv_to_page_indices(kv, 16)
        np.testing.assert_array_equal(result, np.array([0, 1]))

    def test_empty_array(self):
        result = kv_to_page_indices(np.array([], dtype=np.int64), 16)
        self.assertEqual(len(result), 0)

    def test_single_element(self):
        result = kv_to_page_indices(np.array([47]), 16)
        np.testing.assert_array_equal(result, np.array([2]))


class TestKvToPageNum(CustomTestCase):
    def test_exact_multiple(self):
        self.assertEqual(kv_to_page_num(32, 16), 2)

    def test_partial_last_page(self):
        self.assertEqual(kv_to_page_num(33, 16), 3)

    def test_one_token(self):
        self.assertEqual(kv_to_page_num(1, 16), 1)

    def test_zero_tokens(self):
        self.assertEqual(kv_to_page_num(0, 16), 0)

    def test_page_size_1(self):
        self.assertEqual(kv_to_page_num(5, 1), 5)


class TestPageAlignFloor(CustomTestCase):
    def test_already_aligned(self):
        self.assertEqual(page_align_floor(32, 16), 32)

    def test_not_aligned(self):
        self.assertEqual(page_align_floor(33, 16), 32)

    def test_below_page_size(self):
        self.assertEqual(page_align_floor(5, 16), 0)

    def test_zero(self):
        self.assertEqual(page_align_floor(0, 16), 0)

    def test_page_size_1(self):
        self.assertEqual(page_align_floor(5, 1), 5)


class TestGetAllocLenPerDecode(CustomTestCase):
    def _args(self, **kwargs):
        defaults = {
            "model_path": "none",
            "speculative_algorithm": None,
            "speculative_num_steps": None,
            "speculative_eagle_topk": None,
            "speculative_num_draft_tokens": 0,
            "page_size": 1,
        }
        defaults.update(kwargs)
        return ServerArgs(**defaults)

    def test_no_spec_returns_1(self):
        args = self._args()
        self.assertEqual(get_alloc_len_per_decode(args), 1)

    def test_spec_v1_simple(self):
        args = self._args(
            speculative_algorithm="EAGLE",
            speculative_num_steps=5,
            speculative_eagle_topk=3,
            speculative_num_draft_tokens=10,
            page_size=1,
        )
        self.assertEqual(get_alloc_len_per_decode(args), 15)

    def test_spec_v1_draft_tokens_larger(self):
        args = self._args(
            speculative_algorithm="EAGLE",
            speculative_num_steps=3,
            speculative_eagle_topk=2,
            speculative_num_draft_tokens=20,
            page_size=1,
        )
        self.assertEqual(get_alloc_len_per_decode(args), 20)

    def test_spec_v2_tree_mode(self):
        args = self._args(
            speculative_algorithm="EAGLE",
            speculative_num_steps=5,
            speculative_eagle_topk=4,
            speculative_num_draft_tokens=10,
            page_size=16,
        )
        result = get_alloc_len_per_decode(args)
        self.assertGreater(result, 5 * 4)


class TestGetAllocReservePerDecode(CustomTestCase):
    def test_double_of_alloc_len(self):
        args = ServerArgs(
            model_path="none",
            speculative_algorithm="EAGLE",
            speculative_num_steps=5,
            speculative_eagle_topk=3,
            speculative_num_draft_tokens=10,
            page_size=1,
        )
        alloc_len = get_alloc_len_per_decode(args)
        self.assertEqual(get_alloc_reserve_per_decode(args), 2 * alloc_len)

    def test_no_spec_returns_2(self):
        args = ServerArgs(model_path="none")
        self.assertEqual(get_alloc_reserve_per_decode(args), 2)


class TestGetReqToTokenExtraContextLen(CustomTestCase):
    def _args(self, **kwargs):
        defaults = {
            "model_path": "none",
            "speculative_algorithm": None,
            "speculative_eagle_topk": None,
            "speculative_num_draft_tokens": 0,
            "page_size": 1,
        }
        defaults.update(kwargs)
        return ServerArgs(**defaults)

    def test_no_spec_default_extra(self):
        args = self._args()
        self.assertEqual(get_req_to_token_extra_context_len(args), 4)

    def test_with_draft_tokens(self):
        args = self._args(speculative_num_draft_tokens=16)
        self.assertEqual(get_req_to_token_extra_context_len(args), 20)

    def test_spec_v2_tree_increases_extra(self):
        args = self._args(
            speculative_algorithm="EAGLE",
            speculative_eagle_topk=4,
            speculative_num_draft_tokens=10,
            page_size=16,
            speculative_num_steps=5,
        )
        extra = get_req_to_token_extra_context_len(args)
        self.assertGreater(extra, 14)


if __name__ == "__main__":
    unittest.main()
