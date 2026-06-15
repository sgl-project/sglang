"""Pointer arithmetic tests for CommonKVManager MHA PP transfers."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_MAIN_K, _MAIN_V, _DRAFT_K, _DRAFT_V = 1000, 2000, 3000, 4000


def _dst_ptrs(num_main, num_draft=0):
    return (
        [_MAIN_K + i for i in range(num_main)]
        + [_MAIN_V + i for i in range(num_main)]
        + [_DRAFT_K + i for i in range(num_draft)]
        + [_DRAFT_V + i for i in range(num_draft)]
    )


def _src_ptrs(start, num):
    return [_MAIN_K + i for i in range(start, start + num)] + [
        _MAIN_V + i for i in range(start, start + num)
    ]


class TestGetMhaKvPtrsWithPp(CustomTestCase):
    def _mgr(self, start_layer):
        mgr = MagicMock(spec=CommonKVManager)
        mgr.get_mha_kv_ptrs_with_pp = CommonKVManager.get_mha_kv_ptrs_with_pp.__get__(
            mgr, CommonKVManager
        )
        mgr.kv_args = SimpleNamespace(prefill_start_layer=start_layer)
        return mgr

    def test_uneven_pp_no_draft_both_stages(self):
        dst = _dst_ptrs(num_main=15)
        _, _, dst_k0, dst_v0, n0 = self._mgr(0).get_mha_kv_ptrs_with_pp(
            _src_ptrs(0, 7), dst
        )
        _, _, dst_k1, dst_v1, n1 = self._mgr(7).get_mha_kv_ptrs_with_pp(
            _src_ptrs(7, 8), dst
        )
        self.assertEqual((n0, n1), (7, 8))
        self.assertEqual(dst_k0, [_MAIN_K + i for i in range(0, 7)])
        self.assertEqual(dst_v0, [_MAIN_V + i for i in range(0, 7)])
        self.assertEqual(dst_k1, [_MAIN_K + i for i in range(7, 15)])
        self.assertEqual(dst_v1, [_MAIN_V + i for i in range(7, 15)])

    def test_equal_layers_no_mtp(self):
        dst = _dst_ptrs(num_main=4)
        _, _, dst_k, dst_v, n = self._mgr(0).get_mha_kv_ptrs_with_pp(
            _src_ptrs(0, 4), dst, num_dst_target_kv_layers=4
        )
        self.assertEqual(n, 4)
        self.assertEqual(dst_k, [_MAIN_K + i for i in range(4)])
        self.assertEqual(dst_v, [_MAIN_V + i for i in range(4)])

    def test_draft_layout_uses_explicit_target_count(self):
        dst = _dst_ptrs(num_main=15, num_draft=1)
        _, _, dst_k, dst_v, n = self._mgr(11).get_mha_kv_ptrs_with_pp(
            _src_ptrs(11, 4), dst, num_dst_target_kv_layers=15
        )
        self.assertEqual(n, 4)
        self.assertEqual(dst_k, [_MAIN_K + i for i in range(11, 15)])
        self.assertEqual(dst_v, [_MAIN_V + i for i in range(11, 15)])
        self.assertNotIn(_DRAFT_K, dst_v)

    def test_missing_none_and_minus_one_fall_back_to_total_count(self):
        dst = _dst_ptrs(num_main=15)
        src = _src_ptrs(7, 8)
        missing = self._mgr(7).get_mha_kv_ptrs_with_pp(src, dst)
        none = self._mgr(7).get_mha_kv_ptrs_with_pp(
            src, dst, num_dst_target_kv_layers=None
        )
        minus_one = self._mgr(7).get_mha_kv_ptrs_with_pp(
            src, dst, num_dst_target_kv_layers=-1
        )
        self.assertEqual(missing, none)
        self.assertEqual(none, minus_one)
        self.assertEqual(minus_one[2], [_MAIN_K + i for i in range(7, 15)])
        self.assertEqual(minus_one[3], [_MAIN_V + i for i in range(7, 15)])


if __name__ == "__main__":
    unittest.main()
