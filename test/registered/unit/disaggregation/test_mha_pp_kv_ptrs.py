"""Unit tests for srt/disaggregation/common/conn — get_mha_kv_ptrs_with_pp.

Regression coverage for issue #27740: an uneven PP split of a plain
full-attention model was mis-detected as a draft/MTP-augmented decode layout,
which mis-sliced the destination V pointers and silently corrupted the
transferred KV cache.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.test.test_utils import CustomTestCase

# Pointer encoding for readability: main K layer i -> 1000+i, main V layer i ->
# 2000+i, draft K layer i -> 3000+i, draft V layer i -> 4000+i.
_MAIN_K, _MAIN_V, _DRAFT_K, _DRAFT_V = 1000, 2000, 3000, 4000


def _dst_ptrs(num_main, num_draft=0):
    """Decode-side flat ptr list: [K_main, V_main, draft_K, draft_V]."""
    k_main = [_MAIN_K + i for i in range(num_main)]
    v_main = [_MAIN_V + i for i in range(num_main)]
    dk = [_DRAFT_K + i for i in range(num_draft)]
    dv = [_DRAFT_V + i for i in range(num_draft)]
    return k_main + v_main + dk + dv


def _src_ptrs(start, num):
    """Prefill-side flat ptr list for one PP stage: [K, V]."""
    return [_MAIN_K + i for i in range(start, start + num)] + [
        _MAIN_V + i for i in range(start, start + num)
    ]


class TestGetMhaKvPtrsWithPp(CustomTestCase):
    def _mgr(self, start_layer, num_total_layers="unset"):
        from sglang.srt.disaggregation.common.conn import CommonKVManager

        mgr = MagicMock(spec=CommonKVManager)
        mgr.get_mha_kv_ptrs_with_pp = CommonKVManager.get_mha_kv_ptrs_with_pp.__get__(
            mgr, CommonKVManager
        )
        if num_total_layers == "unset":
            mgr.kv_args = SimpleNamespace(prefill_start_layer=start_layer)
        else:
            mgr.kv_args = SimpleNamespace(
                prefill_start_layer=start_layer,
                prefill_num_total_layers=num_total_layers,
            )
        return mgr

    def test_uneven_pp_no_draft(self):
        """15 full-attn layers served as PP0=7 + PP1=8, decode PP=1.

        This is the #27740 case: dst_num_total_layers (15) % num_kv_layers
        (7 or 8) != 0, which used to take the draft branch and corrupt V.
        """
        dst = _dst_ptrs(num_main=15)

        # PP stage 0: layers [0, 7)
        _, _, dst_k0, dst_v0, n0 = self._mgr(0, 15).get_mha_kv_ptrs_with_pp(
            _src_ptrs(0, 7), dst
        )
        self.assertEqual(n0, 7)
        self.assertEqual(dst_k0, [_MAIN_K + i for i in range(0, 7)])
        self.assertEqual(dst_v0, [_MAIN_V + i for i in range(0, 7)])

        # PP stage 1: layers [7, 15) — the slice that previously leaked K14
        # into V and dropped the tail.
        _, _, dst_k1, dst_v1, n1 = self._mgr(7, 15).get_mha_kv_ptrs_with_pp(
            _src_ptrs(7, 8), dst
        )
        self.assertEqual(n1, 8)
        self.assertEqual(dst_k1, [_MAIN_K + i for i in range(7, 15)])
        self.assertEqual(dst_v1, [_MAIN_V + i for i in range(7, 15)])
        # No K pointer must ever appear in the V slice.
        self.assertTrue(all(p >= _MAIN_V for p in dst_v1))

    def test_draft_kv_layout_preserved(self):
        """Decode has draft KV (#17212), prefill PP=1 over 4 main layers.

        dst layout [K_main(4), V_main(4), draft_K(2), draft_V(2)]; V_main must
        be selected, not the draft section.
        """
        dst = _dst_ptrs(num_main=4, num_draft=2)
        _, _, dst_k, dst_v, n = self._mgr(0, 4).get_mha_kv_ptrs_with_pp(
            _src_ptrs(0, 4), dst
        )
        self.assertEqual(n, 4)
        self.assertEqual(dst_k, [_MAIN_K + i for i in range(4)])
        self.assertEqual(dst_v, [_MAIN_V + i for i in range(4)])

    def test_even_pp_full_match(self):
        """Prefill and decode hold the same number of layers."""
        dst = _dst_ptrs(num_main=4)
        _, _, dst_k, dst_v, n = self._mgr(0, 4).get_mha_kv_ptrs_with_pp(
            _src_ptrs(0, 4), dst
        )
        self.assertEqual(n, 4)
        self.assertEqual(dst_k, [_MAIN_K + i for i in range(4)])
        self.assertEqual(dst_v, [_MAIN_V + i for i in range(4)])

    def test_missing_total_layers_falls_back_to_plain_pp(self):
        """Without prefill_num_total_layers, default to the plain-PP slice
        (correct for the common no-draft case) rather than the old modulo
        heuristic."""
        dst = _dst_ptrs(num_main=15)
        _, _, dst_k, dst_v, n = self._mgr(7).get_mha_kv_ptrs_with_pp(
            _src_ptrs(7, 8), dst
        )
        self.assertEqual(dst_k, [_MAIN_K + i for i in range(7, 15)])
        self.assertEqual(dst_v, [_MAIN_V + i for i in range(7, 15)])


if __name__ == "__main__":
    unittest.main()
