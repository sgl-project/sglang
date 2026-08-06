"""Unit tests for CommonKVManager.get_mha_kv_ptrs_with_pp V-pointer offset.

Focus of these tests: the else-branch (case 3) that fires when
prefill's PP layout differs from decode's total KV pointer count in a
way that ISN'T the multiplier_ratio path. This case is triggered in
production by:

  * Non-MLA / non-hybrid-MLA models (regular MHA / GQA), AND
  * PP > 1 on the prefill side, AND
  * Decode side has appended MTP / NEXTN draft KV pointers to a
    target-model KV pool -> dst_kv_ptrs layout becomes
        [K_target..., V_target..., K_draft..., V_draft...]
    where `len(dst_kv_ptrs) // 2` OVERCOUNTS the target section by the
    number of draft K layers.

Before the fix, this branch used `dst_num_total_layers = len // 2` as
the V-pointer base, which shifts every target-V slice by `M`
(= number of draft layers) and, on the last PP rank, causes the tail
of the target-V writes to overwrite the leading draft-K pointers ->
decode reads corrupted target-V from layer 2 onwards, first token is
still correct (aux buffer path), but every subsequent decode step
attends to wrong V.

The fix adds an optional `num_dst_target_kv_layers` parameter that
prefill obtains from the wire-protocol frame filled in by decode BEFORE
draft KV was concatenated. When present and > 0, we use it as
`v_layer_offset` instead of `len // 2`.

These tests exercise the pointer arithmetic in isolation by calling
`CommonKVManager.get_mha_kv_ptrs_with_pp` unbound with a minimal fake
`self`. That keeps the tests independent of the surrounding
CommonKVManager __init__ (which requires ZMQ sockets, bootstrap
server, model config etc.) while still exercising the exact production
code path.
"""

import unittest
from types import SimpleNamespace
from typing import List, Tuple

from sglang.srt.disaggregation.common.conn import CommonKVManager


def _fake_manager(prefill_start_layer: int) -> SimpleNamespace:
    """Minimal object exposing `.kv_args.prefill_start_layer`.

    `get_mha_kv_ptrs_with_pp` only reads `self.kv_args.prefill_start_layer`,
    so a SimpleNamespace shim is sufficient and avoids the heavyweight
    CommonKVManager constructor.
    """
    return SimpleNamespace(
        kv_args=SimpleNamespace(prefill_start_layer=prefill_start_layer)
    )


def _call_case3(
    src_kv_ptrs: List[int],
    dst_kv_ptrs: List[int],
    prefill_start_layer: int,
    num_dst_target_kv_layers=None,
) -> Tuple[List[int], List[int], List[int], List[int], int]:
    return CommonKVManager.get_mha_kv_ptrs_with_pp(
        _fake_manager(prefill_start_layer),
        src_kv_ptrs,
        dst_kv_ptrs,
        num_dst_target_kv_layers=num_dst_target_kv_layers,
    )


class TestGetMhaKvPtrsWithPp(unittest.TestCase):
    """Cover case 3 (PP>1 else-branch) both without and with the fix's arg."""

    # Layout mimicking Qwen3.5-397B W4A8 + PP=4 prefill (15 full-attn
    # layers split 3/4/4/4) + decode with MTP=1 draft layer:
    #   dst_kv_ptrs layout: 15 target K + 15 target V + 1 draft K + 1 draft V
    # We encode each ptr as an integer tag so the assertions read as
    # "which slot did we select" instead of comparing opaque addresses.
    N_TARGET = 15
    N_DRAFT = 1

    # Tag scheme -- makes error messages self-describing.
    # 100..114 = target K layers 0..14
    # 200..214 = target V layers 0..14
    # 300      = draft K layer 0
    # 400      = draft V layer 0
    DST_KV_PTRS = (
        [100 + i for i in range(N_TARGET)]  # K_t
        + [200 + i for i in range(N_TARGET)]  # V_t
        + [300 + i for i in range(N_DRAFT)]  # K_d
        + [400 + i for i in range(N_DRAFT)]  # V_d
    )

    def _run_qwen35_rank(
        self,
        num_layers_this_rank: int,
        prefill_start_layer: int,
        num_dst_target_kv_layers,
    ):
        # Prefill side has only this rank's slice of the (K,V) pool.
        # Values don't matter for the case-3 dst indexing test; we build
        # a plausible K||V src list.
        src_kv_ptrs = (
            [500 + i for i in range(num_layers_this_rank)]  # local K
            + [600 + i for i in range(num_layers_this_rank)]  # local V
        )
        return _call_case3(
            src_kv_ptrs=src_kv_ptrs,
            dst_kv_ptrs=list(self.DST_KV_PTRS),
            prefill_start_layer=prefill_start_layer,
            num_dst_target_kv_layers=num_dst_target_kv_layers,
        )

    def _expected_v_slice(self, start: int, count: int) -> List[int]:
        return [200 + start + i for i in range(count)]

    def _expected_k_slice(self, start: int, count: int) -> List[int]:
        return [100 + start + i for i in range(count)]

    # -- with the fix (num_dst_target_kv_layers=15) --------------------

    def test_rank0_fixed(self):
        _, _, dst_k, dst_v, n = self._run_qwen35_rank(
            num_layers_this_rank=3,
            prefill_start_layer=0,
            num_dst_target_kv_layers=self.N_TARGET,
        )
        self.assertEqual(n, 3)
        self.assertEqual(dst_k, self._expected_k_slice(0, 3))
        self.assertEqual(dst_v, self._expected_v_slice(0, 3))

    def test_rank1_fixed(self):
        _, _, dst_k, dst_v, n = self._run_qwen35_rank(
            num_layers_this_rank=4,
            prefill_start_layer=3,
            num_dst_target_kv_layers=self.N_TARGET,
        )
        self.assertEqual(n, 4)
        self.assertEqual(dst_k, self._expected_k_slice(3, 4))
        self.assertEqual(dst_v, self._expected_v_slice(3, 4))

    def test_rank2_fixed(self):
        _, _, dst_k, dst_v, n = self._run_qwen35_rank(
            num_layers_this_rank=4,
            prefill_start_layer=7,
            num_dst_target_kv_layers=self.N_TARGET,
        )
        self.assertEqual(n, 4)
        self.assertEqual(dst_k, self._expected_k_slice(7, 4))
        self.assertEqual(dst_v, self._expected_v_slice(7, 4))

    def test_rank3_fixed_does_not_touch_draft_k(self):
        _, _, dst_k, dst_v, n = self._run_qwen35_rank(
            num_layers_this_rank=4,
            prefill_start_layer=11,
            num_dst_target_kv_layers=self.N_TARGET,
        )
        self.assertEqual(n, 4)
        self.assertEqual(dst_k, self._expected_k_slice(11, 4))
        self.assertEqual(dst_v, self._expected_v_slice(11, 4))
        # Regression guard: no draft-K tag (300) may appear in dst_v.
        self.assertNotIn(300, dst_v)

    # -- without the fix (legacy sentinel) reproduces the shift --------

    def test_rank3_legacy_overwrites_draft_k(self):
        """Sanity: with sentinel -> len//2=16 -> rank3 last slot picks up
        draft-K (tag 300). This is the pre-fix bug we're guarding
        against. If this assertion ever fails, the fallback path has
        drifted and the test suite is no longer catching the regression.
        """
        _, _, _, dst_v, _ = self._run_qwen35_rank(
            num_layers_this_rank=4,
            prefill_start_layer=11,
            num_dst_target_kv_layers=None,
        )
        # Legacy behavior: v_layer_offset = len(dst_kv_ptrs)//2 = 16.
        # rank 3 V slice = dst_kv_ptrs[16+11 : 16+15] = tags 227..230.
        # Tag 230 is the draft-K ptr (300 - N_TARGET*2 index = 30).
        self.assertEqual(dst_v[-1], 300, "expected pre-fix corruption to expose draft-K in dst_v tail")

    # -- backward-compat: -1 sentinel behaves like the legacy path -----

    def test_sentinel_minus_one_behaves_like_legacy(self):
        _, _, _, dst_v_a, _ = self._run_qwen35_rank(
            num_layers_this_rank=4,
            prefill_start_layer=11,
            num_dst_target_kv_layers=-1,
        )
        _, _, _, dst_v_b, _ = self._run_qwen35_rank(
            num_layers_this_rank=4,
            prefill_start_layer=11,
            num_dst_target_kv_layers=None,
        )
        self.assertEqual(dst_v_a, dst_v_b, "-1 and None must both mean 'fall back to len//2'")

    # -- no-MTP layouts: the fix is a no-op ----------------------------

    def test_no_mtp_layout_unaffected(self):
        """Decode without MTP: dst_kv_ptrs is [K..., V...] and
        `len//2 == num_target_kv_layers`; either code path must produce
        identical slices."""
        n = 15
        dst = [100 + i for i in range(n)] + [200 + i for i in range(n)]
        src = [500 + i for i in range(4)] + [600 + i for i in range(4)]
        legacy = _call_case3(src, dst, prefill_start_layer=3, num_dst_target_kv_layers=None)
        fixed = _call_case3(src, dst, prefill_start_layer=3, num_dst_target_kv_layers=n)
        self.assertEqual(legacy, fixed)

    # -- case 1 / case 2 branches untouched by the new arg -------------

    def test_case1_equal_layers_ignores_new_arg(self):
        """Case 1 (num_kv == dst_num_total): new arg must not perturb."""
        n = 15
        src = [500 + i for i in range(n)] + [600 + i for i in range(n)]
        dst = [100 + i for i in range(n)] + [200 + i for i in range(n)]
        without = _call_case3(src, dst, 0, num_dst_target_kv_layers=None)
        withit = _call_case3(src, dst, 0, num_dst_target_kv_layers=99)
        self.assertEqual(without, withit)

    def test_case2_pp1_multiplier_ignores_new_arg(self):
        """Case 2 (PP=1, dst_num_total % num_kv != 0) uses its own
        multiplier_ratio path; the new arg must not perturb it."""
        # 4 local layers -> src = 4K + 4V; dst has 4 target + 1 draft
        # per K/V section -> dst has 4+4+1+1=10 -> len//2=5, 5%4!=0
        src = [500 + i for i in range(4)] + [600 + i for i in range(4)]
        dst = (
            [100 + i for i in range(4)]
            + [200 + i for i in range(4)]
            + [300]
            + [400]
        )
        without = _call_case3(src, dst, 0, num_dst_target_kv_layers=None)
        withit = _call_case3(src, dst, 0, num_dst_target_kv_layers=4)
        self.assertEqual(without, withit)


if __name__ == "__main__":
    unittest.main()
