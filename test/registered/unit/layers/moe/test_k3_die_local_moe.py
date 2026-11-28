"""Unit tests for the K3 die-local routed MoE (SGLANG_K3_ROUTED_MOE_DEEPGEMM)."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import contextlib
import json
import types
import unittest
from unittest import mock

from sglang.srt.environ import envs
from sglang.srt.layers.communication import k3_ar_fusion
from sglang.srt.layers.moe import k3_deepgemm_routed as routed
from sglang.srt.layers.moe import k3_die_local_moe as policy
from sglang.test.test_utils import CustomTestCase


@contextlib.contextmanager
def _env(ar_fusion=False):
    with (
        envs.SGLANG_K3_ROUTED_MOE_DEEPGEMM.override(True),
        envs.SGLANG_K3_AR_FUSION.override(ar_fusion),
    ):
        policy.reset_caches()
        try:
            yield
        finally:
            policy.reset_caches()


def _taken(verify):
    token = policy._verify.set(verify)
    try:
        return [rows for rows in range(1, 300) if policy.takes(rows)]
    finally:
        policy._verify.reset(token)


def _released_deep_gemm():
    def fp8_fp4_mega_moe(y, l1_weights, l2_weights, sym_buffer, weight_layout=None):
        pass

    class SymmBuffer:
        def __init__(self, group, num_experts, mma_type=None):
            pass

    return types.SimpleNamespace(
        fp8_fp4_mega_moe=fp8_fp4_mega_moe,
        mega=types.SimpleNamespace(SymmBuffer=SymmBuffer),
        __version__="0.2.9",
    )


class TestK3DieLocalMoE(CustomTestCase):
    def test_deep_gemm_without_the_api_keeps_the_default_path(self):
        config = types.SimpleNamespace(
            activation="situ",
            gemm1_alpha=4.0,
            gemm1_clamp_limit=25.0,
            gate_up_interleaved=False,
        )
        layer = types.SimpleNamespace(moe_runner_config=config)
        warnings = []
        routed.missing_capability.cache_clear()
        self.addCleanup(routed.missing_capability.cache_clear)
        with (
            _env(),
            mock.patch.object(routed, "_deep_gemm", _released_deep_gemm),
            mock.patch.object(policy, "print_warning_once", warnings.append),
        ):
            self.assertFalse(policy.wanted(layer, False, "default"))
        self.assertEqual(len(warnings), 1)
        self.assertIn("SGLANG_K3_ROUTED_MOE_DEEPGEMM is set but not used", warnings[0])
        self.assertIn("deep_gemm 0.2.9 lacks the one-rank MegaMoE API", warnings[0])

    def test_in_tree_table_dispatch(self):
        table = json.loads(policy._TABLE.read_text())
        fold = table["ar_fusion_fold_max_rows"]
        self.assertEqual(
            table["widths_ar_fusion"], [rows for rows in table["widths"] if rows > fold]
        )
        # the fold limit of the table is the one k3_ar_fusion applies
        state = types.SimpleNamespace(push_slot_bytes=1 << 30)
        with mock.patch.object(k3_ar_fusion, "_get_state", return_value=state):
            self.assertTrue(k3_ar_fusion.finalize_push_fits(fold))
            self.assertFalse(k3_ar_fusion.finalize_push_fits(fold + 1))
        for ar_fusion, key in ((False, "widths"), (True, "widths_ar_fusion")):
            with self.subTest(ar_fusion=ar_fusion), _env(ar_fusion):
                self.assertEqual(_taken(verify=True), table[key])
                self.assertEqual(_taken(verify=False), [])

    def test_page_homes(self):
        page = 2 << 20
        self.assertEqual(policy.page_homes((3, 4, page), 1, page), [0, 0, 1, 1] * 3)
        # a page split evenly between the domains alternates by page index
        self.assertEqual(policy.page_homes((4, 2, page // 2), 1, page), [0, 1, 0, 1])


if __name__ == "__main__":
    unittest.main()
