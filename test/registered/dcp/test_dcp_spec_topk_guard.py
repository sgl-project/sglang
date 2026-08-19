"""CPU unit test for the DCP x speculative-decoding tree-drafting guard.

The DCP verify path folds draft tokens as a linear causal chain, so EAGLE /
EAGLE3 / STANDALONE / DFLASH require topk == 1. DSPARK is not gated: it never
resolves speculative_eagle_topk and already ships with DCP.
"""

import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, stage="base-a")


def _args(algo, topk, dcp_size):
    return SimpleNamespace(
        speculative_algorithm=algo,
        speculative_eagle_topk=topk,
        dcp_size=dcp_size,
    )


class TestDCPSpecTopkGuard(CustomTestCase):
    def _validate(self, algo, topk, dcp_size):
        from sglang.srt.arg_groups.speculative_hook import _validate_dcp_spec

        _validate_dcp_spec(_args(algo, topk, dcp_size))

    def _assert_rejected(self, algo, topk=2, dcp_size=8):
        with self.assertRaises(ValueError) as ctx:
            self._validate(algo, topk, dcp_size)
        self.assertIn("chain speculative drafts", str(ctx.exception))

    def test_rejects_tree_draft_under_dcp(self):
        for algo in ("EAGLE", "EAGLE3", "STANDALONE"):
            with self.subTest(algo=algo):
                self._assert_rejected(algo)

    def test_allows_chain_draft_under_dcp(self):
        for algo in ("EAGLE", "EAGLE3", "STANDALONE", "DFLASH"):
            for topk in (None, 1):
                with self.subTest(algo=algo, topk=topk):
                    self._validate(algo, topk, dcp_size=8)

    def test_ignores_topk_without_dcp(self):
        for dcp_size in (None, 1):
            with self.subTest(dcp_size=dcp_size):
                self._validate("EAGLE3", topk=4, dcp_size=dcp_size or 1)

    def test_ignores_when_no_spec_algorithm(self):
        self._validate(None, topk=4, dcp_size=8)

    def test_dspark_is_not_gated(self):
        """DSPARK + DCP already ships (Kimi-Linear); it must not be newly rejected."""
        for topk in (None, 1, 4):
            with self.subTest(topk=topk):
                self._validate("DSPARK", topk, dcp_size=8)

    def test_algorithm_predicates_are_not_getattr_defaulted(self):
        """A renamed predicate must break loudly, not silently disable the guard."""
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        algo = SpeculativeAlgorithm.from_string("EAGLE3")
        for name in ("is_eagle", "is_dflash", "is_dspark"):
            self.assertTrue(hasattr(algo, name), f"SpeculativeAlgorithm lost {name}()")


if __name__ == "__main__":
    unittest.main()
