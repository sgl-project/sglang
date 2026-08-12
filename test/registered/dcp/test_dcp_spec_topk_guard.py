"""CPU unit test for the DCP x speculative-decoding tree-drafting guard.

``_validate_dcp_spec_topk`` rejects ``--speculative-eagle-topk > 1`` under ``--dcp-size > 1``:
the DCP verify path resolves each draft token's causal bound from its GLOBAL position
(``causal_seqs`` = prefix + T), which can only express a linear draft chain. A tree draft
would need tree-causal masking in the verify kernel, which does not exist -- so without this
guard the configuration runs and returns silently wrong tokens.

The scoping is deliberately narrow and is the part most likely to regress:

  * EAGLE / EAGLE3 / DFLASH -> chain drafting, guarded.
  * DSPARK -> draws a one-shot block and never resolves speculative_eagle_topk. It must NOT
    be gated: ``is_dflash_family()`` is ``is_dflash() or is_dspark()``, so using it here would
    newly reject the Kimi-Linear + DSPARK + DCP path that already ships.

Usage:
    python -m pytest test_dcp_spec_topk_guard.py -v
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
        from sglang.srt.arg_groups.speculative_hook import _validate_dcp_spec_topk

        _validate_dcp_spec_topk(_args(algo, topk, dcp_size))

    def _assert_rejected(self, algo, topk=2, dcp_size=8):
        with self.assertRaises(ValueError) as ctx:
            self._validate(algo, topk, dcp_size)
        self.assertIn("chain speculative drafts", str(ctx.exception))

    # --- rejected: tree drafting on a chain-only verify path -------------------------
    def test_rejects_tree_draft_under_dcp(self):
        for algo in ("EAGLE", "EAGLE3"):
            with self.subTest(algo=algo):
                self._assert_rejected(algo)

    # --- allowed: chain drafting ------------------------------------------------------
    def test_allows_chain_draft_under_dcp(self):
        for algo in ("EAGLE", "EAGLE3", "DFLASH"):
            for topk in (None, 1):
                with self.subTest(algo=algo, topk=topk):
                    self._validate(algo, topk, dcp_size=8)

    # --- allowed: DCP off entirely ----------------------------------------------------
    def test_ignores_topk_without_dcp(self):
        for dcp_size in (None, 1):
            with self.subTest(dcp_size=dcp_size):
                self._validate("EAGLE3", topk=4, dcp_size=dcp_size or 1)

    def test_ignores_when_no_spec_algorithm(self):
        self._validate(None, topk=4, dcp_size=8)

    # --- the scoping regression this guard is most likely to suffer -------------------
    def test_dspark_is_not_gated(self):
        """DSPARK + DCP already ships (Kimi-Linear); it must not be newly rejected.

        Guards against someone "simplifying" is_dflash() to is_dflash_family(), which is
        is_dflash() or is_dspark().
        """
        # DSPARK never resolves speculative_eagle_topk, but even a stray >1 value must not
        # trip the guard, because the guard must not apply to DSPARK at all.
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
