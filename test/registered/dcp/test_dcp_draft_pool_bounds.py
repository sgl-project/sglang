"""CPU unit test for DCP-aware EAGLE draft KV-pool budgeting.

Under DCP the target KV pool is sharded across ranks, but the draft pool spans the allocator's
*widened virtual location space* — it is replicated, not sharded. So the draft term in cell_size
must scale with ``get_parallel().attn_dcp_size``. Upstream already applies exactly this rule to the
DFLASH term (``_dflash_draft_cell_size_per_token``); these tests pin the EAGLE counterpart at both
sites that budget an EAGLE draft:

  * ``DefaultPoolConfigurator``   — ratio form, (1 + dcp * draft_layers / num_layers)
  * ``HybridSWAPoolConfigurator`` — layer-count form, _draft_full_layers_num * dcp

Under-budgeting here does not fail loudly: the widened draft pool quietly eats the transient
headroom and the server OOMs at the first large prefill, far from the cause. Hence a unit test.

Usage:
    python -m pytest test_dcp_draft_pool_bounds.py -v
"""

import unittest
from types import SimpleNamespace

from sglang.srt import runtime_context as rc
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, stage="base-a")


class _Algo:
    """Minimal spec-algorithm stand-in; only the predicates the branch reads."""

    def __init__(self, eagle=False, dflash=False):
        self._eagle, self._dflash = eagle, dflash

    def is_eagle(self):
        return self._eagle

    def is_standalone(self):
        return False

    def is_dflash_family(self):
        return self._dflash

    def is_dflash(self):
        return self._dflash


def _kvc(draft_layers=2):
    return SimpleNamespace(
        spec_algorithm=_Algo(eagle=True),
        is_draft_worker=False,
        spec_aux_config=SimpleNamespace(
            eagle_draft_num_layers=draft_layers,
            dflash_draft_num_layers=None,
            dflash_draft_cell_size_per_token=None,
        ),
        server_args=SimpleNamespace(),
    )


class TestEagleDraftPoolDCPBudget(CustomTestCase):
    """DefaultPoolConfigurator: the draft term must scale with attn_dcp_size."""

    def _draft_layers(self, dcp_size, draft_layers=2):
        from sglang.srt.model_executor.pool_configurator import _eagle_draft_layers

        with rc.get_parallel().override(attn_dcp_size=dcp_size):
            return _eagle_draft_layers(_kvc(draft_layers))

    def _cell_size(self, dcp_size, draft_layers=2, num_layers=60, base=1000):
        # Mirrors the DefaultPoolConfigurator branch: cell_size * (1 + L_eff / L_target).
        return int(base * (1 + self._draft_layers(dcp_size, draft_layers) / num_layers))

    def test_dcp1_matches_upstream_ratio(self):
        # dcp=1 must reproduce the original (1 + L_draft/L_target) exactly.
        self.assertEqual(self._cell_size(1), int(1000 * (1 + 2 / 60)))

    def test_draft_term_scales_with_dcp(self):
        for dcp in (2, 4, 8):
            with self.subTest(dcp=dcp):
                self.assertEqual(self._draft_layers(dcp), 2 * dcp)
                self.assertEqual(self._cell_size(dcp), int(1000 * (1 + dcp * 2 / 60)))

    def test_draft_worker_is_not_budgeted(self):
        """The draft worker must not budget a draft pool into its own cell size."""
        from sglang.srt.model_executor.pool_configurator import _eagle_draft_layers

        kvc = _kvc()
        kvc.is_draft_worker = True
        with rc.get_parallel().override(attn_dcp_size=8):
            self.assertEqual(_eagle_draft_layers(kvc), 0)

    def test_budget_is_monotonic_in_dcp(self):
        sizes = [self._cell_size(d) for d in (1, 2, 4, 8)]
        self.assertEqual(sizes, sorted(sizes))
        self.assertLess(sizes[0], sizes[-1])

    def test_no_invented_draft_depth(self):
        """A None draft depth must NOT be silently replaced by a guessed floor.

        Inventing a depth trades a loud failure for a quiet under-allocation; upstream skips the
        scaling when the depth is unknown, and so must we.
        """
        from sglang.srt.model_executor.pool_configurator import _eagle_draft_layers

        kvc = _kvc()
        kvc.spec_aux_config.eagle_draft_num_layers = None
        with rc.get_parallel().override(attn_dcp_size=8):
            # 0 => caller skips the scaling entirely; no invented floor.
            self.assertEqual(_eagle_draft_layers(kvc), 0)


if __name__ == "__main__":
    unittest.main()
