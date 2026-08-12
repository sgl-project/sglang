"""CPU unit test for ``draft_forward_guard`` — the DCP-disable scope for draft forwards.

Under spec x DCP the target KV pool is sharded but the draft pool is replicated, so every DCP
branch must observe ``dcp_enabled == False`` for the whole draft forward. This used to be enforced
twice: by this guard AND by a shadow ``ModelRunner.dcp_size = 1 if is_draft_worker`` field. Upstream
has since removed ``ModelRunner.dcp_size``/``dcp_rank`` entirely and made the attention-facing
accessors derive from ``dcp_enabled``:

    attn_dcp_size = dcp_size if dcp_enabled else 1      (runtime_context.py)
    attn_dcp_rank = dcp_rank if dcp_enabled else 0

so the guard is now the *only* mechanism. That makes these assertions load-bearing rather than
redundant: if the guard stops driving ``attn_dcp_size`` to 1, draft forwards silently take DCP
branches against an unsharded pool, which is the chain-decode corruption this exists to prevent.

Usage:
    python -m pytest test_dcp_draft_guard.py -v
"""

import unittest

from sglang.srt import runtime_context as rc
from sglang.srt.layers.dcp import draft_forward_guard
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, stage="base-a")


class TestDraftForwardGuard(CustomTestCase):
    def test_guard_disables_dcp_for_draft(self):
        """is_draft=True must drive attn_dcp_size -> 1 and attn_dcp_rank -> 0."""
        parallel = rc.get_parallel()
        with parallel.override(dcp_size=8, dcp_enabled=True, dcp_rank=3):
            self.assertEqual(parallel.attn_dcp_size, 8)
            self.assertEqual(parallel.attn_dcp_rank, 3)

            with draft_forward_guard(True):
                # The whole point: the draft forward must see an unsharded world.
                self.assertFalse(parallel.dcp_enabled)
                self.assertEqual(parallel.attn_dcp_size, 1)
                self.assertEqual(parallel.attn_dcp_rank, 0)

            # ...and the target state must come back afterwards.
            self.assertTrue(parallel.dcp_enabled)
            self.assertEqual(parallel.attn_dcp_size, 8)
            self.assertEqual(parallel.attn_dcp_rank, 3)

    def test_guard_is_noop_for_target(self):
        """is_draft=False must not perturb DCP state (target forwards keep sharding)."""
        parallel = rc.get_parallel()
        with parallel.override(dcp_size=8, dcp_enabled=True, dcp_rank=3):
            with draft_forward_guard(False):
                self.assertTrue(parallel.dcp_enabled)
                self.assertEqual(parallel.attn_dcp_size, 8)
                self.assertEqual(parallel.attn_dcp_rank, 3)

    def test_guard_restores_on_exception(self):
        """A draft forward that raises must still restore DCP state."""
        parallel = rc.get_parallel()
        with parallel.override(dcp_size=4, dcp_enabled=True, dcp_rank=1):
            with self.assertRaises(RuntimeError):
                with draft_forward_guard(True):
                    self.assertEqual(parallel.attn_dcp_size, 1)
                    raise RuntimeError("draft forward blew up")
            self.assertTrue(parallel.dcp_enabled)
            self.assertEqual(parallel.attn_dcp_size, 4)

    def test_guard_nests(self):
        """Nested guards (chain draft inside a guarded forward) must stay disabled."""
        parallel = rc.get_parallel()
        with parallel.override(dcp_size=8, dcp_enabled=True, dcp_rank=2):
            with draft_forward_guard(True):
                with draft_forward_guard(True):
                    self.assertEqual(parallel.attn_dcp_size, 1)
                self.assertEqual(parallel.attn_dcp_size, 1)
            self.assertEqual(parallel.attn_dcp_size, 8)

    def test_guard_harmless_without_dcp(self):
        """With DCP off entirely the guard must be a no-op, not an error."""
        parallel = rc.get_parallel()
        with parallel.override(dcp_size=1, dcp_enabled=False, dcp_rank=0):
            with draft_forward_guard(True):
                self.assertEqual(parallel.attn_dcp_size, 1)
                self.assertEqual(parallel.attn_dcp_rank, 0)
            self.assertEqual(parallel.attn_dcp_size, 1)


if __name__ == "__main__":
    unittest.main()
