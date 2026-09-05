"""CPU unit test for ``draft_forward_guard``.

The draft KV pool is replicated, not sharded, so draft forwards must observe
``dcp_enabled == False`` (which drives attn_dcp_size to 1 and attn_dcp_rank to 0).
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
                self.assertFalse(parallel.dcp_enabled)
                self.assertEqual(parallel.attn_dcp_size, 1)
                self.assertEqual(parallel.attn_dcp_rank, 0)

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
