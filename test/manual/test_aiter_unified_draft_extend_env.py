"""
Unit test for the aiter NEXTN spec-decode draft_extend routing flag.

aiter_backend.py routes EAGLE-v2 draft_extend (KV catch-up) through
unified_attention (GQA-packed + split-KV) instead of the occupancy-starved
mha_batch_prefill FMHA, gated behind SGLANG_AITER_UNIFIED_DRAFT_EXTEND. This
guards the flag's registration and default so an accidental rename or default
flip is caught in CI. The kernel path itself requires ROCm/gfx950 and is
covered by the accuracy/speed evidence in the PR.

Run:
    python test/manual/test_aiter_unified_draft_extend_env.py
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../python"))


class TestAiterUnifiedDraftExtendEnv(unittest.TestCase):
    def setUp(self):
        from sglang.srt.environ import envs

        self.flag = envs.SGLANG_AITER_UNIFIED_DRAFT_EXTEND

    def test_registered_and_default_on(self):
        # Registered as an EnvBool defaulting to True (path enabled in prod).
        self.assertTrue(self.flag.get())

    def test_override_toggles_and_restores(self):
        self.assertTrue(self.flag.get())
        with self.flag.override(False):
            self.assertFalse(self.flag.get())
        self.assertTrue(self.flag.get())


if __name__ == "__main__":
    unittest.main()
