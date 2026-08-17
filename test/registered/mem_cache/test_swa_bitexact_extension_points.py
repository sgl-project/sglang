"""Interface tests for the extension points the strict SWA HiCache hangs on.

These lock the shape of the hooks rather than any behaviour: the base-class
implementations must exist, accept the arguments their callers pass, and do
nothing. A shared call site that reaches the base class is what makes the
feature inert when it is not wired, so losing one of these silently would turn
an unwired deployment into an AttributeError on the hot path.
"""

import types
import unittest

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache, MatchPrefixParams
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=15, suite="stage-b-test-1-gpu-small-amd")


class TestForReuseParam(unittest.TestCase):
    def test_defaults_to_false(self):
        # Every existing caller omits it and must keep the self-match behaviour.
        self.assertIs(MatchPrefixParams(None).for_reuse, False)

    def test_accepts_true_without_touching_other_fields(self):
        a = MatchPrefixParams(None)
        b = MatchPrefixParams(None, for_reuse=True)
        differing = [
            f
            for f in a.__dataclass_fields__
            if getattr(a, f) != getattr(b, f) and f != "for_reuse"
        ]
        self.assertEqual(differing, [])


class TestCaptureHookDefaults(unittest.TestCase):
    """AttentionBackend's capture hooks default to doing nothing."""

    def test_prefill_hook_is_noop(self):
        kv = torch.zeros(4, 2)
        fb = types.SimpleNamespace()
        self.assertIsNone(AttentionBackend.capture_swa_windows(None, 0, kv, fb))

    def test_decode_hooks_are_noop(self):
        fb = types.SimpleNamespace()
        self.assertIsNone(AttentionBackend.capture_swa_windows_decode(None, fb))
        self.assertIsNone(
            AttentionBackend.capture_compress_state_windows_decode(None, fb)
        )

    def test_a_backend_without_capture_still_answers_the_hooks(self):
        class Bare(AttentionBackend):
            pass

        b = Bare()
        for name in (
            "capture_swa_windows",
            "capture_swa_windows_decode",
            "capture_compress_state_windows_decode",
        ):
            self.assertTrue(callable(getattr(b, name)), name)


class TestRestoreHookDefault(unittest.TestCase):
    def test_restore_is_noop(self):
        reqs = [types.SimpleNamespace(req_pool_idx=0)]
        idx = torch.zeros(1, dtype=torch.int64)
        self.assertIsNone(BasePrefixCache.restore_swa_windows(None, reqs, idx))


if __name__ == "__main__":
    unittest.main()
