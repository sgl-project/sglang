"""Replicated draft execution resets all DCP state and restores the target."""

import unittest

from sglang.srt.layers.dcp import draft_forward_guard
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestDCPDraftGuard(CustomTestCase):
    def test_restores_target_after_nested_guard_and_exception(self):
        for size in (2, 4, 8):
            with get_parallel().override(
                dcp_enabled=True, attn_dcp_size=size, attn_dcp_rank=size - 1
            ):
                with draft_forward_guard(enabled=False):
                    self.assertEqual(get_parallel().attn_dcp_size, size)
                with self.assertRaisesRegex(RuntimeError, "intentional"):
                    with draft_forward_guard():
                        self.assertFalse(get_parallel().dcp_enabled)
                        self.assertEqual(get_parallel().attn_dcp_size, 1)
                        self.assertEqual(get_parallel().attn_dcp_rank, 0)
                        with draft_forward_guard():
                            self.assertFalse(get_parallel().dcp_enabled)
                        raise RuntimeError("intentional")
                self.assertTrue(get_parallel().dcp_enabled)
                self.assertEqual(get_parallel().attn_dcp_size, size)
                self.assertEqual(get_parallel().attn_dcp_rank, size - 1)

    def test_disabled_flag_does_not_leave_stale_size_or_rank(self):
        for size in (2, 4, 8):
            with get_parallel().override(
                dcp_enabled=False, attn_dcp_size=size, attn_dcp_rank=size - 1
            ):
                with draft_forward_guard():
                    self.assertFalse(get_parallel().dcp_enabled)
                    self.assertEqual(get_parallel().attn_dcp_size, 1)
                    self.assertEqual(get_parallel().attn_dcp_rank, 0)
                self.assertEqual(get_parallel().attn_dcp_size, size)
                self.assertEqual(get_parallel().attn_dcp_rank, size - 1)


if __name__ == "__main__":
    unittest.main()
