"""The DP extend view of a decode step must place the new token at seq_len - 1."""

import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import (  # noqa: E402
    ForwardMode,
    ScheduleBatch,
)

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


def _make_viewed_step(seq_lens, extra_rows=0):
    step = ScheduleBatch(reqs=[object() for _ in seq_lens])
    step.forward_mode = ForwardMode.DECODE
    # A beam tail appends rows after the reqs-aligned ones; mimic it by
    # repeating the last row, which is what append_beam_tail does.
    tail = [seq_lens[-1]] * extra_rows
    step.seq_lens_cpu = torch.tensor(seq_lens + tail, dtype=torch.int64)
    step.set_extend_view()
    return step


class TestExtendViewGeometry(CustomTestCase):
    def _assert_geometry(self, step, seq_lens):
        self.assertEqual(step.forward_mode, ForwardMode.EXTEND)
        self.assertEqual(step.prefix_lens, [s - 1 for s in seq_lens])
        self.assertEqual(step.extend_lens, [1] * len(seq_lens))
        self.assertEqual(step.extend_num_tokens, len(seq_lens))
        # Every row decodes: none of them is a prefill the launch moves.
        self.assertEqual(step.prefill_reqs(), [])

    def test_geometry(self):
        seq_lens = [8, 97, 143]
        self._assert_geometry(_make_viewed_step(seq_lens), seq_lens)

    def test_prefix_lens_stays_reqs_aligned_under_a_beam_tail(self):
        """seq_lens_cpu carries beam member rows with no req of their own; the
        reqs-aligned lists must not grow to the row count."""
        seq_lens = [8, 97]
        step = _make_viewed_step(seq_lens, extra_rows=3)
        self.assertEqual(len(step.seq_lens_cpu), len(seq_lens) + 3)
        self._assert_geometry(step, seq_lens)

    def test_short_seq_lens_fails_loudly(self):
        """A row/req divergence the slice cannot explain must raise, not
        silently truncate."""
        step = ScheduleBatch(reqs=[object() for _ in range(3)])
        step.forward_mode = ForwardMode.DECODE
        step.seq_lens_cpu = torch.tensor([8, 8], dtype=torch.int64)
        with self.assertRaises(ValueError):
            step.set_extend_view()


if __name__ == "__main__":
    unittest.main()
