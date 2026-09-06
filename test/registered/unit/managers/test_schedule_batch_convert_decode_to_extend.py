"""The dp decode->extend view must place the new token at seq_len - 1.

Deriving the prefix from len(origin_input_ids) + len(output_ids) put RoPE one
position past the row's own KV slot whenever the overlap output_ids lag was
drained.
"""

import unittest
from array import array

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import (  # noqa: E402
    ForwardMode,
    Req,
    ScheduleBatch,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeReq:
    """Carries the fill-id state convert_decode_to_extend touches, with the
    real Req methods so the array bookkeeping is not re-implemented here."""

    _refresh_fill_ids = Req._refresh_fill_ids
    set_extend_range = Req.set_extend_range

    def __init__(self, *, num_prompt_tokens: int, num_output_tokens: int):
        self.origin_input_ids = array("l", range(num_prompt_tokens))
        self.output_ids = list(range(num_output_tokens))
        self.full_untruncated_fill_ids = array("l", self.origin_input_ids)
        self.extend_range = None
        self.beam_group = None


def _make_converted_batch(*, rows, output_lag, extra_rows=0):
    """rows: (num_prompt_tokens, seq_len) per request. output_lag: how many
    tokens output_ids trails seq_len by (1 in steady decode, 0 once drained)."""
    reqs = [
        _FakeReq(
            num_prompt_tokens=num_prompt_tokens,
            num_output_tokens=seq_len - num_prompt_tokens - output_lag,
        )
        for num_prompt_tokens, seq_len in rows
    ]
    seq_lens = [seq_len for _, seq_len in rows]
    batch = ScheduleBatch(reqs=reqs)
    batch.forward_mode = ForwardMode.DECODE
    batch.enable_overlap = True
    # A beam tail appends rows after the reqs-aligned ones; mimic it by
    # repeating the last row, which is what append_beam_tail does.
    tail = [seq_lens[-1]] * extra_rows
    batch.seq_lens_cpu = torch.tensor(seq_lens + tail, dtype=torch.int64)
    batch.convert_decode_to_extend()
    return batch, seq_lens


class TestConvertDecodeToExtendGeometry(CustomTestCase):
    def _assert_geometry(self, batch, seq_lens):
        self.assertEqual(batch.forward_mode, ForwardMode.EXTEND)
        self.assertEqual(batch.prefix_lens, [s - 1 for s in seq_lens])
        self.assertEqual(batch.extend_lens, [1] * len(seq_lens))
        self.assertEqual(batch.extend_num_tokens, len(seq_lens))
        for req, seq_len in zip(batch.reqs, seq_lens, strict=True):
            self.assertEqual(tuple(req.extend_range), (seq_len - 1, seq_len))
        # What the attention path actually consumes: arange(prefix, prefix+len)
        # must land on the slot prepare_for_decode allocated, at seq_len - 1.
        for prefix_len, extend_len, seq_len in zip(
            batch.prefix_lens, batch.extend_lens, seq_lens, strict=True
        ):
            self.assertEqual(prefix_len + extend_len, seq_len)

    def test_geometry_with_the_overlap_lag_present(self):
        """Steady decode: the previous step's token is not in output_ids yet."""
        batch, seq_lens = _make_converted_batch(
            rows=[(6, 8), (96, 97), (142, 143)], output_lag=1
        )
        self._assert_geometry(batch, seq_lens)

    def test_geometry_once_the_overlap_lag_is_drained(self):
        """An iteration that ran prefill instead of decode lets the pending
        result land, so origin + output_ids reaches seq_len. The pre-fix
        formula returned prefix_len == seq_len here, i.e. RoPE one past the
        row's own KV slot."""
        batch, seq_lens = _make_converted_batch(
            rows=[(6, 8), (96, 97), (142, 143)], output_lag=0
        )
        self._assert_geometry(batch, seq_lens)

    def test_prefix_lens_stays_reqs_aligned_under_a_beam_tail(self):
        """seq_lens_cpu carries beam member rows with no req of their own; the
        reqs-aligned lists must not grow to the row count."""
        batch, seq_lens = _make_converted_batch(
            rows=[(6, 8), (96, 97)], output_lag=1, extra_rows=3
        )
        self.assertEqual(len(batch.seq_lens_cpu), len(seq_lens) + 3)
        self.assertEqual(len(batch.prefix_lens), len(batch.reqs))
        self._assert_geometry(batch, seq_lens)

    def test_short_seq_lens_fails_loudly(self):
        """A row/req divergence the slice cannot explain must raise, not
        silently truncate the way a plain zip would."""
        reqs = [_FakeReq(num_prompt_tokens=6, num_output_tokens=1) for _ in range(3)]
        batch = ScheduleBatch(reqs=reqs)
        batch.forward_mode = ForwardMode.DECODE
        batch.enable_overlap = True
        batch.seq_lens_cpu = torch.tensor([8, 8], dtype=torch.int64)
        with self.assertRaises(ValueError):
            batch.convert_decode_to_extend()


if __name__ == "__main__":
    unittest.main()
