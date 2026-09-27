"""Unit tests for the speculative-decoding verify span.

set_spec_verify_end_time_batch closes the spec_verify span for a verify batch.
These tests run it against the real SchedulerReqTimeStats and observe what
reaches the tracing context: one SPEC_VERIFY slice per request carrying the
drafts-only count (accept_lens minus the bonus token), and no host read of
accept_lens when tracing is off. They also pin the setters that the speculative
workers pass to set_time_batch by name, since a renamed or removed setter would
otherwise fail only at runtime, and only with tracing enabled.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import sglang.srt.observability.req_time_stats as rts
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class _FakeDeviceTensor:
    """Stands in for accept_lens: records whether it was read to the host."""

    def __init__(self, values):
        self._values = list(values)
        self._counter = [0]

    def __sub__(self, other):
        # Share the counter with the derived tensor: the host read happens on
        # `accept_lens - 1`, not on accept_lens itself, so counting only the
        # original would let a host read go unnoticed.
        derived = _FakeDeviceTensor([v - other for v in self._values])
        derived._counter = self._counter
        return derived

    def tolist(self):
        self._counter[0] += 1
        return list(self._values)

    @property
    def host_reads(self):
        return self._counter[0]


def _traced_stats():
    stats = rts.SchedulerReqTimeStats()
    stats.trace_ctx = mock.MagicMock()
    stats.trace_ctx.tracing_enable = True
    return stats


def _emitted_slices(stats):
    return [call.args[0] for call in stats.trace_ctx.trace_slice.call_args_list]


class TestSpecVerifyEndTime(CustomTestCase):
    def test_verify_slice_carries_drafts_only_count(self):
        stats = _traced_stats()
        stats.set_spec_verify_start_time(10.0)
        stats.set_spec_verify_end_time(12.5, num_correct_drafts=3)

        (verify_slice,) = _emitted_slices(stats)
        self.assertEqual(
            verify_slice.slice_name, rts.RequestStage.SPEC_VERIFY.stage_name
        )
        self.assertEqual(
            verify_slice.start_time_ns, rts.convert_time_to_realtime_ns(10.0)
        )
        self.assertEqual(
            verify_slice.end_time_ns, rts.convert_time_to_realtime_ns(12.5)
        )
        self.assertEqual(verify_slice.attrs, {"num_correct_drafts": 3})


class TestSetSpecVerifyEndTimeBatch(CustomTestCase):
    def _reqs(self, n, start):
        reqs = []
        for _ in range(n):
            stats = _traced_stats()
            stats.set_spec_verify_start_time(start)
            reqs.append(SimpleNamespace(time_stats=stats))
        return reqs

    def test_tracing_off_leaves_accept_lens_on_device(self):
        reqs = self._reqs(3, start=1.0)
        accept_lens = _FakeDeviceTensor([3, 2, 4])

        with mock.patch.object(rts, "get_global_tracing_enabled", return_value=False):
            rts.set_spec_verify_end_time_batch(reqs, accept_lens)

        self.assertEqual(accept_lens.host_reads, 0)
        for req in reqs:
            self.assertEqual(_emitted_slices(req.time_stats), [])

    def test_each_request_gets_its_own_drafts_only_count(self):
        reqs = self._reqs(3, start=1.0)
        accept_lens = _FakeDeviceTensor([3, 2, 4])

        with mock.patch.object(rts, "get_global_tracing_enabled", return_value=True):
            rts.set_spec_verify_end_time_batch(reqs, accept_lens)

        slices = [_emitted_slices(req.time_stats) for req in reqs]
        self.assertEqual([len(s) for s in slices], [1, 1, 1])
        self.assertEqual([s[0].attrs["num_correct_drafts"] for s in slices], [2, 1, 3])
        # One timestamp for the batch, one host read for the batch.
        self.assertEqual(len({s[0].end_time_ns for s in slices}), 1)
        self.assertEqual(accept_lens.host_reads, 1)


class TestSpecSettersDispatchedByName(CustomTestCase):
    """The workers name these setters as strings; set_time_batch resolves them."""

    def test_start_setters_stamp_their_field(self):
        for setter, field in (
            ("set_spec_draft_start_time", "spec_draft_start_time"),
            ("set_spec_verify_start_time", "spec_verify_start_time"),
        ):
            with self.subTest(setter=setter):
                stats = _traced_stats()
                with mock.patch.object(
                    rts, "get_global_tracing_enabled", return_value=True
                ):
                    rts.set_time_batch(
                        [SimpleNamespace(time_stats=stats)], setter, trace_only=True
                    )
                self.assertGreater(getattr(stats, field), 0.0)

    def test_draft_end_setter_emits_spec_draft_slice(self):
        stats = _traced_stats()
        stats.set_spec_draft_start_time(1.0)
        with mock.patch.object(rts, "get_global_tracing_enabled", return_value=True):
            rts.set_time_batch(
                [SimpleNamespace(time_stats=stats)],
                "set_spec_draft_end_time",
                trace_only=True,
            )

        (draft_slice,) = _emitted_slices(stats)
        self.assertEqual(draft_slice.slice_name, rts.RequestStage.SPEC_DRAFT.stage_name)


if __name__ == "__main__":
    unittest.main()
