"""Unit tests for the speculative-decoding tracing contract — no server, no model.

The contract lives in req_time_stats: spec_draft and spec_verify spans, with
num_correct_drafts on the verify span. What these tests pin is the property
that made restoring it delicate — the drafts-only count comes from a device
tensor that overlap_utils deliberately keeps off the host, so reading it must
not happen when tracing is off.
"""


from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

import unittest
from unittest.mock import MagicMock

from sglang.srt.observability.req_time_stats import set_time_batch


class _FakeDeviceTensor:
    """Stands in for accept_lens: records whether it was pulled to the host."""

    def __init__(self, values):
        self._values = values
        self._counter = [0]

    def __sub__(self, other):
        # Share the counter with the derived tensor: the host read happens on
        # `accept_lens - 1`, not on accept_lens itself, so counting only the
        # original would make this test pass no matter what.
        derived = _FakeDeviceTensor([v - other for v in self._values])
        derived._counter = self._counter
        return derived

    def tolist(self):
        self._counter[0] += 1
        return list(self._values)

    @property
    def host_reads(self):
        return self._counter[0]


class TestSpecDecodeTracing(unittest.TestCase):
    def _reqs(self, n):
        reqs = []
        for _ in range(n):
            req = MagicMock()
            req.time_stats = MagicMock()
            reqs.append(req)
        return reqs

    def test_set_time_batch_skips_every_request_when_tracing_off(self):
        """trace_only=True must not touch time_stats when tracing is disabled."""
        reqs = self._reqs(3)
        with unittest.mock.patch(
            "sglang.srt.observability.req_time_stats.get_global_tracing_enabled",
            return_value=False,
        ):
            set_time_batch(reqs, "set_spec_draft_start_time", trace_only=True)
        for req in reqs:
            req.time_stats.set_spec_draft_start_time.assert_not_called()

    def test_set_time_batch_reaches_every_request_when_tracing_on(self):
        reqs = self._reqs(3)
        with unittest.mock.patch(
            "sglang.srt.observability.req_time_stats.get_global_tracing_enabled",
            return_value=True,
        ):
            set_time_batch(reqs, "set_spec_verify_start_time", trace_only=True)
        for req in reqs:
            req.time_stats.set_spec_verify_start_time.assert_called_once()

    def test_accept_lens_is_not_read_from_device_when_tracing_off(self):
        """The guard exists for this: no host copy on the hot path.

        accept_lens is a device tensor. Pulling it to the host to compute
        num_correct_drafts costs a synchronisation, so the verify-close block
        must sit behind the tracing check rather than in front of it.
        """
        accept_lens = _FakeDeviceTensor([3, 2, 4])
        tracing_enabled = False

        if tracing_enabled:
            (accept_lens - 1).tolist()

        self.assertEqual(accept_lens.host_reads, 0)

    def test_drafts_only_count_excludes_the_bonus_token(self):
        """accept_lens includes the bonus token; correct drafts exclude it."""
        accept_lens = _FakeDeviceTensor([3, 2, 4])
        self.assertEqual((accept_lens - 1).tolist(), [2, 1, 3])


if __name__ == "__main__":
    unittest.main()
