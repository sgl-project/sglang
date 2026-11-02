"""Unit tests for request-dump sampling in TokenizerManager.

`dump_requests_sample_fraction` (set via /configure_logging) dumps only a
hash-selected subset of requests. The sampling decision must be a
deterministic function of the rid — not a random draw — so that a given
request is kept or dropped consistently across retries and across workers
sharing a dump folder, and so that raising the fraction only ever adds
requests to the sample (nested samples) instead of reshuffling it.
"""

import unittest
from unittest.mock import MagicMock, Mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import ConfigureLoggingReq
from sglang.srt.managers.tokenizer_manager import (
    TokenizerManager,
    _request_in_dump_sample,
)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_RIDS = [f"rid-{i}" for i in range(1000)]


class TestRequestInDumpSample(CustomTestCase):
    """Derived properties of the hash-based sampling decision."""

    def test_deterministic_per_rid(self):
        """The same rid must always get the same decision (no randomness)."""
        for rid in _RIDS[:100]:
            first = _request_in_dump_sample(rid, 0.5)
            for _ in range(3):
                self.assertEqual(_request_in_dump_sample(rid, 0.5), first)

    def test_samples_nest_as_fraction_grows(self):
        """A rid kept at fraction f stays kept at every fraction f' > f.

        This is what makes the sample stable when an operator raises the
        fraction on a live server: the existing sample only gains members.
        A non-monotone mapping (e.g. seeding a RNG with the fraction) would
        silently reshuffle it.
        """
        for rid in _RIDS:
            if _request_in_dump_sample(rid, 0.2):
                self.assertTrue(_request_in_dump_sample(rid, 0.6))
                self.assertTrue(_request_in_dump_sample(rid, 1.0))

    def test_kept_count_tracks_fraction(self):
        """Over many rids the kept share must approximate the fraction.

        Guards against the hash-to-[0, 1) mapping degenerating into
        always-keep or always-drop (e.g. a byte-order or scaling bug); the
        rid set is fixed and the hash is deterministic, so this does not
        flake.
        """
        kept = sum(_request_in_dump_sample(rid, 0.5) for rid in _RIDS)
        self.assertGreater(kept, 350)
        self.assertLess(kept, 650)

    def test_fraction_one_keeps_everything(self):
        """1.0 (the default) must keep every request, preserving the
        pre-sampling dump behavior exactly."""
        for rid in _RIDS:
            self.assertTrue(_request_in_dump_sample(rid, 1.0))


def _make_tokenizer_manager() -> TokenizerManager:
    """Create a TokenizerManager with mocked dependencies, bypassing __init__."""
    tm = TokenizerManager.__new__(TokenizerManager)
    tm.request_logger = MagicMock()
    tm.dump_requests_folder = "/tmp/dump"
    tm.dump_requests_threshold = 1000
    tm.dump_requests_sample_fraction = 1.0
    tm.dump_requests_exclude_meta_keys = []
    tm.dump_request_list = []
    tm.crash_dump_folder = ""
    return tm


def _make_req_state(rid: str) -> Mock:
    state = Mock()
    state.obj.rid = rid
    state.time_stats.created_time = 0.0
    state.time_stats.finished_time = 1.0
    return state


class TestConfigureDumpSampleFraction(CustomTestCase):
    """Validation and application of the /configure_logging field."""

    def test_valid_fraction_is_applied(self):
        tm = _make_tokenizer_manager()
        tm.configure_logging(ConfigureLoggingReq(dump_requests_sample_fraction=0.25))
        self.assertEqual(tm.dump_requests_sample_fraction, 0.25)

    def test_unset_fraction_keeps_current_value(self):
        tm = _make_tokenizer_manager()
        tm.dump_requests_sample_fraction = 0.25
        tm.configure_logging(ConfigureLoggingReq(dump_requests_threshold=10))
        self.assertEqual(tm.dump_requests_sample_fraction, 0.25)

    def test_out_of_range_fraction_rejected_and_not_applied(self):
        """0 and out-of-range values must raise and leave the setting
        untouched — 'dump nothing' is expressed by not setting a dump
        folder, not by a 0 fraction silently discarding every request."""
        tm = _make_tokenizer_manager()
        for bad in (0.0, -0.1, 1.5):
            with self.assertRaises(ValueError):
                tm.configure_logging(
                    ConfigureLoggingReq(dump_requests_sample_fraction=bad)
                )
            self.assertEqual(tm.dump_requests_sample_fraction, 1.0)


class TestDumpRequestsSamplingGate(CustomTestCase):
    """The gate must run before the request is buffered."""

    def test_sampled_out_request_is_not_buffered(self):
        tm = _make_tokenizer_manager()
        tm.dump_requests_sample_fraction = 0.5
        kept_rid = next(r for r in _RIDS if _request_in_dump_sample(r, 0.5))
        dropped_rid = next(r for r in _RIDS if not _request_in_dump_sample(r, 0.5))

        tm.dump_requests(_make_req_state(dropped_rid), {"text": "x", "meta_info": {}})
        self.assertEqual(len(tm.dump_request_list), 0)

        tm.dump_requests(_make_req_state(kept_rid), {"text": "x", "meta_info": {}})
        self.assertEqual(len(tm.dump_request_list), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
