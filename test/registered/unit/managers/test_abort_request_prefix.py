"""Unit tests for TokenizerManager.abort_request prefix matching.

With prefix=True, the rid is treated as a prefix: the tokenizer-side
early-return check matches any tracked rid starting with it, and the
AbortReq is forwarded with prefix=True so the scheduler prefix-matches too
(the scheduler always matches via rid.startswith).
"""

import unittest
from unittest.mock import MagicMock, Mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.managers.tokenizer_manager import TokenizerManager

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_tokenizer_manager(rids=(), tokenizer_worker_num=1) -> TokenizerManager:
    """Create a TokenizerManager with mocked dependencies, bypassing __init__."""
    tm = TokenizerManager.__new__(TokenizerManager)
    tm.server_args = MagicMock()
    tm.server_args.tokenizer_worker_num = tokenizer_worker_num
    tm.enable_metrics = False
    tm.rid_to_state = {rid: Mock() for rid in rids}
    tm.send_to_scheduler = MagicMock()
    return tm


def _sent_req(tm) -> AbortReq:
    tm.send_to_scheduler.send_pyobj.assert_called_once()
    return tm.send_to_scheduler.send_pyobj.call_args.args[0]


class TestAbortRequestPrefix(CustomTestCase):
    def test_prefix_match_sends_abort(self):
        tm = _make_tokenizer_manager(rids=["job-1-seq-0", "job-1-seq-1", "other"])
        tm.abort_request(rid="job-1", prefix=True)

        req = _sent_req(tm)
        self.assertEqual(req.rid, "job-1")
        self.assertTrue(req.prefix)
        self.assertFalse(req.abort_all)

    def test_prefix_without_match_is_ignored(self):
        tm = _make_tokenizer_manager(rids=["other-1", "other-2"])
        tm.abort_request(rid="job-1", prefix=True)

        tm.send_to_scheduler.send_pyobj.assert_not_called()

    def test_prefix_requires_full_prefix_not_substring(self):
        tm = _make_tokenizer_manager(rids=["seq-job-1"])
        tm.abort_request(rid="job-1", prefix=True)

        tm.send_to_scheduler.send_pyobj.assert_not_called()

    def test_exact_match_still_works_without_prefix(self):
        tm = _make_tokenizer_manager(rids=["job-1"])
        tm.abort_request(rid="job-1")

        req = _sent_req(tm)
        self.assertEqual(req.rid, "job-1")
        self.assertFalse(req.prefix)

    def test_exact_mode_does_not_prefix_match(self):
        # rid is only a prefix of a tracked request; exact mode must ignore it.
        tm = _make_tokenizer_manager(rids=["job-1-seq-0"])
        tm.abort_request(rid="job-1")

        tm.send_to_scheduler.send_pyobj.assert_not_called()

    def test_empty_rid_is_ignored(self):
        # An empty rid would prefix-match every request on the scheduler.
        tm = _make_tokenizer_manager(rids=["job-1"])
        tm.abort_request(rid="", prefix=True)

        tm.send_to_scheduler.send_pyobj.assert_not_called()

    def test_multi_tokenizer_worker_skips_local_check(self):
        # With >1 tokenizer workers, rid_to_state is not authoritative; the
        # abort must be forwarded even if this worker tracks no matching rid.
        tm = _make_tokenizer_manager(rids=[], tokenizer_worker_num=2)
        tm.abort_request(rid="job-1", prefix=True)

        req = _sent_req(tm)
        self.assertEqual(req.rid, "job-1")
        self.assertTrue(req.prefix)


if __name__ == "__main__":
    unittest.main(verbosity=2)
