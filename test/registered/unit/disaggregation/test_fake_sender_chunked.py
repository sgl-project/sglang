"""Unit tests for FakeKVSender — chunked-prefill KV transfer must not conclude
before every page has been sent, otherwise the prefill scheduler pops the
request from the inflight queue and its tail KV is never released (pool leak)."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

import unittest

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.fake.conn import FakeKVSender
from sglang.test.test_utils import CustomTestCase


def _make_sender(num_pages: int) -> FakeKVSender:
    sender = FakeKVSender(
        mgr=None,
        bootstrap_addr="",
        bootstrap_room=0,
        dest_tp_ranks=[0],
        pp_rank=0,
    )
    # init() declares how many pages the request will transfer in total.
    sender.init(num_pages, aux_index=0)
    return sender


class TestFakeSenderChunked(CustomTestCase):
    def test_poll_before_init_is_waiting(self):
        # pop_bootstrapped polls the sender BEFORE finalize_bootstrap calls
        # init(). poll() must be WaitingForInput here; returning Success (state
        # 4) crashes pop_bootstrapped with "Unexpected poll state 4".
        sender = FakeKVSender(
            mgr=None,
            bootstrap_addr="",
            bootstrap_room=0,
            dest_tp_ranks=[0],
            pp_rank=0,
        )
        self.assertEqual(sender.poll(), KVPoll.WaitingForInput)

    def test_single_chunk_concludes_after_send(self):
        sender = _make_sender(num_pages=4)
        self.assertEqual(sender.poll(), KVPoll.WaitingForInput)
        sender.send([0, 1, 2, 3])
        self.assertEqual(sender.poll(), KVPoll.Success)

    def test_chunked_request_only_concludes_on_last_chunk(self):
        # A request split into 3 chunks (chunked prefill). Success must not be
        # reported until every page has been sent, otherwise the scheduler pops
        # the req from the inflight queue and never releases the tail KV.
        sender = _make_sender(num_pages=10)

        sender.send([0, 1, 2, 3])  # chunk 1
        self.assertEqual(
            sender.poll(),
            KVPoll.WaitingForInput,
            "fake sender concluded before all chunks were sent",
        )

        sender.send([4, 5, 6, 7])  # chunk 2
        self.assertEqual(sender.poll(), KVPoll.WaitingForInput)

        sender.send([8, 9])  # last chunk
        self.assertEqual(sender.poll(), KVPoll.Success)

    def test_zero_page_request_concludes_immediately(self):
        # Fully cache-hit request: num_kv_indices_to_send == 0 -> num_pages 0.
        # No send() ever arrives, so poll must conclude rather than hang.
        sender = _make_sender(num_pages=0)
        self.assertEqual(sender.poll(), KVPoll.Success)

    def test_skipped_empty_chunk_does_not_block_conclusion(self):
        # should_send_kv_chunk drops empty chunks (num_pages == 0 and not
        # last_chunk), so those pages never reach send(). Conclusion must depend
        # only on the pages actually emitted vs num_pages (computed from
        # num_kv_indices_to_send), never the raw chunk count.
        sender = _make_sender(num_pages=6)
        sender.send([0, 1, 2, 3])  # chunk 1
        self.assertEqual(sender.poll(), KVPoll.WaitingForInput)
        # an empty chunk was skipped upstream -> no send() call
        sender.send([4, 5])  # last non-empty chunk completes the transfer
        self.assertEqual(sender.poll(), KVPoll.Success)


if __name__ == "__main__":
    unittest.main()
