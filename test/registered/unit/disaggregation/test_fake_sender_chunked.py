import unittest

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.fake.conn import FakeKVSender
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _make_sender(num_pages: int) -> FakeKVSender:
    sender = FakeKVSender(
        mgr=None,
        bootstrap_addr="",
        bootstrap_room=0,
        dest_tp_ranks=[0],
        pp_rank=0,
    )
    sender.init(num_pages, aux_index=0)
    return sender


class TestFakeSenderChunked(CustomTestCase):
    def test_poll_before_init_is_waiting(self):
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
        sender = _make_sender(num_pages=10)

        sender.send([0, 1, 2, 3])
        self.assertEqual(sender.poll(), KVPoll.WaitingForInput)

        sender.send([4, 5, 6, 7])
        self.assertEqual(sender.poll(), KVPoll.WaitingForInput)

        sender.send([8, 9])
        self.assertEqual(sender.poll(), KVPoll.Success)

    def test_zero_page_request_concludes_immediately(self):
        sender = _make_sender(num_pages=0)
        self.assertEqual(sender.poll(), KVPoll.Success)

    def test_skipped_empty_chunk_does_not_block_conclusion(self):
        sender = _make_sender(num_pages=6)
        sender.send([0, 1, 2, 3])
        self.assertEqual(sender.poll(), KVPoll.WaitingForInput)

        sender.send([4, 5])
        self.assertEqual(sender.poll(), KVPoll.Success)


if __name__ == "__main__":
    unittest.main()
