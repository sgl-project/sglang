"""Unit tests for NIXL transfer completion state."""

import unittest

from sglang.srt.disaggregation.nixl.conn import NixlKVSender, TransferStatus
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestNixlTransferStatus(CustomTestCase):
    def test_not_done_until_aux_and_expected_count_arrive(self):
        status = TransferStatus()

        self.assertFalse(status.is_done())

        status.received_aux = True
        self.assertFalse(status.is_done())

        status.num_pp_ranks_expected = 1
        self.assertFalse(status.is_done())

        status.expected_kvs_per_pp[0] = 1
        self.assertFalse(status.is_done())

        status.received_kvs_per_pp[0].add(0)
        self.assertTrue(status.is_done())

    def test_zero_kv_aux_only_completion(self):
        status = TransferStatus()
        status.received_aux = True
        status.num_pp_ranks_expected = 1
        status.expected_kvs_per_pp[0] = 0

        self.assertTrue(status.is_done())

    def test_multi_pp_requires_each_rank_expected_chunks(self):
        status = TransferStatus()
        status.received_aux = True
        status.num_pp_ranks_expected = 2
        status.expected_kvs_per_pp[0] = 1
        status.received_kvs_per_pp[0].add(0)

        self.assertFalse(status.is_done())

        status.expected_kvs_per_pp[1] = 2
        status.received_kvs_per_pp[1].update({0, 1})
        self.assertTrue(status.is_done())

    def test_state_required_completion_waits_for_all_pp_ranks(self):
        status = TransferStatus()
        status.received_aux = True
        status.num_pp_ranks_expected = 2
        status.expected_kvs_per_pp[0] = 0
        status.expected_kvs_per_pp[1] = 0
        status.expects_state = True

        self.assertFalse(status.is_done())

        status.received_state_per_pp.add(0)
        self.assertFalse(status.is_done())

        status.received_state_per_pp.add(1)
        self.assertTrue(status.is_done())


class TestNixlKVSenderChunkPolicy(CustomTestCase):
    def test_last_zero_page_chunk_is_sent_for_aux_only_completion(self):
        sender = object.__new__(NixlKVSender)

        self.assertTrue(sender.should_send_kv_chunk(0, last_chunk=True))
        self.assertFalse(sender.should_send_kv_chunk(0, last_chunk=False))
        self.assertTrue(sender.should_send_kv_chunk(3, last_chunk=False))


if __name__ == "__main__":
    unittest.main()
