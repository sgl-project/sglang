"""Unit tests for NIXL completion notification parsing."""

import unittest
from collections import defaultdict

from sglang.srt.disaggregation.nixl.conn import NixlKVManager, TransferStatus
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class FakeAgent:
    def __init__(self, messages):
        self.messages = messages

    def get_new_notifs(self):
        return {"peer": [msg.encode("ascii") for msg in self.messages]}


class TestNixlNotifications(CustomTestCase):
    def _make_manager(self, messages, required=None):
        mgr = object.__new__(NixlKVManager)
        mgr.agent = FakeAgent(messages)
        mgr.transfer_statuses = defaultdict(TransferStatus)
        mgr.required_prefill_response_num_table = required or {}
        mgr.enable_staging = False
        mgr._staging_handler = None
        mgr._chunk_writer_counts = defaultdict(lambda: defaultdict(list))
        return mgr

    def test_kv_last_notification_sets_expected_count(self):
        mgr = self._make_manager(["5_kv_2_1_0"])

        mgr.update_transfer_status()

        status = mgr.transfer_statuses[5]
        self.assertEqual(status.received_kvs_per_pp[0], {2})
        self.assertEqual(status.expected_kvs_per_pp[0], 3)
        self.assertEqual(status.num_pp_ranks_expected, 1)

    def test_staging_notification_preserves_agent_name_with_underscores(self):
        mgr = self._make_manager(["5_stg_0_1_0_2_4_8_agent_with_underscores"])
        calls = []
        mgr._handle_staging_chunk_arrived = lambda *args: calls.append(args)

        mgr.update_transfer_status()

        self.assertEqual(calls, [(5, 2, 4, 8, "agent_with_underscores")])
        status = mgr.transfer_statuses[5]
        self.assertEqual(status.received_kvs_per_pp[0], {0})
        self.assertEqual(status.expected_kvs_per_pp[0], 1)

    def test_aux_nokv_marks_zero_expected_chunks_for_pp_rank(self):
        mgr = self._make_manager(["6_aux_nokv_3"], required={6: 4})

        mgr.update_transfer_status()

        status = mgr.transfer_statuses[6]
        self.assertTrue(status.received_aux)
        self.assertEqual(status.expected_kvs_per_pp[3], 0)
        self.assertEqual(status.num_pp_ranks_expected, 4)

    def test_state_notification_marks_pp_rank(self):
        mgr = self._make_manager(["7_state_2"])

        mgr.update_transfer_status()

        self.assertEqual(mgr.transfer_statuses[7].received_state_per_pp, {2})

    def test_aux_nokv_allows_full_hit_completion(self):
        mgr = self._make_manager(["8_aux_nokv_0"], required={8: 1})

        mgr.update_transfer_status()

        self.assertTrue(mgr.transfer_statuses[8].is_done())


if __name__ == "__main__":
    unittest.main()
