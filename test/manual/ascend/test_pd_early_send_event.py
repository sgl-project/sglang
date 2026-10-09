"""Regression checks for early-send dependencies in the Ascend/Mooncake path."""

import threading
import unittest
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.utils import FastQueue
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager, MooncakeKVSender
from sglang.srt.disaggregation.utils import DisaggregationMode


class TestEarlySendEvent(unittest.TestCase):
    def make_manager(self):
        manager = Mock()
        manager.disaggregation_mode = DisaggregationMode.PREFILL
        manager.request_status = {1: KVPoll.WaitingForInput}
        manager.check_status.return_value = KVPoll.WaitingForInput
        manager.enable_trace = False
        manager.enable_staging = False
        manager.enable_deferred_decode_kv_release = False
        manager._staging_outstanding = defaultdict(int)
        manager.failed_sessions = set()
        manager.session_lock = threading.Lock()
        manager.attn_tp_size = 2
        manager.is_mla_backend = False
        manager.is_hybrid_mla_backend = False
        manager.kv_args = SimpleNamespace(kv_data_ptrs=[1])
        manager._get_dsa_cache_transfer_skip_flags.return_value = (False, False)
        req = SimpleNamespace(
            is_dummy=False, mooncake_session_id="127.0.0.1:1",
            dst_kv_indices=np.array([7], dtype=np.int32),
            dst_device_kv_indices=None,
        )
        registration = SimpleNamespace(
            requires_dcp_relayout=False, dst_kv_ptrs=[2], dst_kv_layer_ids=[0],
            dst_kv_item_len=128, dst_attn_tp_size=2,
        )
        manager.transfer_infos = {1: {"127.0.0.1:1": req}}
        manager.decode_kv_args_table = {"127.0.0.1:1": registration}
        manager.transfer_queues = [FastQueue()]
        manager.send_kvcache.return_value = 0
        return manager

    def enqueue_from_sender(self, manager, event, *, skip=False, final=False):
        sender = Mock()
        sender._early_send_wait_event = event
        sender.bootstrap_room = 1
        sender.kv_mgr = manager
        sender._prepare_send_indices.return_value = (
            np.array([3], dtype=np.int32), slice(0, 1), final, skip,
        )
        manager.add_transfer_request.side_effect = lambda *args, **kwargs: (
            MooncakeKVManager.add_transfer_request(manager, *args, **kwargs)
        )
        MooncakeKVSender.send(sender, np.array([3], dtype=np.int32))
        self.assertIsNone(sender._early_send_wait_event)
        return sender

    def test_sender_enqueues_without_waiting_and_worker_waits_before_read(self):
        manager = self.make_manager()
        entered, release = threading.Event(), threading.Event()
        event = Mock()

        def wait():
            entered.set()
            if not release.wait(5):
                raise TimeoutError("Test did not release the source-write event")

        event.synchronize.side_effect = wait
        self.enqueue_from_sender(manager, event)
        event.synchronize.assert_not_called()
        manager.transfer_queues[0].put(None)
        failures = []

        def run():
            try:
                MooncakeKVManager.transfer_worker(manager, manager.transfer_queues[0], None)
            except Exception as exc:
                failures.append(exc)

        worker = threading.Thread(target=run)
        worker.start()
        try:
            self.assertTrue(entered.wait(5))
            manager.send_kvcache.assert_not_called()
        finally:
            release.set()
            worker.join(5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(failures, [])
        event.synchronize.assert_called_once()
        manager.send_kvcache.assert_called_once()

    def test_skipped_send_discards_event(self):
        manager = self.make_manager()
        event = Mock()
        self.enqueue_from_sender(manager, event, skip=True)
        manager.add_transfer_request.assert_not_called()
        event.synchronize.assert_not_called()

    def test_failed_request_is_skipped_without_waiting(self):
        manager = self.make_manager()
        event = Mock()
        self.enqueue_from_sender(manager, event)
        manager.check_status.return_value = KVPoll.Failed
        manager.transfer_queues[0].put(None)
        MooncakeKVManager.transfer_worker(manager, manager.transfer_queues[0], None)
        event.synchronize.assert_not_called()
        manager.send_kvcache.assert_not_called()

    def test_final_send_keeps_event_on_task(self):
        manager = self.make_manager()
        event = Mock()
        self.enqueue_from_sender(manager, event, final=True)
        task = manager.transfer_queues[0].get()
        self.assertIs(task.wait_event, event)
        self.assertTrue(task.is_last_chunk)
        event.synchronize.assert_not_called()


if __name__ == "__main__":
    unittest.main()
