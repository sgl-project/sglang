import concurrent.futures
import unittest
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import numpy as np

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.utils import TransferKVChunk
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestMooncakeTransferBatching(unittest.TestCase):
    @staticmethod
    def _make_manager(
        side_effect=None, enable_custom_mem_pool=False, max_batch_indices=0
    ):
        engine = MagicMock()
        if side_effect is None:
            engine.batch_transfer_sync.return_value = 0
        else:
            engine.batch_transfer_sync.side_effect = side_effect
        manager = SimpleNamespace(
            engine=engine,
            is_mla_backend=True,
            is_hybrid_mla_backend=False,
            pp_size=1,
            enable_custom_mem_pool=enable_custom_mem_pool,
            enable_deferred_decode_kv_release=False,
            max_transfer_batch_indices=max_batch_indices,
            get_mla_kv_ptrs_with_pp=MagicMock(
                return_value=([1000, 2000], [5000, 6000], 2)
            ),
        )
        manager._transfer_data = lambda session, blocks: (
            MooncakeKVManager._transfer_data(manager, session, blocks)
        )
        manager._await_transfer_futures = lambda futures: (
            MooncakeKVManager._await_transfer_futures(manager, futures)
        )
        return manager

    @staticmethod
    def _send(
        manager,
        dst_device_data_indices=None,
        dst_device_data_ptrs=None,
    ):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            return MooncakeKVManager._send_kvcache_generic(
                manager,
                mooncake_session_id="session",
                src_data_ptrs=[1000, 2000],
                dst_data_ptrs=[5000, 6000],
                item_lens=[10, 20],
                prefill_data_indices=np.array([0, 1, 2, 3, 4], dtype=np.int32),
                dst_data_indices=np.array([10, 11, 12, 13, 14], dtype=np.int32),
                executor=executor,
                dst_device_data_indices=dst_device_data_indices,
                dst_device_data_ptrs=dst_device_data_ptrs,
            )

    def test_slices_index_arrays_before_forming_transfer_ranges(self):
        manager = self._make_manager(max_batch_indices=2)
        ret = self._send(manager)

        self.assertEqual(ret, 0)
        self.assertEqual(
            manager.engine.batch_transfer_sync.call_args_list,
            [
                call("session", [1000, 2000], [5100, 6200], [20, 40]),
                call("session", [1020, 2040], [5120, 6240], [20, 40]),
                call("session", [1040, 2080], [5140, 6280], [10, 20]),
            ],
        )

    def test_preserves_legacy_single_batch_path_for_short_transfers(self):
        for max_batch_indices in (0, 5, 6):
            with self.subTest(max_batch_indices=max_batch_indices):
                manager = self._make_manager(max_batch_indices=max_batch_indices)
                ret = self._send(manager)

                self.assertEqual(ret, 0)
                manager.engine.batch_transfer_sync.assert_called_once_with(
                    "session",
                    [1000, 2000],
                    [5100, 6200],
                    [50, 100],
                )

    def test_stops_after_first_failed_index_batch(self):
        manager = self._make_manager(side_effect=[0, -1], max_batch_indices=2)
        ret = self._send(manager)

        self.assertEqual(ret, -1)
        self.assertEqual(manager.engine.batch_transfer_sync.call_count, 2)

    def test_uses_device_page_indices_in_batched_path(self):
        manager = self._make_manager(max_batch_indices=2)
        ret = self._send(
            manager,
            dst_device_data_indices=np.array([20, 21, 22, 23, 24], dtype=np.int32),
            dst_device_data_ptrs={6000},
        )

        self.assertEqual(ret, 0)
        self.assertEqual(
            manager.engine.batch_transfer_sync.call_args_list,
            [
                call("session", [1000, 2000], [5100, 6400], [20, 40]),
                call("session", [1020, 2040], [5120, 6440], [20, 40]),
                call("session", [1040, 2080], [5140, 6480], [10, 20]),
            ],
        )

    def test_preserves_one_transfer_per_layer_for_custom_mem_pool(self):
        manager = self._make_manager(enable_custom_mem_pool=True, max_batch_indices=2)
        ret = self._send(manager)

        self.assertEqual(ret, 0)
        self.assertEqual(manager.engine.batch_transfer_sync.call_count, 2)
        manager.engine.batch_transfer_sync.assert_has_calls(
            [
                call("session", [1000], [5100], [50]),
                call("session", [2000], [6200], [100]),
            ],
            any_order=True,
        )


class TestMooncakeEarlySendWaitEvent(unittest.TestCase):
    ROOM = 41

    def _make_chunk(self, wait_event):
        return TransferKVChunk(
            room=self.ROOM,
            prefill_kv_indices=np.array([1], dtype=np.int32),
            index_slice=slice(0, 1),
            is_last_chunk=False,
            prefill_aux_index=None,
            state_indices=None,
            wait_event=wait_event,
        )

    @staticmethod
    def _make_manager(enable_staging):
        manager = object.__new__(MooncakeKVManager)
        manager.enable_trace = False
        manager.enable_staging = enable_staging
        manager.enable_deferred_decode_kv_release = False
        manager._staging_outstanding = defaultdict(int)
        manager.request_status = {
            TestMooncakeEarlySendWaitEvent.ROOM: KVPoll.WaitingForInput
        }
        manager.transfer_infos = {TestMooncakeEarlySendWaitEvent.ROOM: {}}
        manager.check_status = lambda room: KVPoll.WaitingForInput
        # Reached past the wait on the non-staged path, which has no earlier probe.
        manager.attn_tp_rank = 0
        manager.attn_cp_rank = 0
        manager.attn_cp_size = 1
        manager.pp_rank = 0
        manager.pp_size = 1
        return manager

    def test_worker_synchronizes_the_early_send_event_before_the_staging_gather(self):
        """The wait precedes the staging strategy build, hence the gather.

        Early-send issues the KV read before the step's forward is enqueued, so
        the prior step's prefill forward may still be writing these pages. The
        staged path gathers them into the staging buffer before posting the RDMA,
        so a wait covering only the post ships partially written KV as success.
        """
        manager = self._make_manager(enable_staging=True)
        order = []

        def stop_at_strategy(_staging_buffer):
            order.append("gather")
            raise SystemExit

        manager._try_create_staging_strategy = stop_at_strategy
        chunk = self._make_chunk(
            SimpleNamespace(synchronize=lambda: order.append("wait"))
        )
        # Bounded: a body that reaches neither probe must end the worker, not spin.
        queue = SimpleNamespace(get=MagicMock(side_effect=[chunk, SystemExit()]))

        with self.assertRaises(SystemExit):
            manager.transfer_worker(queue, None, staging_buffer=object())

        self.assertEqual(order, ["wait", "gather"])

    def test_worker_waits_on_the_early_send_event_with_staging_disabled(self):
        """The wait is not conditional on the staged path.

        Non-staged is the common deployment, and it reads the pages straight out
        of the device pool, so a wait reachable only when staging is enabled would
        leave that path racing the prior step's prefill writes.
        """
        manager = self._make_manager(enable_staging=False)
        waited = []
        chunk = self._make_chunk(
            SimpleNamespace(synchronize=lambda: waited.append(self.ROOM))
        )
        queue = SimpleNamespace(get=MagicMock(side_effect=[chunk, SystemExit()]))

        with self.assertRaises(SystemExit):
            manager.transfer_worker(queue, None)

        self.assertEqual(waited, [self.ROOM])

    def test_add_transfer_request_forwards_the_wait_event_to_the_queued_chunk(self):
        """The worker only waits on TransferKVChunk.wait_event.

        A backend that accepts the event but does not put it on the chunk drops
        the wait with nothing failing: the transfer still reports success.
        """
        manager = object.__new__(MooncakeKVManager)
        manager.disaggregation_mode = DisaggregationMode.PREFILL
        manager.request_status = {self.ROOM: KVPoll.WaitingForInput}
        manager.check_status = lambda room: KVPoll.WaitingForInput
        manager.transfer_infos = {self.ROOM: {"127.0.0.1:5555": SimpleNamespace()}}
        queue = SimpleNamespace(put=MagicMock())
        manager.transfer_queues = [queue]
        event = object()

        manager.add_transfer_request(
            self.ROOM,
            np.array([1], dtype=np.int32),
            slice(0, 1),
            False,
            wait_event=event,
        )

        self.assertIs(queue.put.call_args.args[0].wait_event, event)


if __name__ == "__main__":
    unittest.main()
