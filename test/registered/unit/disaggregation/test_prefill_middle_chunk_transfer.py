import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.utils import TransferKVChunk
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.srt.disaggregation.nixl.conn import NixlKVManager
from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def make_chunk(wait_event):
    return TransferKVChunk(
        room=7,
        prefill_kv_indices=np.array([1], dtype=np.int32),
        index_slice=slice(0, 1),
        is_last_chunk=False,
        prefill_aux_index=None,
        state_indices=None,
        wait_event=wait_event,
    )


class TestPrefillMiddleChunkTransfer(unittest.TestCase):
    def test_middle_chunk_is_enqueued_with_producer_event(self):
        wait_event = object()
        req = SimpleNamespace(
            rid="middle",
            pending_bootstrap=False,
            to_finish=None,
            finished_reason=None,
            metadata_buffer_index=3,
            extend_range=SimpleNamespace(end=8192),
            origin_input_ids=range(20_000),
        )
        scheduler = SimpleNamespace(send_kv_chunk=MagicMock())
        batch = SimpleNamespace(
            reqs=[req],
            chunked_req=req,
            contains_last_prefill_chunk=False,
        )
        result = SimpleNamespace(copy_done=wait_event)

        sent = SchedulerDisaggregationPrefillMixin._enqueue_middle_chunk_transfer(
            scheduler, batch, result
        )

        self.assertTrue(sent)
        scheduler.send_kv_chunk.assert_called_once_with(
            req,
            last_chunk=False,
            end_idx=8192,
            wait_event=wait_event,
        )

    def test_final_chunk_stays_on_result_processing_path(self):
        scheduler = SimpleNamespace(send_kv_chunk=MagicMock())
        sent = SchedulerDisaggregationPrefillMixin._enqueue_middle_chunk_transfer(
            scheduler,
            SimpleNamespace(contains_last_prefill_chunk=True),
            SimpleNamespace(copy_done=object()),
        )

        self.assertFalse(sent)
        scheduler.send_kv_chunk.assert_not_called()

    def test_mooncake_worker_waits_before_accessing_source(self):
        wait_event = MagicMock()
        wait_event.synchronize.side_effect = KeyboardInterrupt
        manager = object.__new__(MooncakeKVManager)
        manager.enable_trace = False
        manager.request_status = {7: KVPoll.WaitingForInput}
        manager._staging_outstanding = {7: 0}
        manager.check_status = MagicMock(return_value=KVPoll.WaitingForInput)
        queue = SimpleNamespace(get=MagicMock(return_value=make_chunk(wait_event)))

        with self.assertRaises(KeyboardInterrupt):
            manager.transfer_worker(queue, MagicMock())

        wait_event.synchronize.assert_called_once_with()
        self.assertEqual(manager._staging_outstanding[7], 1)

    def test_nixl_worker_waits_before_accessing_source(self):
        wait_event = MagicMock()
        wait_event.synchronize.side_effect = KeyboardInterrupt
        manager = object.__new__(NixlKVManager)
        manager._staging_outstanding = {7: 0}
        manager.check_status = MagicMock(return_value=KVPoll.WaitingForInput)
        queue = SimpleNamespace(get=MagicMock(return_value=make_chunk(wait_event)))

        with self.assertRaises(KeyboardInterrupt):
            manager.transfer_worker(queue)

        wait_event.synchronize.assert_called_once_with()
        self.assertEqual(manager._staging_outstanding[7], 1)


if __name__ == "__main__":
    unittest.main()
