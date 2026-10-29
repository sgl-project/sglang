import queue
import unittest
from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.base import KVPoll  # noqa: E402
from sglang.srt.disaggregation.common.conn import CommonKVSender  # noqa: E402
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager  # noqa: E402
from sglang.srt.disaggregation.utils import DisaggregationMode  # noqa: E402
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.runtime_context import (  # noqa: E402
    get_context,
    publish,
    reset_context,
)
from sglang.srt.server_args import ServerArgs  # noqa: E402

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _sender(room=42):
    manager = object.__new__(MooncakeKVManager)
    manager.disaggregation_mode = DisaggregationMode.PREFILL
    manager.request_status = {}
    manager.transfer_infos = {room: {"127.0.0.1:1234": SimpleNamespace(dst_port=1234)}}
    manager.is_dummy_cp_rank = False
    manager.enable_all_cp_ranks_for_transfer = False
    manager.attn_cp_size = 1
    manager.attn_cp_rank = 0
    manager.enable_staging = False
    manager._staging_outstanding = {}
    manager._deferred_ack_targets = {}
    manager._deferred_ack_poisoned_rooms = set()
    manager._num_shards = 1
    manager.transfer_queues = [queue.Queue()]
    manager._transfer_queues = manager.transfer_queues
    with get_context().override_server_args(dp_size=1, enable_trace=False):
        sender = CommonKVSender(manager, "unused", room)
    return sender, manager


class _Req:
    def __init__(self, rid):
        self.rid = rid
        self.kv = SimpleNamespace(holds_kv=True, holds_mamba=False)
        self.to_finish = None
        self._finished = False
        self.discard_output_reason = None
        self.return_logprob = False
        self.output_ids = []
        self.weight_version_events = []
        self.time_stats = Mock()

    def finished(self):
        return self._finished


def _scheduler(req):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.enable_continuous_input_polling = False
    scheduler.result_queue = deque()
    scheduler.chunked_req = req
    scheduler._pending_chunked_abort_req = req
    scheduler.waiting_queue = []
    scheduler.dllm_config = None
    scheduler.grammar_manager = Mock()
    scheduler.disaggregation_mode = DisaggregationMode.PREFILL
    scheduler.enable_hicache_storage = False
    scheduler.mm_receiver = None
    scheduler.rust_server = None
    scheduler.running_batch = SimpleNamespace(reqs=[])
    scheduler.last_batch = None
    scheduler.disagg_prefill_inflight_queue = []
    scheduler.clear_pending_chunk_send = Mock()
    scheduler._release_aborted_request = Mock()
    scheduler.tree_cache = None
    scheduler.ipc_channels = Mock()
    return scheduler


class TestPrefillSourceRetirement(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="tokenizer")

    def test_failure_waits_for_outstanding_staging_chunk(self):
        sender, manager = _sender()
        manager.request_status[42] = KVPoll.Failed
        manager._staging_outstanding[42] = 1
        self.assertEqual(sender.poll(), KVPoll.Transferring)
        self.assertIsNone(sender.conclude_state)
        manager._staging_outstanding[42] = 0
        self.assertEqual(sender.poll(), KVPoll.Failed)
        self.assertEqual(sender.conclude_state, KVPoll.Failed)

    def test_prefill_abort_keeps_source_until_transfer_finishes(self):
        req = _Req("transfer_in_flight")
        req.disagg_kv_sender = Mock()
        req.disagg_kv_sender.poll.return_value = KVPoll.Transferring
        scheduler = _scheduler(req)

        with patch("sglang.srt.managers.scheduler.release_kv_cache") as release:
            scheduler.process_pending_chunked_abort()

        req.disagg_kv_sender.abort.assert_called_once_with()
        self.assertEqual(scheduler.disagg_prefill_inflight_queue, [req])
        scheduler._release_aborted_request.assert_not_called()
        release.assert_not_called()


if __name__ == "__main__":
    unittest.main()
