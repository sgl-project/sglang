import os
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from sglang.srt.disaggregation.mooncake.conn import (
    MooncakeKVManager,
    MooncakeKVSender,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestMooncakeTransferControl(CustomTestCase):
    def test_distinct_session_groups_get_distinct_transfer_queues(self):
        manager = MooncakeKVManager.__new__(MooncakeKVManager)
        manager.transfer_queues = [object() for _ in range(4)]
        manager._session_group_queue_shards = {}
        manager._session_group_queue_lock = threading.Lock()

        groups = [(f"host:{port}",) for port in range(10000, 10004)]
        shards = [manager._get_transfer_queue_shard(group) for group in groups]

        self.assertEqual(shards, [0, 1, 2, 3])
        self.assertEqual(manager._session_group_queue_shards[groups[0]], 0)

    def test_sender_splits_large_kv_work_item_and_preserves_final_state(self):
        sender = MooncakeKVSender.__new__(MooncakeKVSender)
        sender.kv_mgr = SimpleNamespace(add_transfer_request=Mock())
        sender.bootstrap_room = 42
        sender.aux_index = 7
        sender.trace_ctx = Mock()
        sender.trace_ctx.copy_for_thread.return_value = "trace"
        sender._record_transfer_indices = Mock()
        kv_indices = np.arange(10, dtype=np.int32)
        sender._prepare_send_indices = Mock(
            return_value=(kv_indices, slice(20, 30), True, False, 100)
        )
        state_indices = [[3, 4]]

        with patch.dict(
            os.environ,
            {"SGLANG_DISAGGREGATION_KV_TRANSFER_CHUNK_SIZE": "4"},
        ):
            sender.send(kv_indices, state_indices, token_position_offset=100)

        calls = sender.kv_mgr.add_transfer_request.call_args_list
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[0].args[2], slice(20, 24))
        self.assertEqual(calls[1].args[2], slice(24, 28))
        self.assertEqual(calls[2].args[2], slice(28, 30))
        self.assertEqual([call.args[3] for call in calls], [False, False, True])
        self.assertEqual(
            [call.kwargs["token_position_offset"] for call in calls],
            [100, 104, 108],
        )
        self.assertNotIn("state_indices", calls[0].kwargs)
        self.assertEqual(calls[2].kwargs["state_indices"], state_indices)
        self.assertEqual(calls[2].kwargs["aux_index"], 7)


if __name__ == "__main__":
    unittest.main()
