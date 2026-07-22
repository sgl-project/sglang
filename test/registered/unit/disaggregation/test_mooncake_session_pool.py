import os
import threading
import unittest
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from sglang.srt.disaggregation.mooncake.conn import (
    MooncakeDecodeSessionPool,
    MooncakeKVManager,
    MooncakeKVReceiver,
    MooncakeKVSender,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _FakeEngine:
    def __init__(self, session_id):
        self.session_id = session_id
        self.registered = []
        self.deregistered = []

    def get_session_id(self):
        return self.session_id

    def batch_register(self, ptrs, lens):
        self.registered.append((ptrs, lens))
        return 0

    def batch_deregister(self, ptrs):
        self.deregistered.append(ptrs)
        return 0


class TestMooncakeDecodeSessionPool(CustomTestCase):
    def test_spreads_rooms_and_reuses_assignment(self):
        engines = [_FakeEngine(f"session-{i}") for i in range(4)]
        pool = MooncakeDecodeSessionPool(engines, rooms_per_session=2)

        assignments = [pool.acquire(room)[0] for room in range(8)]

        self.assertEqual(assignments, [0, 1, 2, 3, 0, 1, 2, 3])
        self.assertEqual(pool.active_room_counts(), (2, 2, 2, 2))
        self.assertEqual(pool.acquire(0), (0, "session-0"))
        self.assertEqual(pool.active_room_counts(), (2, 2, 2, 2))

    def test_release_is_idempotent_and_makes_session_least_loaded(self):
        engines = [_FakeEngine(f"session-{i}") for i in range(2)]
        pool = MooncakeDecodeSessionPool(engines, rooms_per_session=1)
        pool.acquire(10)
        pool.acquire(11)

        pool.release(10)
        pool.release(10)

        self.assertEqual(pool.active_room_counts(), (0, 1))
        self.assertEqual(pool.acquire(12)[0], 0)

    def test_registers_and_deregisters_buffers_on_every_engine(self):
        manager = MooncakeKVManager.__new__(MooncakeKVManager)
        manager.engines = [_FakeEngine("session-0"), _FakeEngine("session-1")]
        manager._session_registration_lock = threading.Lock()
        manager._session_route_registrations = {}
        manager.kv_args = SimpleNamespace(
            kv_data_ptrs=[100],
            kv_data_lens=[10],
            aux_data_ptrs=[200],
            aux_data_lens=[20],
            state_data_ptrs=[[300]],
            state_data_lens=[[30]],
        )

        manager.register_buffer_to_engine()
        manager.deregister_buffer_to_engine()

        for engine in manager.engines:
            self.assertEqual(
                engine.registered,
                [([100], [10]), ([200], [20]), ([300], [30])],
            )
            self.assertEqual(engine.deregistered, [[100], [200], [300]])

    def test_route_registration_is_per_session_and_route_object(self):
        manager = MooncakeKVManager.__new__(MooncakeKVManager)
        manager._session_registration_lock = threading.Lock()
        manager._session_route_registrations = {}
        route = {}

        self.assertTrue(manager.claim_session_route_registration(route, "session-0"))
        self.assertFalse(manager.claim_session_route_registration(route, "session-0"))
        self.assertTrue(manager.claim_session_route_registration(route, "session-1"))
        self.assertEqual(route, {}, "Registration tracking must not mutate route data")
        self.assertTrue(
            manager.claim_session_route_registration({}, "session-0"),
            "A recreated route must register the session again",
        )

    def test_distinct_session_groups_get_distinct_transfer_queues(self):
        manager = MooncakeKVManager.__new__(MooncakeKVManager)
        manager.transfer_queues = [object() for _ in range(4)]
        manager._session_group_queue_shards = {}
        manager._session_group_queue_lock = threading.Lock()

        groups = [(f"host:{port}",) for port in range(10000, 10004)]
        shards = [manager._get_transfer_queue_shard(group) for group in groups]

        self.assertEqual(shards, [0, 1, 2, 3])
        self.assertEqual(manager._session_group_queue_shards[groups[0]], 0)

    def test_receiver_releases_session_once(self):
        manager = SimpleNamespace(
            acquire_decode_session=Mock(return_value=(1, "session-1")),
            release_decode_session=Mock(),
            addr_to_rooms_tracker=defaultdict(set),
            update_status=Mock(),
            request_status={},
            required_prefill_response_num_table={},
            prefill_response_tracker={},
        )
        receiver = MooncakeKVReceiver(manager, "prefill:8998", bootstrap_room=42)

        receiver.clear()
        receiver.clear()

        manager.acquire_decode_session.assert_called_once_with(42)
        manager.release_decode_session.assert_called_once_with(42)

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
