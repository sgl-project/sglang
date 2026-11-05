import sys
import threading
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import msgspec
import numpy as np

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.mooncake.kvserve import ACK, KVServeAdapter
from sglang.srt.disaggregation.mooncake.kvserve_transport import ReceiveCredits
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class TestKVServeCompletion(CustomTestCase):
    def test_role_switch_fails_before_allocating_codec_buffers(self):
        manager = SimpleNamespace(
            kv_args=SimpleNamespace(gpu_id=0, state_types=[], num_draft_entries=0),
            server_args=SimpleNamespace(enable_pd_role_switch=True),
            is_mla_backend=False,
            is_hybrid_mla_backend=False,
            attn_tp_size=1,
            attn_cp_size=1,
            dcp_size=1,
            pp_size=1,
            enable_staging=False,
        )
        # The optional codec is not installed in CPU CI. Only satisfy its import:
        # validation must reject role switching before any codec is constructed.
        runtime = ModuleType("kvserve_v1.compression.runtime")
        runtime.CacheSpec = Mock()
        runtime.CompressionConfig = Mock()
        with patch.dict(sys.modules, {runtime.__name__: runtime}):
            with self.assertRaisesRegex(ValueError, "role switching"):
                KVServeAdapter(manager, [], [], 1, [])
        runtime.CacheSpec.assert_not_called()
        runtime.CompressionConfig.assert_not_called()

    def setUp(self):
        super().setUp()
        self.adapter = object.__new__(KVServeAdapter)
        self.adapter.manager = SimpleNamespace(
            record_failure=Mock(), _poison_unsettled_send=Mock()
        )
        self.adapter.lock = threading.Lock()
        self.adapter.submit_lock = threading.Lock()
        self.adapter.epoch = "test"
        self.adapter.page_size = 1
        self.adapter.has_mamba = False
        self.adapter.expected_rooms = {}
        self.adapter.restored_rooms = {}
        self.adapter.expect_room(123, [7, 3], False)

    def test_native_completion_cannot_bypass_restoration(self):
        self.assertEqual(self.adapter.check_success(123, KVPoll.Success), KVPoll.Failed)
        self.adapter.manager.record_failure.assert_called_once()

    def test_k_only_completion_cannot_start_decode(self):
        self.adapter.restored_rooms[123][0].update([3, 7])
        self.assertEqual(self.adapter.check_success(123, KVPoll.Success), KVPoll.Failed)

    def test_same_count_wrong_pages_cannot_start_decode(self):
        self.adapter.restored_rooms[123] = {0: {3, 7}, 1: {3, 8}}
        self.assertEqual(self.adapter.check_success(123, KVPoll.Success), KVPoll.Failed)

    def test_all_registered_k_and_v_pages_allow_native_completion(self):
        self.adapter.restored_rooms[123] = {0: {3, 7}, 1: {3, 7}}
        self.assertEqual(
            self.adapter.check_success(123, KVPoll.Success), KVPoll.Success
        )
        self.adapter.manager.record_failure.assert_not_called()

    def test_receiver_cleanup_retires_compression_bookkeeping(self):
        self.adapter.drop_room(123)
        self.assertFalse(self.adapter.expected_rooms)
        self.assertFalse(self.adapter.restored_rooms)

    def test_late_ack_cannot_release_an_unrelated_credit(self):
        self.adapter.pending_lock = threading.Lock()
        event, result = threading.Event(), {}
        self.adapter.pending = {"current": (event, result, ReceiveCredits(0), 0)}
        self.adapter.handle_ack(
            [
                ACK,
                msgspec.msgpack.encode(
                    [{"epoch": "test", "sequence": "previous", "restore_ms": 1}]
                ),
            ]
        )
        self.assertFalse(event.is_set())
        self.adapter.handle_ack(
            [
                ACK,
                msgspec.msgpack.encode(
                    [{"epoch": "test", "sequence": "current", "restore_ms": 2}]
                ),
            ]
        )
        self.assertTrue(event.is_set())
        self.assertEqual(result["restore_ms"], 2)

    def test_unsettled_transfer_prevents_deregister_and_free(self):
        self.adapter.poisoned = True
        self.adapter.runtime = SimpleNamespace(shutdown=Mock())
        with self.assertRaises(RuntimeError):
            self.adapter.close()
        self.adapter.runtime.shutdown.assert_not_called()

    def test_disabled_peer_is_rejected_instead_of_native_fallback(self):
        self.adapter.runtime = SimpleNamespace(wire_profile="profile")
        self.adapter.registration = lambda: {"slots": [{"capacities": [4096, 128]}]}
        with self.assertRaises(ValueError):
            self.adapter.validate_peer(None)

    def test_peer_profile_and_buffer_capacity_must_match(self):
        self.adapter.runtime = SimpleNamespace(wire_profile="profile")
        self.adapter.registration = lambda: {"slots": [{"capacities": [4096, 128]}]}
        peer = {
            "version": 3,
            "has_mamba": False,
            "page_size": 1,
            "profile": "profile",
            "slots": [{"ptrs": [4096, 8192], "capacities": [4096, 128]}],
        }
        self.adapter.validate_peer(peer)
        with self.assertRaises(ValueError):
            self.adapter.validate_peer({**peer, "profile": "other"})
        with self.assertRaises(ValueError):
            self.adapter.validate_peer(
                {**peer, "slots": [{"ptrs": [4096, 8192], "capacities": [4096, 127]}]}
            )

    def test_matching_peer_allows_independent_slot_counts(self):
        self.adapter.runtime = SimpleNamespace(wire_profile="shared-profile")
        slot = {"ptrs": [4096, 8192], "capacities": [4096, 128]}
        for local_count, remote_count in ((2, 8), (8, 2)):
            with self.subTest(local_count=local_count, remote_count=remote_count):
                self.adapter.registration = lambda: {
                    "slots": [dict(slot) for _ in range(local_count)]
                }
                peer = {
                    "version": 3,
                    "has_mamba": False,
                    "page_size": self.adapter.page_size,
                    "profile": "shared-profile",
                    "slots": [dict(slot) for _ in range(remote_count)],
                }
                self.adapter.validate_peer(peer)
                with self.assertRaisesRegex(
                    ValueError, "incompatible compression profile"
                ):
                    self.adapter.validate_peer({**peer, "profile": "different-profile"})

    def test_previous_process_ack_cannot_release_credit(self):
        adapter = self.adapter
        adapter.pending_lock = threading.Lock()
        event = threading.Event()
        adapter.pending = {"0": (event, {}, ReceiveCredits(0), 0)}
        adapter.handle_ack(
            [ACK, msgspec.msgpack.encode([{"epoch": "old", "sequence": "0"}])]
        )
        self.assertFalse(event.is_set())

    def test_paged_indices_exclude_invalid_tail(self):
        self.assertEqual(KVServeAdapter.token_ids([3, 7], 4, 5), [12, 13, 14, 15, 28])
        with self.assertRaises(ValueError):
            KVServeAdapter.token_ids([3, 7], 4, 4)
        self.assertEqual(
            KVServeAdapter.token_ids(np.array([3, 7]), 4, 5), [12, 13, 14, 15, 28]
        )

    def test_duplicate_ack_does_not_double_release(self):
        adapter = self.adapter
        adapter.pending_lock = threading.Lock()
        pool = ReceiveCredits(1)
        slot = pool.acquire()[0]
        event = threading.Event()
        adapter.pending = {"1": (event, {}, pool, slot)}
        msg = [ACK, msgspec.msgpack.encode([{"epoch": "test", "sequence": "1"}])]
        adapter.handle_ack(msg)
        adapter.handle_ack(msg)
        self.assertEqual(pool.available, [0])


if __name__ == "__main__":
    unittest.main()
