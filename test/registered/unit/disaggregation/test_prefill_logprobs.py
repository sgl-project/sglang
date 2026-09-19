import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import msgspec
import numpy as np

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation import prefill_logprobs
from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import (
    CommonKVManager,
    CommonKVReceiver,
    CommonKVSender,
)
from sglang.srt.disaggregation.mooncake.conn import TransferInfo as MooncakeTransferInfo
from sglang.srt.disaggregation.nixl.conn import TransferInfo as NixlTransferInfo

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class MetadataReceiver(CommonKVReceiver):
    poll = Mock()
    failure_exception = Mock()


class TestPrefillLogprobs(unittest.TestCase):
    def setUp(self):
        self.values = SimpleNamespace(
            **{name: None for name in prefill_logprobs.FIELDS}
        )
        self.values.input_token_logprobs_val = [None, -float("inf"), -0.25]
        self.values.input_token_logprobs_idx = [1, 2, 3]
        self.values.output_token_logprobs_val = [np.float32(-0.25)]
        self.values.output_token_logprobs_idx = [4]
        self.values.output_token_ids_logprobs_val = [[-0.5, -1.5]]
        self.values.output_token_ids_logprobs_idx = [[3, 4]]

    def test_request_negotiation_preserves_legacy_frames(self):
        ids = np.array([3, 4], dtype=np.int32).tobytes()
        prefix = [b"7", b"127.0.0.1", b"19130", b"peer", ids, b"0"]
        for cls, tail in (
            (NixlTransferInfo, [b"1", b"", b"0", b"0"]),
            (MooncakeTransferInfo, [b"", b"1", b"0", b""]),
        ):
            with self.subTest(backend=cls.__module__):
                old = cls.from_zmq(prefix + tail)
                new = cls.from_zmq(prefix + tail + [b"1"])
                hidden = cls.from_zmq(prefix + tail + [b"2"])
                self.assertEqual(old.prefill_logprobs_version, 0)
                self.assertEqual(new.prefill_logprobs_version, 1)
                self.assertEqual(hidden.prefill_logprobs_version, 2)
                np.testing.assert_array_equal(old.dst_kv_indices, new.dst_kv_indices)

    def test_roundtrip_preserves_native_values_and_flat_arrays(self):
        self.values.input_top_logprobs_val_flat = np.array(
            [[-1, -2], [-3, -4]], dtype=np.float32
        )
        self.values.input_top_logprobs_idx_flat = np.array(
            [[1, 2], [3, 4]], dtype=np.int32
        )
        self.values.input_top_logprobs_flat_null_prefix = 1
        restored = SimpleNamespace(
            **{name: [] for name in prefill_logprobs.OUTPUT_FIELDS}
        )
        values = prefill_logprobs.decode(prefill_logprobs.encode(self.values))
        prefill_logprobs.restore_inputs(restored, values)
        prefill_logprobs.append_output(restored, values)
        for name in prefill_logprobs.INPUT_FIELDS:
            np.testing.assert_equal(getattr(restored, name), getattr(self.values, name))
        for name in prefill_logprobs.OUTPUT_FIELDS:
            np.testing.assert_equal(
                getattr(restored, name), getattr(self.values, name) or []
            )
        self.assertIsNone(prefill_logprobs.decode(b"\xc0"))
        with self.assertRaises(ValueError):
            prefill_logprobs.decode(msgspec.msgpack.encode((2, [None] * 9)))

    def test_hidden_states_and_legacy_negotiation(self):
        hidden = [[[0.25, -0.5], [0.75, 0.5]]]
        values = prefill_logprobs.decode(
            prefill_logprobs.encode(self.values, hidden_states=hidden, version=2)
        )
        self.assertEqual(values[-1], hidden)
        self.assertIsNone(
            prefill_logprobs.decode(prefill_logprobs.encode(self.values))[-1]
        )
        receiver = MetadataReceiver.__new__(MetadataReceiver)
        receiver.want_prefill_hidden_states = True
        self.assertEqual(receiver.prefill_metadata_version({}), b"0")
        receiver.want_prefill_logprobs = True
        self.assertEqual(receiver.prefill_metadata_version({}), b"1")
        self.assertEqual(
            receiver.prefill_metadata_version({"prefill_metadata_version": 2}), b"2"
        )
        manager = SimpleNamespace(
            supports_prefill_logprobs=True,
            transfer_infos={
                7: {
                    version: SimpleNamespace(
                        is_dummy=False, prefill_logprobs_version=version
                    )
                    for version in (1, 2)
                }
            },
            prefill_logprobs_send={},
        )
        sender = SimpleNamespace(kv_mgr=manager, bootstrap_room=7)
        self.assertTrue(
            CommonKVSender.set_prefill_logprobs(
                sender, self.values, hidden_states=hidden
            )
        )
        legacy_version, legacy_values = msgspec.msgpack.decode(
            manager.prefill_logprobs_send[7][1]
        )
        self.assertEqual((legacy_version, len(legacy_values)), (1, 15))
        self.assertEqual(
            prefill_logprobs.decode(manager.prefill_logprobs_send[7][2])[-1], hidden
        )

    def test_rank_fan_in_duplicate_late_messages_and_timeout(self):
        manager = CommonKVManager.__new__(CommonKVManager)
        manager.supports_prefill_logprobs = True
        manager.request_status = {7: KVPoll.Transferring}
        manager.prefill_logprobs_recv = {}
        manager.prefill_logprobs_lock = threading.Lock()
        manager.record_failure = Mock()
        manager.update_status = lambda room, status: manager.request_status.__setitem__(
            room, status
        )
        receiver = MetadataReceiver.__new__(MetadataReceiver)
        receiver.kv_mgr = manager
        receiver.want_prefill_logprobs = True
        receiver.bootstrap_room = 7
        receiver.bootstrap_infos = [{"prefill_logprobs_version": 1}]
        receiver.required_prefill_response_num = 2
        receiver._check_waiting_timeout = Mock(return_value=None)
        full = [
            b"PREFILL_LOGPROBS_V1",
            b"7",
            b"1",
            prefill_logprobs.encode(self.values),
        ]
        empty = [b"PREFILL_LOGPROBS_V1", b"7", b"0", b"\xc0"]
        self.assertTrue(manager.handle_prefill_logprobs(full))
        manager.handle_prefill_logprobs(full)
        self.assertEqual(receiver.poll_prefill_logprobs(), KVPoll.Transferring)
        manager.handle_prefill_logprobs(empty)
        self.assertEqual(receiver.poll_prefill_logprobs(), KVPoll.Success)
        self.assertEqual(
            receiver.prefill_logprobs()[0], self.values.input_token_logprobs_val
        )
        # A PP rank without logits must not overwrite hidden-only metadata.
        hidden_only = SimpleNamespace(
            **{name: None for name in prefill_logprobs.FIELDS}
        )
        hidden = [[[0.25, -0.5]]]
        manager.handle_prefill_logprobs(
            full[:3]
            + [prefill_logprobs.encode(hidden_only, hidden_states=hidden, version=2)]
        )
        manager.handle_prefill_logprobs(empty)
        self.assertEqual(receiver.prefill_logprobs()[-1], hidden)
        manager.prefill_logprobs_recv.clear()
        receiver._check_waiting_timeout.return_value = KVPoll.Failed
        self.assertEqual(receiver.poll_prefill_logprobs(), KVPoll.Failed)
        receiver.bootstrap_infos = [{}]
        self.assertEqual(receiver.poll_prefill_logprobs(), KVPoll.Success)
        receiver.bootstrap_addr = "peer"
        manager.required_prefill_response_num_table = {7: 2}
        manager.prefill_response_tracker = {}
        manager.addr_to_rooms_tracker = {"peer": {7}}
        receiver.clear()
        manager.handle_prefill_logprobs(full)
        self.assertEqual(manager.prefill_logprobs_recv, {})
        manager.request_status[7] = KVPoll.Transferring
        manager.handle_prefill_logprobs(full[:3] + [b"invalid msgpack"])
        self.assertEqual(manager.request_status[7], KVPoll.Failed)
        manager.record_failure.assert_called_once()


if __name__ == "__main__":
    unittest.main()
