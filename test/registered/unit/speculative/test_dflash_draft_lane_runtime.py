"""CPU transport/state regression: real byte copies, two independent peers."""

import ctypes
import unittest
from dataclasses import asdict
from queue import SimpleQueue
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.disaggregation.draft_lane import (
    CONTROL_TAG,
    DraftPeer,
    DraftRegion,
    decode_control,
    peer_to_dict,
)
from sglang.srt.disaggregation.draft_lane_runtime import DraftPDLane, DraftReadyPoller
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class DraftRuntimeTests(CustomTestCase):
    def setUp(self):
        # The protocol fixture copies real CPU tensor bytes. GPU stream drains
        # are exercised separately by actual RDMA and full serving regressions.
        drain = patch("torch.cuda.synchronize")
        drain.start()
        self.addCleanup(drain.stop)

    def fixture(self, decode_width=2):
        network = {}

        def make(name, prefill, ordinal):
            lane = object.__new__(DraftPDLane)
            lane.prefill = prefill
            lane.endpoint = (
                f"tcp://127.0.0.1:{41000 + ordinal + (10 if prefill else 0)}"
            )
            lane.inbox = SimpleQueue()
            lane.owners = {}
            lane.destinations = {}
            lane.slot_leases = {}
            lane.sources = {}
            lane.retired_sources = set()
            lane.receivers = {}
            lane.quarantined = set()
            lane.authority = True
            heads = 8 if prefill else 16 // decode_width
            lane.head_begin = ordinal * heads
            lane.head_end = (ordinal + 1) * heads
            lane.fingerprint = "checkpoint"
            lane.dtype = "bfloat16"
            lane.tensors = {}
            lane.regions = []
            for layer in range(5):
                for kind in ("K", "V"):
                    tensor = torch.zeros((512, heads, 4), dtype=torch.bfloat16)
                    region = DraftRegion(
                        layer,
                        kind,
                        "FULL" if layer == 4 else "WINDOW",
                        tensor.data_ptr(),
                        tensor.numel() * 2,
                        heads * 4 * 2,
                        512,
                    )
                    lane.regions.append(region)
                    lane.tensors[region.key] = tensor

            class Layout:
                context_length = 32
                stride = 64
                window_stride = 8

                def full_loc(self, slot, pos):
                    return 64 + (slot - 1) * 64 + pos

                def window_loc(self, slot, pos):
                    return 64 + (slot - 1) * 8 + pos % 8

            cache = SimpleNamespace(
                layout=Layout(),
                generations=[-1] * (decode_width + 1),
                allocated_lens=[0] * (decode_width + 1),
                target_requests=SimpleNamespace(req_generation=torch.tensor([0, 1])),
            )
            lane.worker = SimpleNamespace(
                device="cpu",
                _independent_cache=cache,
                _cohort=SimpleNamespace(
                    owner_rank=ordinal, owner_slots=1, committed={}
                ),
            )

            def send(endpoint, parts):
                network[endpoint].inbox.put(decode_control(parts))

            lane.manager = SimpleNamespace(
                _send_multipart_locked=send,
                record_failure=MagicMock(),
                update_status=MagicMock(),
                waiting_timeout=600,
            )
            network[lane.endpoint] = lane
            if not prefill:
                lane.peer = DraftPeer(
                    name,
                    "cohort",
                    "epoch",
                    ordinal,
                    "engine-" + name,
                    lane.endpoint,
                    "checkpoint",
                    "bfloat16",
                    ordinal * heads,
                    (ordinal + 1) * heads,
                    16,
                    4,
                    64,
                    12,
                    tuple(lane.regions),
                )
            return lane

        destinations = tuple(make(f"d{i}", False, i) for i in range(decode_width))
        d0 = destinations[0]
        p0, p1 = make("p0", True, 0), make("p1", True, 1)
        peers = tuple(d.peer for d in destinations)
        for destination in destinations:
            destination.peers = peers
        for source in (p0, p1):
            allowed = {
                d.peer.engine_session
                for d in destinations
                if source.head_begin <= d.head_begin < d.head_end <= source.head_end
            }

            def transfer(session, blocks, expected=allowed):
                self.assertIn(session, expected)
                for src, dst, size in blocks:
                    ctypes.memmove(dst, src, size)
                return 0

            source.manager._transfer_data = transfer

        class Receiver:
            bootstrap_room = 9
            prefill_dp_rank = 0
            prefill_info = SimpleNamespace(attn_tp_size=1, attn_cp_size=2, pp_size=1)
            sent = []

            def _get_bootstrap_info_from_server(self, dp, cp, tp, pp):
                return dict(rank_ip="127.0.0.1", rank_port=41010 + cp)

            def send_metadata(self, **kwargs):
                if not d0.defer_metadata(self, kwargs):
                    self.sent.append(kwargs)

        receiver = Receiver()
        receiver.sent = []
        req = SimpleNamespace(rid="request", kv=SimpleNamespace(req_pool_idx=1))
        return (*destinations, p0, p1), receiver, req

    def pump(self, lanes):
        for _ in range(12):
            for lane in lanes:
                lane.service()

    def send_chunk(self, source, receiver, start, end, distinct_heads=False):
        positions = torch.arange(start, end, dtype=torch.int64)
        rows = 64 + positions.remainder(8)
        for key, tensor in source.tensors.items():
            for p, row in zip(positions.tolist(), rows.tolist()):
                tensor[row].fill_(
                    p
                    + key[0] * 100
                    + (1000 if key[1] == "V" else 0)
                    + source.head_begin
                )
                if distinct_heads:
                    for head in range(tensor.shape[1]):
                        tensor[row, head].fill_(
                            p
                            + key[0] * 100
                            + (1000 if key[1] == "V" else 0)
                            + (source.head_begin + head) * 16
                        )
        batch = SimpleNamespace(
            reqs=[
                SimpleNamespace(
                    disagg_kv_sender=SimpleNamespace(
                        bootstrap_room=receiver.bootstrap_room
                    )
                )
            ],
            extend_lens=[end - start],
        )
        source.send_context_chunk(batch, positions, rows)

    @patch("torch.cuda.current_stream")
    def test_reserve_prepare_target_gate_and_both_peer_ready(self, stream):
        lanes, receiver, req = self.fixture()
        d0, d1, p0, p1 = lanes
        lease = d0.begin_lease(req, receiver, 20)
        receiver.send_metadata(kv_indices=[1, 2])
        self.assertEqual(receiver.sent, [])
        self.pump(lanes)
        self.assertEqual(receiver.sent, [{"kv_indices": [1, 2]}])
        self.assertFalse(d0.ready(9))
        for begin, end in [(0, 8), (8, 16), (16, 20)]:
            self.send_chunk(p0, receiver, begin, end)
            self.pump(lanes)
        self.assertFalse(d0.ready(9), "One peer cannot complete the draft cohort")
        for begin, end in [(0, 8), (8, 16), (16, 20)]:
            self.send_chunk(p1, receiver, begin, end)
            self.pump(lanes)
        self.assertTrue(d0.ready(9))
        for destination in (d0, d1):
            self.assertEqual(
                destination.worker._cohort.committed[1], (1, "request", 20)
            )
            for key, tensor in destination.tensors.items():
                for pos in range(0 if key[2] == "FULL" else 12, 20):
                    expected = torch.tensor(
                        pos
                        + key[0] * 100
                        + (1000 if key[1] == "V" else 0)
                        + destination.head_begin,
                        dtype=torch.bfloat16,
                    )
                    self.assertTrue(
                        torch.all(tensor[lease.row(pos, key[2])] == expected)
                    )
        d0.release_owner_slot(1)
        self.pump(lanes)
        self.assertFalse(d0.slot_leases)
        self.assertFalse(d1.slot_leases)
        self.assertFalse(d0.worker._cohort.committed)
        self.assertFalse(d1.worker._cohort.committed)
        receiver.bootstrap_room = 10
        req.rid = "next-request"
        d0.worker._independent_cache.target_requests.req_generation[1] = 2
        d0.begin_lease(req, receiver, 20)
        receiver.send_metadata(kv_indices=[3])
        self.pump(lanes)
        self.assertTrue(d0.slot_leases and d1.slot_leases)
        self.assertEqual(len(receiver.sent), 2)

    @patch("torch.cuda.current_stream")
    def test_duplicate_prepare_preserves_sent_and_retired_lease_stays_retired(
        self, stream
    ):
        lanes, receiver, req = self.fixture()
        d0, d1, p0, p1 = lanes
        lease = d0.begin_lease(req, receiver, 20)
        receiver.send_metadata(kv_indices=[1])
        self.pump(lanes)
        self.send_chunk(p0, receiver, 0, 8)
        payload = dict(
            room=9,
            lease=asdict(lease),
            peers=[peer_to_dict(p) for p in d0.peers],
            reply_to=d0.endpoint,
        )
        prior = set(p0.sources[9]["sent"])
        p0._service_message("PREPARE", payload)
        self.assertEqual(p0.sources[9]["sent"], prior)
        with self.assertRaisesRegex(RuntimeError, "rewrite"):
            self.send_chunk(p0, receiver, 0, 8)
        p0._service_message(
            "RELEASE", dict(room=9, lease_id=lease.lease_id, transport_drained=True)
        )
        with self.assertRaisesRegex(ValueError, "Retired"):
            p0._service_message("PREPARE", payload)

    @patch("torch.cuda.current_stream")
    def test_native_fake_health_initializes_both_peers_and_can_reuse_room(self, stream):
        from sglang.srt.disaggregation.base.conn import KVPoll
        from sglang.srt.disaggregation.mode import FAKE_BOOTSTRAP_HOST

        lanes, receiver, req = self.fixture()
        d0, d1, p0, p1 = lanes
        req.bootstrap_host = FAKE_BOOTSTRAP_HOST
        req.bootstrap_room = 9
        for destination in (d0, d1):
            for tensor in destination.tensors.values():
                tensor.fill_(9)
        lease = d0.begin_lease(req, receiver, 4)
        gate = DraftReadyPoller(SimpleNamespace(poll=lambda: KVPoll.Success), d0, 9)
        self.assertEqual(gate.poll(), KVPoll.Transferring)
        self.pump(lanes)
        self.assertEqual(gate.poll(), KVPoll.Success)
        self.assertTrue(d0.owners[9]["admitted"])
        self.assertFalse(p0.sources or p1.sources)
        for destination in (d0, d1):
            self.assertEqual(destination.worker._cohort.committed[1], (1, "request", 4))
            for key, tensor in destination.tensors.items():
                rows = [lease.row(p, key[2]) for p in range(4)]
                self.assertTrue(torch.all(tensor[rows] == 0))
                self.assertTrue(torch.all(tensor[100] == 9))
        d0.release_owner_slot(1)
        self.pump(lanes)
        req.rid = "next-health"
        d0.worker._independent_cache.target_requests.req_generation[1] = 2
        d0.begin_lease(req, receiver, 4)
        self.pump(lanes)
        self.assertTrue(d0.ready(9))
        self.assertEqual(d1.worker._cohort.committed[1], (2, "next-health", 4))
        p0.manager._transfer_data = MagicMock()
        fake_batch = SimpleNamespace(
            reqs=[SimpleNamespace(bootstrap_host=FAKE_BOOTSTRAP_HOST)], extend_lens=[4]
        )
        p0.send_context_chunk(fake_batch, torch.arange(4), torch.arange(64, 68))
        p0.manager._transfer_data.assert_not_called()

    def test_malformed_typed_envelope_does_not_consume_the_target_channel(self):
        lanes, receiver, req = self.fixture()
        d0 = lanes[0]
        self.assertTrue(d0.handle_control([CONTROL_TAG, b"{"]))
        self.assertTrue(d0.inbox.empty())
        self.assertFalse(d0.handle_control([b"9", b"legacy-target-status"]))

    @patch("torch.cuda.current_stream")
    def test_missing_ready_deadline_quarantines_without_releasing(self, stream):
        lanes, receiver, req = self.fixture()
        d0, d1, p0, p1 = lanes
        d0.begin_lease(req, receiver, 20)
        receiver.send_metadata(kv_indices=[1])
        self.pump(lanes)
        d0.owners[9]["deadline"] = 0
        d0.service()
        self.assertIn(9, d0.quarantined)
        self.assertTrue(d0.slot_leases and d1.slot_leases)
        self.assertFalse(d0.ready(9))
        d0.manager.record_failure.assert_called()

    @patch("torch.cuda.current_stream")
    def test_uncertain_dma_cannot_release_or_reuse_a_slot(self, stream):
        lanes, receiver, req = self.fixture()
        d0, d1, p0, p1 = lanes
        d0.begin_lease(req, receiver, 20)
        receiver.send_metadata(kv_indices=[1])
        self.pump(lanes)
        p0.manager._transfer_data = lambda session, blocks: -1
        with self.assertRaisesRegex(RuntimeError, "Unsettled"):
            self.send_chunk(p0, receiver, 0, 8)
        d0.release_owner_slot(1)
        self.pump(lanes)
        self.assertIn(9, d0.quarantined)
        self.assertIn(9, d1.quarantined)
        self.assertTrue(d0.slot_leases and d1.slot_leases)
        receiver.bootstrap_room = 10
        req.rid = "next-request"
        d0.begin_lease(req, receiver, 20)
        receiver.send_metadata(kv_indices=[2])
        self.pump(lanes)
        self.assertIn(10, d0.quarantined)
        self.assertFalse(d0.ready(10))
        self.assertEqual(
            len(receiver.sent), 1, "No target metadata after unsafe draft slot reuse"
        )

    @patch("torch.cuda.current_stream")
    def test_tp2_to_tp4_copies_distinct_heads_and_waits_for_all_four_ready(
        self, stream
    ):
        lanes, receiver, req = self.fixture(decode_width=4)
        d0, d1, d2, d3, p0, p1 = lanes
        lease = d0.begin_lease(req, receiver, 20)
        receiver.send_metadata(kv_indices=[1])
        self.pump(lanes)
        self.assertEqual(len(receiver.sent), 1)
        for source in (p0, p1):
            for begin, end in [(0, 8), (8, 16), (16, 20)]:
                self.send_chunk(source, receiver, begin, end, distinct_heads=True)
                self.pump(lanes)
            if source is p0:
                self.assertEqual(d0.owners[9]["ready"], {0, 1})
                self.assertFalse(d0.ready(9))
        self.assertTrue(d0.ready(9))
        self.assertEqual(d0.owners[9]["ready"], {0, 1, 2, 3})
        for destination in (d0, d1, d2, d3):
            self.assertEqual(
                destination.worker._cohort.committed[1], (1, "request", 20)
            )
            for key, tensor in destination.tensors.items():
                for pos in range(0 if key[2] == "FULL" else 12, 20):
                    expected = torch.tensor(
                        [
                            [
                                pos
                                + key[0] * 100
                                + (1000 if key[1] == "V" else 0)
                                + (destination.head_begin + head) * 16
                            ]
                            * 4
                            for head in range(4)
                        ],
                        dtype=torch.bfloat16,
                    )
                    self.assertTrue(
                        torch.equal(tensor[lease.row(pos, key[2])], expected)
                    )
                self.assertTrue(torch.all(tensor[400] == 0))
        d0.release_owner_slot(1)
        self.pump(lanes)
        for destination in (d0, d1, d2, d3):
            self.assertFalse(destination.slot_leases)
            self.assertFalse(destination.worker._cohort.committed)

    @patch("torch.cuda.current_stream")
    def test_tp4_partial_destination_dma_failure_keeps_all_slots_quarantined(
        self, stream
    ):
        lanes, receiver, req = self.fixture(decode_width=4)
        d0, d1, d2, d3, p0, p1 = lanes
        d0.begin_lease(req, receiver, 4)
        receiver.send_metadata(kv_indices=[1])
        self.pump(lanes)
        original = p0.manager._transfer_data

        def transfer(session, blocks):
            return (
                -1 if session == d1.peer.engine_session else original(session, blocks)
            )

        p0.manager._transfer_data = transfer
        with self.assertRaisesRegex(RuntimeError, "Unsettled"):
            self.send_chunk(p0, receiver, 0, 4, distinct_heads=True)
        self.pump(lanes)
        self.assertFalse(d0.ready(9))
        d0.release_owner_slot(1)
        self.pump(lanes)
        for destination in (d0, d1, d2, d3):
            self.assertTrue(destination.slot_leases)
            self.assertIn(9, destination.quarantined)


if __name__ == "__main__":
    unittest.main()
