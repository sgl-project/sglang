"""Regressions for a live peer remaining unusable after metadata invalidation.

The fake implements the external NIXL status/generation boundary; the SGLang
worker, posting helpers, barrier, descriptor cleanup, and status messages run.
"""

import threading
import unittest
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.nixl import conn
from sglang.srt.disaggregation.nixl.peer_recovery import PeerRecovery
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class Disconnected(Exception):
    pass


class Missing(Exception):
    pass


class NativeAgent:
    def __init__(self):
        self.metadata = {"bad", "healthy"}
        self.handles = []
        self.post_fault = False
        self.read_fault = False
        self.blocked = threading.Event()
        self.proceed = threading.Event()
        self.proceed.set()
        self.released = []
        self.rebuilds = []
        self.prepared = []

    def check_remote_metadata(self, peer):
        return peer in self.metadata

    def add_remote_agent(self, metadata):
        self.metadata.add(metadata.decode())

    def remove_remote_agent(self, peer):
        self.metadata.remove(peer)

    def get_xfer_descs(self, addrs, kind):
        return addrs

    def initialize_xfer(self, op, src, dst, peer, notif):
        if peer not in self.metadata:
            raise Missing()
        h = SimpleNamespace(peer=peer, state="PROC", notif=notif)
        self.handles.append(h)
        return h

    def transfer(self, h):
        if self.post_fault and h.peer == "bad":
            self.post_fault = False
            self.metadata.remove(h.peer)
            h.state = "DISCONNECTED"
            raise Disconnected()
        return "PROC"

    def check_xfer_state(self, h):
        if h.state == "DISCONNECTED":
            raise Disconnected()
        if h.peer not in self.metadata:
            raise Missing()
        if self.read_fault and h.peer == "bad":
            self.read_fault = False
            self.metadata.remove(h.peer)
            h.state = "DISCONNECTED"
            raise Disconnected()
        if h.peer == "bad" and not self.proceed.is_set():
            self.blocked.set()
            return "PROC"
        h.state = "DONE"
        return h.state

    def release_xfer_handle(self, h):
        assert h.state in ("DONE", "DISCONNECTED"), "released an active transfer"
        self.released.append(h)

    def prep_xfer_dlist(self, peer, descs, kind):
        h = (peer, descs, kind)
        self.prepared.append(h)
        return h

    def release_dlist_handle(self, h):
        self.rebuilds.append(h)


class Queue:
    def __init__(self, chunks):
        self.chunks = deque(chunks)

    def get(self):
        if not self.chunks:
            raise SystemExit()
        return self.chunks.popleft()


class TestPeerRecovery(CustomTestCase):
    def setUp(self):
        self.patches = [
            patch.object(conn, "_NIXL_REMOTE_DISCONNECT_ERRORS", (Disconnected,)),
            patch.object(conn, "_NIXL_NOT_FOUND_ERRORS", (Missing,)),
        ]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

    def manager(self):
        mgr = object.__new__(conn.NixlKVManager)
        mgr.agent = NativeAgent()
        mgr.request_status = {}
        mgr.transfer_infos = {}
        mgr.req_to_decode_prefix_len = {}
        mgr._staging_outstanding = defaultdict(int)
        mgr._deferred_ack_targets = {}
        mgr.enable_staging = False
        mgr.enable_deferred_decode_kv_release = False
        mgr._staging_ctx = None
        mgr.is_mla_backend = False
        mgr.is_hybrid_mla_backend = False
        mgr.attn_tp_size = 1
        mgr.attn_tp_rank = mgr.attn_cp_rank = mgr.pp_rank = 0
        mgr.attn_cp_size = mgr.pp_size = 1
        mgr.transfer_source_rank = 0
        mgr.kv_args = SimpleNamespace(
            engine_rank=0,
            kv_data_ptrs=[],
            kv_item_lens=[],
            aux_data_ptrs=[100],
            aux_item_lens=[8],
            pp_rank=0,
        )
        mgr.exceptions = {}
        mgr.failure_records = {}
        mgr.failure_lock = threading.Lock()
        mgr.prep_handles = {
            "bad": "bad-dlist",
            "healthy": "healthy-dlist",
            "": "source",
        }
        mgr.prep_handles_slice_dst = {}
        mgr.decode_kv_args_table = {
            peer: SimpleNamespace(
                decode_tp_size=1,
                dst_aux_ptrs=[200],
                gpu_id=0,
                kv_xfer_segments=None,
                dst_kv_item_lens=[],
                agent_metadata=peer.encode(),
            )
            for peer in ("bad", "healthy")
        }
        mgr.messages = []
        endpoint = "tcp://127.0.0.1:5555"
        mgr._socket_lock = threading.Lock()
        mgr._socket_cache = {
            endpoint: SimpleNamespace(send_multipart=mgr.messages.append)
        }
        mgr._monitor_cache = {}
        mgr._socket_send_locks = {endpoint: threading.Lock()}
        mgr.shutdowns = []
        mgr._peer_recovery = PeerRecovery(
            mgr.agent,
            lambda peer: mgr.decode_kv_args_table[peer].agent_metadata,
            mgr._rebuild_peer_descriptors,
            mgr.shutdowns.append,
        )
        return mgr

    def chunk(self, mgr, room, peers=("bad",)):
        mgr.request_status[room] = KVPoll.WaitingForInput
        mgr.transfer_infos[room] = {
            peer: conn.TransferInfo(
                room=room,
                endpoint="127.0.0.1",
                dst_port=5555,
                agent_name=peer,
                dst_kv_indices=np.array([], dtype=np.int32),
                dst_aux_index=0,
                required_dst_info_num=1,
                dst_state_indices=[],
            )
            for peer in peers
        }
        return conn.TransferKVChunk(
            room=room,
            prefill_kv_indices=np.array([], dtype=np.int32),
            index_slice=slice(0, 0),
            is_last_chunk=True,
            chunk_id=0,
            prefill_aux_index=0,
            state_indices=None,
        )

    def run_chunks(self, mgr, chunks):
        with self.assertRaises(SystemExit):
            mgr.transfer_worker(Queue(chunks))

    def test_recovery_default_version_gate_and_opt_out(self):
        for version, backend, mode, disabled, expected in (
            ("1.4.1", "UCX", "prefill", False, True),
            ("1.5.0", "UCX", "prefill", False, True),
            ("1.4.0", "UCX", "prefill", False, False),
            ("1.3.2", "UCX", "prefill", False, False),
            ("1.4.1", "UCCL", "prefill", False, False),
            ("1.4.1", "UCX", "decode", False, False),
            ("1.4.1", "UCX", "prefill", True, False),
        ):
            with (
                self.subTest(
                    version=version, backend=backend, mode=mode, disabled=disabled
                ),
                patch.dict("os.environ"),
                patch("sglang.srt.utils.common.version", return_value=version),
            ):
                flag = conn.envs.SGLANG_DISAGGREGATION_NIXL_ENABLE_RECONNECT
                flag.clear()
                if disabled:
                    flag.set(False)
                mgr = self.manager()
                mgr.disaggregation_mode = conn.DisaggregationMode(mode)
                mgr._init_peer_recovery(backend)
                self.assertEqual(mgr._peer_recovery is not None, expected)

    def test_post_disconnect_fails_one_request_and_next_request_succeeds(self):
        """A post exception previously lost the handle and permanently poisoned the peer."""
        mgr = self.manager()
        mgr.agent.post_fault = True
        self.run_chunks(mgr, [self.chunk(mgr, 1), self.chunk(mgr, 2)])
        self.assertEqual(mgr.request_status, {1: KVPoll.Failed, 2: KVPoll.Success})
        self.assertEqual(len(mgr.agent.released), 2)
        self.assertEqual(mgr.agent.rebuilds, ["bad-dlist"])
        self.assertEqual(mgr.prep_handles, {"healthy": "healthy-dlist", "": "source"})
        self.assertEqual(mgr.shutdowns, [])
        self.assertEqual(len(mgr.messages), 1)

    def test_drained_failed_request_acknowledges_decode_abort(self):
        """Recovery must not leave decode holding pages until its abort timeout."""
        for cleared in (False, True):
            with self.subTest(cleared=cleared):
                mgr = self.manager()
                mgr.enable_deferred_decode_kv_release = True
                mgr.agent.post_fault = True
                chunk = self.chunk(mgr, 1)
                check = mgr.agent.check_xfer_state
                aborted = []

                def abort_during_poll(handle):
                    if not aborted:
                        aborted.append(True)
                        mgr._handle_abort_notification(
                            [b"ABORT", b"1", b"127.0.0.1", b"5555"]
                        )
                        self.assertFalse(mgr.messages)
                        if cleared:
                            sender = object.__new__(conn.NixlKVSender)
                            sender.kv_mgr = mgr
                            sender.bootstrap_room = 1
                            sender.clear()
                    return check(handle)

                mgr.agent.check_xfer_state = abort_during_poll
                self.run_chunks(mgr, [chunk])
                self.assertEqual(len(mgr.agent.released), 1)
                self.assertNotIn(1, mgr._staging_outstanding)
                self.assertNotIn(1, mgr._deferred_ack_targets)
                self.assertEqual(
                    [m for m in mgr.messages if m[0] == b"ABORT_ACK"],
                    [[b"ABORT_ACK", b"1", b"0"]],
                )
                self.assertEqual(mgr.shutdowns, [])

    def test_registration_failure_after_post_error_does_not_kill_worker(self):
        mgr = self.manager()
        mgr.agent.post_fault = True
        add = mgr.agent.add_remote_agent
        failures = [True]

        def add_once_failing(metadata):
            if failures:
                failures.pop()
                raise RuntimeError("temporary metadata import failure")
            return add(metadata)

        mgr.agent.add_remote_agent = add_once_failing
        self.run_chunks(mgr, [self.chunk(mgr, 1), self.chunk(mgr, 2)])
        self.assertEqual(mgr.request_status, {1: KVPoll.Failed, 2: KVPoll.Success})
        self.assertEqual(len(mgr.agent.released), 2)
        self.assertEqual(mgr.shutdowns, [])

    def test_idle_metadata_loss_rebuilds_destination_geometry_only(self):
        mgr = self.manager()
        mgr.kv_args.kv_item_lens = [4]
        mgr.kv_args.kv_layer_ids = []
        mgr.src_mem_kind = "VRAM"
        peer = mgr.decode_kv_args_table["bad"]
        peer.agent_name = "bad"
        peer.dst_kv_item_lens = [4]
        peer.dst_kv_ptrs = [1000]
        peer.dst_kv_mem_kinds = ["VRAM"]
        peer.dst_kv_layer_ids = []
        peer.dst_num_slots = 3
        peer.requires_dcp_relayout = False
        mgr.agent.metadata.remove("bad")
        self.run_chunks(mgr, [self.chunk(mgr, 4)])
        self.assertEqual(mgr.request_status[4], KVPoll.Success)
        self.assertEqual(len(mgr.agent.prepared), 1)
        peer, descs, kind = mgr.agent.prepared[0]
        self.assertEqual((peer, kind), ("bad", "VRAM"))
        np.testing.assert_array_equal(descs, [[1000, 4, 0], [1004, 4, 0], [1008, 4, 0]])
        self.assertEqual(mgr.prep_handles["healthy"], "healthy-dlist")
        self.assertEqual(mgr.prep_handles[""], "source")

    def test_recovery_releases_all_peer_descriptor_cache_variants(self):
        """Stale slice or mixed-memory descriptors must not survive a peer generation."""
        mgr = self.manager()
        mgr.prep_handles_slice_dst = {
            "bad": ("slice", 3, 0),
            "healthy": ("healthy-slice", 3, 0),
        }
        mgr.decode_kv_args_table["bad"].kv_xfer_segments = [
            SimpleNamespace(src_handle="shared-source", dst_handle="mixed-dst")
        ]
        mgr.agent.metadata.remove("bad")
        self.run_chunks(mgr, [self.chunk(mgr, 5)])
        self.assertEqual(mgr.request_status[5], KVPoll.Success)
        self.assertEqual(mgr.agent.rebuilds, ["bad-dlist", "slice", "mixed-dst"])
        self.assertEqual(
            mgr.prep_handles_slice_dst, {"healthy": ("healthy-slice", 3, 0)}
        )

    def test_missing_metadata_does_not_make_a_running_sibling_terminal(self):
        """After one error erases metadata, a sibling must be polled again before ack."""
        mgr = self.manager()
        batch = mgr._peer_recovery.begin(["bad"])
        first = mgr.send_aux("bad", 0, [200], 0, "1_aux")
        second = mgr.send_aux("bad", 0, [200], 1, "2_aux")
        mgr.agent.read_fault = True
        mgr.agent.proceed.clear()
        # A deterministic native boundary: the sibling's poll first observes
        # missing metadata, then blocks until the test permits completion.
        original = mgr.agent.check_xfer_state
        polls = []

        def check(h):
            polls.append(h)
            if h is second and len(polls) >= 4:
                self.assertEqual(mgr.agent.released, [])
                self.assertEqual(mgr.agent.rebuilds, [])
                mgr.agent.proceed.set()
            return original(h)

        mgr.agent.check_xfer_state = check
        try:
            self.assertEqual(mgr._await_handles([], failure_seen=False), (True, True))
        finally:
            mgr._peer_recovery.end()
        self.assertIn(second, mgr.agent.released)
        self.assertEqual(len(mgr.agent.released), 2)
        self.assertEqual(mgr.shutdowns, [])

    def test_unsettled_transfer_never_releases_handles_or_notifies_decode(self):
        mgr = self.manager()
        mgr.agent.proceed.clear()
        mgr._peer_recovery.begin(["bad"])
        mgr.send_aux("bad", 0, [200], 0, "1_aux")
        with patch.object(conn, "NIXL_ERR_SETTLE_TIMEOUT_S", 0):
            with self.assertRaises(SystemExit):
                mgr._await_handles([], failure_seen=True)
        self.assertEqual(mgr.agent.released, [])
        self.assertEqual(mgr.agent.rebuilds, [])
        self.assertEqual(mgr.messages, [])
        self.assertEqual(len(mgr.shutdowns), 1)
        mgr._peer_recovery.end()

    def test_duplicate_registration_does_not_block_bootstrap_thread(self):
        mgr = self.manager()
        with ThreadPoolExecutor(max_workers=1) as worker:
            with mgr._peer_recovery.lock("bad"):
                worker.submit(
                    mgr._add_remote_peer, SimpleNamespace(agent_name="bad")
                ).result(timeout=5)

    def test_peer_wait_does_not_block_healthy_peer(self):
        """A held peer lock must not become a global transport lock."""
        mgr = self.manager()
        chunk = self.chunk(mgr, 3, ("healthy",))
        with ThreadPoolExecutor(max_workers=1) as worker:
            with mgr._peer_recovery.lock("bad"):
                worker.submit(self.run_chunks, mgr, [chunk]).result(timeout=5)
        self.assertEqual(mgr.request_status[3], KVPoll.Success)


if __name__ == "__main__":
    unittest.main()
