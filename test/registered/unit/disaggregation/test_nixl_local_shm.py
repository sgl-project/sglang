"""Completion and bounds contracts for same-node HiSparse writes."""

import os
import socket
import sys
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.disaggregation.nixl.local_shm import (
    LocalShmExport,
    LocalShmMapping,
    LocalShmTransfer,
    copy_plan,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestLocalShm(CustomTestCase):
    def handle(self, error=None):
        owner = SimpleNamespace(pending=set())
        event = Mock()
        event.query.return_value = False
        notify = Mock()
        handle = LocalShmTransfer(owner, event, object(), notify, error)
        owner.pending.add(handle)
        return handle

    def test_completion_notifies_once(self):
        handle = self.handle()
        self.assertEqual(handle.poll(), "PROC")
        handle.notify.assert_not_called()
        self.assertIn(handle, handle.owner.pending)
        handle.event.query.return_value = True
        self.assertEqual(handle.poll(), "DONE")
        self.assertEqual(handle.poll(), "DONE")
        handle.notify.assert_called_once()
        self.assertFalse(handle.owner.pending)

    def test_partial_submit_error_still_waits_for_dma(self):
        handle = self.handle(RuntimeError("submission failed"))
        self.assertEqual(handle.poll(), "PROC")
        self.assertIn(handle, handle.owner.pending)
        handle.event.query.return_value = True
        self.assertEqual(handle.poll(), "ERR")
        handle.notify.assert_not_called()
        self.assertFalse(handle.owner.pending)

    def test_unknown_completion_retains_mapping(self):
        handle = self.handle()
        handle.event.query.side_effect = RuntimeError("device unavailable")
        with self.assertRaises(RuntimeError):
            handle.poll()
        self.assertIn(handle, handle.owner.pending)
        handle.notify.assert_not_called()

    def test_missing_event_retains_mapping(self):
        handle = self.handle()
        handle.event = None
        with self.assertRaises(RuntimeError):
            handle.poll()
        self.assertIn(handle, handle.owner.pending)

    def test_notification_failure_is_settled_error(self):
        handle = self.handle()
        handle.event.query.return_value = True
        handle.notify.side_effect = RuntimeError("peer gone")
        self.assertEqual(handle.poll(), "ERR")
        self.assertFalse(handle.owner.pending)

    def test_shuffled_pages_and_reused_slots(self):
        self.assertEqual(
            copy_plan([64], [128], [16], [3, 1], [0, 7]),
            [(0, 48, 0, 16), (0, 16, 112, 16)],
        )
        self.assertEqual(copy_plan([64], [128], [16], [0], [7]), [(0, 0, 112, 16)])

    def test_invalid_indices_and_geometry(self):
        for src, dst in [([-1], [0]), ([4], [0]), ([0], [8]), ([0, 1], [0])]:
            with self.subTest(src=src, dst=dst), self.assertRaises(ValueError):
                copy_plan([64], [128], [16], src, dst)
        with self.assertRaises(ValueError):
            copy_plan([64, 64], [128], [16], [0], [0])
        with self.assertRaises(TypeError):
            copy_plan([64], [128], [16], [0.5], [0])
        with self.assertRaises(ValueError):
            copy_plan([64], [128], [0], [0], [0])

    def test_barrier_waits_for_local_copy_and_nixl_state(self):
        from sglang.srt.disaggregation.nixl.conn import NixlKVManager

        manager = NixlKVManager.__new__(NixlKVManager)
        manager.agent = Mock()
        manager.agent.check_xfer_state.side_effect = ["PROC", "DONE"]
        handle = self.handle()
        handle.event.query.side_effect = [False, True]
        self.assertEqual(
            manager._await_handles([handle, "state"], failure_seen=False),
            (True, False),
        )
        handle.notify.assert_called_once()

    def test_barrier_does_not_ack_unknown_local_completion(self):
        from sglang.srt.disaggregation.nixl.conn import NixlKVManager

        manager = NixlKVManager.__new__(NixlKVManager)
        handle = self.handle()
        handle.event = None
        self.assertEqual(
            manager._await_handles([handle], failure_seen=True), (False, True)
        )
        self.assertIn(handle, handle.owner.pending)

    def test_unenabled_send_keeps_existing_backend(self):
        from sglang.srt.disaggregation.nixl.conn import NixlKVManager

        manager = NixlKVManager.__new__(NixlKVManager)
        manager.local_shm_enabled = False
        manager.local_shm_errors = {}
        manager.local_shm_mappings = {}
        manager.src_mem_kind = "VRAM"
        manager.kv_args = SimpleNamespace(kv_data_ptrs=[10], kv_item_lens=[16])
        manager._send_kvcache_generic = Mock(return_value="nixl")
        self.assertEqual(
            manager.send_kvcache("peer", [0], [20], [0], 0, "done"), "nixl"
        )
        manager._send_kvcache_generic.assert_called_once()

    def test_failed_mapping_cannot_fall_back_to_nixl_write(self):

        manager, info = self.mapping_manager()
        manager.local_shm_errors = {"peer": "mapping failed"}
        manager.decode_kv_args_table = {"peer": info}
        manager._send_kvcache_generic = Mock()
        with self.assertRaisesRegex(RuntimeError, "mapping failed"):
            manager.send_kvcache("peer", [0], [20], [0], 0, "done")
        manager._send_kvcache_generic.assert_not_called()

    def test_graceful_cleanup_releases_only_settled_mappings(self):
        from sglang.srt.disaggregation.nixl.conn import NixlKVManager

        manager = NixlKVManager.__new__(NixlKVManager)
        settled = Mock()
        pending = Mock()
        pending.close.side_effect = RuntimeError("DMA unresolved")
        manager._local_shm_shutdown = threading.Event()
        manager._local_shm_reaper = None
        manager._local_shm_lock = threading.RLock()
        manager.local_shm_mappings = {"settled": settled, "pending": pending}
        with self.assertLogs("sglang.srt.disaggregation.nixl.conn", level="ERROR"):
            manager.release_host_resources()
        settled.close.assert_called_once()
        self.assertEqual(manager.local_shm_mappings, {"pending": pending})

    def test_sender_preserves_early_send_producer_event(self):
        from sglang.srt.disaggregation.nixl.conn import NixlKVSender

        sender = NixlKVSender.__new__(NixlKVSender)
        sender._send_failed = False
        sender._prepare_send_indices = Mock(
            return_value=([0], slice(0, 1), False, False)
        )
        sender._transfer_start_time = None
        sender.kv_mgr = SimpleNamespace(
            local_shm_enabled=True, add_transfer_request=Mock()
        )
        sender.bootstrap_room = 1
        sender.chunk_id = 0
        sender.aux_index = 0
        sender._record_transfer_indices = Mock()
        event = object()
        sender._early_send_wait_event = event
        sender.send([0])
        self.assertIs(
            sender.kv_mgr.add_transfer_request.call_args.kwargs["wait_event"], event
        )
        self.assertIsNone(sender._early_send_wait_event)

    def test_host_pool_closes_export_before_releasing_buffer(self):
        from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost

        pool = MLATokenToKVPoolHost.__new__(MLATokenToKVPoolHost)
        pool.pin_memory = False
        buffer = object()
        pool.kv_buffer = buffer
        observed = []
        pool.register_host_resource_cleanup(lambda: observed.append(pool.kv_buffer))
        other = MLATokenToKVPoolHost.__new__(MLATokenToKVPoolHost)
        self.assertIsNone(other._host_resource_cleanup)
        pool.destroy()
        pool.destroy()
        self.assertEqual(observed, [buffer])
        self.assertIsNone(pool.kv_buffer)
        self.assertFalse(pool._host_resource_cleanup)

    def mapping_manager(self):
        from sglang.srt.disaggregation.nixl.conn import NixlKVManager

        manager = NixlKVManager.__new__(NixlKVManager)
        manager.local_shm_enabled = True
        manager.local_shm_mappings = {}
        manager.local_shm_errors = {}
        manager._local_shm_retry_at = {}
        manager._local_shm_lock = threading.RLock()
        manager.is_mla_backend = True
        manager.attn_tp_size = 4
        manager.kv_args = SimpleNamespace(
            kv_data_ptrs=[10], kv_item_lens=[16], kv_layer_ids=[0], gpu_id=0
        )
        info = SimpleNamespace(
            agent_name="peer",
            local_shm={"items": [16]},
            decode_tp_size=4,
            dst_dcp_size=1,
            staging_base_ptr=0,
            dst_kv_mem_kinds=["DRAM"],
            dst_kv_item_lens=[16],
            dst_kv_layer_ids=[0],
        )
        return manager, info

    def test_transient_mapping_failure_recovers_after_backoff(self):
        manager, info = self.mapping_manager()
        module = "sglang.srt.disaggregation.nixl.conn"
        mapping = object()
        with (
            patch(module + ".same_host", return_value=True),
            patch(
                module + ".LocalShmMapping",
                side_effect=[TimeoutError("FD timeout"), mapping],
            ) as create,
            patch(module + ".time.monotonic", return_value=10) as clock,
        ):
            with self.assertRaisesRegex(RuntimeError, "FD timeout"):
                manager._ensure_local_shm_mapping(info)
            with self.assertRaises(RuntimeError):
                manager._ensure_local_shm_mapping(info)
            self.assertEqual(create.call_count, 1)
            clock.return_value = 12
            self.assertIs(manager._ensure_local_shm_mapping(info), mapping)
            self.assertEqual(create.call_count, 2)
            self.assertFalse(manager.local_shm_errors)
            self.assertFalse(manager._local_shm_retry_at)
            self.assertIs(manager._ensure_local_shm_mapping(info), mapping)
            self.assertEqual(create.call_count, 2)

    def test_layout_failure_is_not_retried(self):
        manager, info = self.mapping_manager()
        info.decode_tp_size = 8
        module = "sglang.srt.disaggregation.nixl.conn"
        with (
            patch(module + ".same_host", return_value=True),
            patch(module + ".LocalShmMapping") as create,
        ):
            for _ in range(2):
                with self.assertRaisesRegex(RuntimeError, "matching flat MLA"):
                    manager._ensure_local_shm_mapping(info)
            create.assert_not_called()
            self.assertFalse(manager._local_shm_retry_at)

    def test_dead_exporter_released_only_after_pending_copy_settles(self):
        manager, info = self.mapping_manager()
        mapping = Mock(pending={"copy"})
        mapping.exporter_closed.return_value = True
        manager.local_shm_mappings["peer"] = mapping
        manager._reap_local_shm_mappings()
        mapping.close.assert_not_called()
        mapping.pending.clear()
        manager._reap_local_shm_mappings()
        mapping.close.assert_called_once()
        self.assertFalse(manager.local_shm_mappings)
        # Late queued requests must not silently use NIXL or reopen the dead pool.
        with self.assertRaisesRegex(RuntimeError, "exporter has closed"):
            manager._ensure_local_shm_mapping(info)

    def test_live_exporter_and_failed_unregister_are_retained(self):
        manager, _ = self.mapping_manager()
        live = Mock(pending=set())
        live.exporter_closed.return_value = False
        dead = Mock(pending=set())
        dead.exporter_closed.return_value = True
        dead.close.side_effect = RuntimeError("still pinned")
        manager.local_shm_mappings = {"live": live, "dead": dead}
        with self.assertLogs("sglang.srt.disaggregation.nixl.conn", level="ERROR"):
            manager._reap_local_shm_mappings()
        live.close.assert_not_called()
        self.assertEqual(set(manager.local_shm_mappings), {"live", "dead"})

    def test_reaper_cannot_race_submission(self):
        manager, _ = self.mapping_manager()
        mapping = Mock(pending=set())
        mapping.exporter_closed.return_value = True
        manager.local_shm_mappings["peer"] = mapping
        started = threading.Event()

        def reap():
            started.set()
            manager._reap_local_shm_mappings()

        with manager._local_shm_lock:
            worker = threading.Thread(target=reap)
            worker.start()
            self.assertTrue(started.wait(2))
            mapping.pending.add("submitted")
        worker.join(2)
        self.assertFalse(worker.is_alive())
        mapping.close.assert_not_called()

    def test_empty_chunk(self):
        self.assertEqual(copy_plan([64], [128], [16], [], []), [])


@unittest.skipUnless(sys.platform == "linux", "FD exchange requires Linux")
class TestLocalShmFDExchange(CustomTestCase):
    def setUp(self):
        self.fd = os.memfd_create("sglang-shm-test", os.MFD_CLOEXEC)
        os.ftruncate(self.fd, 4096)
        os.pwrite(self.fd, b"test", 0)
        buffer = Mock()
        buffer.is_contiguous.return_value = True
        buffer.data_ptr.return_value = 1000
        buffer.numel.return_value = 4096
        buffer.element_size.return_value = 1
        self.pool = SimpleNamespace(
            fd=self.fd,
            kv_buffer=buffer,
            get_contiguous_buf_infos=lambda: ([1000], [4096], [256]),
        )
        self.export = LocalShmExport(self.pool)

    def tearDown(self):
        self.export.close()
        os.close(self.fd)

    def receive(self, token):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
            conn.settimeout(2)
            conn.connect("\0" + self.export.descriptor["endpoint"])
            conn.sendall(token)
            return socket.recv_fds(conn, 1, 1, socket.MSG_CMSG_CLOEXEC)

    def test_wrong_generation_rejected_then_valid_fd_works(self):
        message, fds, _, _ = self.receive(b"x" * 32)
        self.assertEqual((message, fds), (b"", []))
        message, fds, _, _ = self.receive(self.export.token)
        self.assertEqual(message, b"F")
        self.assertEqual(len(fds), 1)
        try:
            self.assertEqual(os.pread(fds[0], 4, 0), b"test")
        finally:
            os.close(fds[0])

    def test_exporter_liveness_tracks_endpoint_shutdown(self):
        mapping = LocalShmMapping.__new__(LocalShmMapping)
        mapping.endpoint = self.export.descriptor["endpoint"]
        self.assertFalse(mapping.exporter_closed())
        self.export.close()
        self.assertTrue(mapping.exporter_closed())

    def test_probe_timeout_does_not_establish_peer_death(self):
        mapping = LocalShmMapping.__new__(LocalShmMapping)
        mapping.endpoint = self.export.descriptor["endpoint"]
        with patch("socket.socket") as sock:
            sock.return_value.__enter__.return_value.connect.side_effect = (
                TimeoutError()
            )
            self.assertFalse(mapping.exporter_closed())

    def test_pool_restart_has_new_identity(self):
        old = self.export.descriptor
        self.export.close()
        self.export = LocalShmExport(self.pool)
        self.assertNotEqual(old["endpoint"], self.export.descriptor["endpoint"])
        self.assertNotEqual(old["token"], self.export.descriptor["token"])
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
            with self.assertRaises(ConnectionRefusedError):
                conn.connect("\0" + old["endpoint"])


if __name__ == "__main__":
    unittest.main()
