"""Completion and bounds contracts for same-node HiSparse writes."""

import os
import socket
import sys
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from sglang.srt.disaggregation.nixl.local_shm import (
    LocalShmExport,
    LocalShmMapping,
    LocalShmPeer,
    LocalShmTransfer,
    build_copy_plan,
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
            build_copy_plan([64], [128], [16], [3, 1], [0, 7]),
            [(0, 48, 0, 16), (0, 16, 112, 16)],
        )
        self.assertEqual(
            build_copy_plan([64], [128], [16], [0], [7]), [(0, 0, 112, 16)]
        )

    def test_invalid_indices_and_geometry(self):
        for src, dst in [([-1], [0]), ([4], [0]), ([0], [8]), ([0, 1], [0])]:
            with self.subTest(src=src, dst=dst), self.assertRaises(ValueError):
                build_copy_plan([64], [128], [16], src, dst)
        with self.assertRaises(ValueError):
            build_copy_plan([64, 64], [128], [16], [0], [0])
        with self.assertRaises(TypeError):
            build_copy_plan([64], [128], [16], [0.5], [0])
        with self.assertRaises(ValueError):
            build_copy_plan([64], [128], [0], [0], [0])

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
        manager.src_mem_kind = "VRAM"
        manager.kv_args = SimpleNamespace(kv_data_ptrs=[10], kv_item_lens=[16])
        manager._send_kvcache_generic = Mock(return_value="nixl")
        self.assertEqual(
            manager.send_kvcache("peer", [0], [20], [0], 0, "done"), "nixl"
        )
        manager._send_kvcache_generic.assert_called_once()

    def test_failed_mapping_cannot_fall_back_to_nixl_write(self):

        manager, info = self.mapping_manager()
        manager._get_local_shm_peer("peer").error = RuntimeError("mapping failed")
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
        manager._local_shm_peers = {
            "settled": LocalShmPeer(mapping=settled),
            "pending": LocalShmPeer(mapping=pending),
        }
        with self.assertLogs("sglang.srt.disaggregation.nixl.conn", level="ERROR"):
            manager.release_host_resources()
        settled.close.assert_called_once()
        self.assertIsNone(manager._local_shm_peers["settled"].mapping)
        self.assertIs(manager._local_shm_peers["pending"].mapping, pending)

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
        manager._local_shm_peers = {}
        manager._local_shm_lock = threading.Lock()
        manager.is_mla_backend = True
        manager.attn_tp_size = 4
        manager.kv_args = SimpleNamespace(
            kv_data_ptrs=[10],
            kv_item_lens=[16],
            kv_layer_ids=[0],
            gpu_id=0,
            num_draft_entries=0,
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

    def test_mapping_exports_only_target_with_gpu_draft(self):
        manager, info = self.mapping_manager()
        manager.kv_args.num_draft_entries = 1
        manager.kv_args.kv_data_ptrs += [20]
        manager.kv_args.kv_item_lens += [32]
        manager.kv_args.kv_layer_ids += [78]
        info.dst_kv_mem_kinds += ["VRAM"]
        info.dst_kv_item_lens += [32]
        info.dst_kv_layer_ids += [78]
        module = "sglang.srt.disaggregation.nixl.conn"
        with (
            patch(module + ".same_host", return_value=True),
            patch(module + ".LocalShmMapping") as create,
        ):
            self.assertIs(manager._ensure_local_shm_mapping(info), create.return_value)
            self.assertEqual(create.call_args.args[0]["items"], [16])

    def mixed_manager(self, *, local_shm=True):
        from sglang.srt.disaggregation.nixl.conn import (
            NixlKVManager,
            _KVXferPreparedSegment,
        )

        manager = NixlKVManager.__new__(NixlKVManager)
        manager.local_shm_enabled = local_shm
        manager.agent = Mock()
        manager.agent.make_prepped_xfer.side_effect = ["gpu-handle", "second-handle"]
        manager.agent.transfer.return_value = "PROC"
        manager._num_slots_src = 32
        manager.kv_args = SimpleNamespace(
            kv_data_ptrs=[100, 200, 300],
            kv_data_lens=[512, 512, 1024],
            num_draft_entries=1,
        )
        manager.decode_kv_args_table = {
            "peer": SimpleNamespace(
                kv_xfer_segments=[
                    _KVXferPreparedSegment(
                        0, 2, "src-target", "dst-target", 64, "DRAM"
                    ),
                    _KVXferPreparedSegment(2, 3, "src-draft", "dst-draft", 128, "VRAM"),
                ]
            )
        }
        manager._get_local_shm_peer = Mock(
            return_value=SimpleNamespace(lock=threading.RLock())
        )
        mapping = Mock()
        mapping.submit.return_value = "cpu-handle"
        manager._ensure_local_shm_mapping = Mock(return_value=mapping)
        return manager, mapping

    def test_mixed_target_and_draft_use_independent_pages(self):
        manager, mapping = self.mixed_manager()
        handles = list(
            manager.send_kvcache_mixed(
                "peer",
                np.array([3, 1]),
                np.array([7, 8]),
                "7_kv_0_1_0",
                device_kv_indices=np.array([19, 11]),
            )
        )
        self.assertEqual(handles, ["cpu-handle", "gpu-handle"])
        args = mapping.submit.call_args.args
        self.assertEqual(args[0], [100, 200])
        self.assertEqual(args[1], [512, 512])
        np.testing.assert_array_equal(args[3], [7, 8])
        args[4]()
        manager.agent.send_notif.assert_called_once_with("peer", b"7_kv_0_1_0_part_0_2")
        transfer_args = manager.agent.make_prepped_xfer.call_args.args
        np.testing.assert_array_equal(transfer_args[2], [3, 1])
        np.testing.assert_array_equal(transfer_args[4], [19, 11])
        self.assertEqual(transfer_args[5], b"7_kv_0_1_0_part_1_2")

    def test_mixed_nixl_path_preserves_separate_device_indices(self):
        manager, mapping = self.mixed_manager(local_shm=False)
        handles = list(
            manager.send_kvcache_mixed(
                "peer",
                np.array([3]),
                np.array([7]),
                "7_kv_0_1_0",
                device_kv_indices=np.array([19]),
            )
        )
        self.assertEqual(len(handles), 2)
        mapping.submit.assert_not_called()
        calls = manager.agent.make_prepped_xfer.call_args_list
        np.testing.assert_array_equal(calls[0].args[4], [7, 71])
        np.testing.assert_array_equal(calls[1].args[4], [19])

    def test_mixed_bad_draft_pages_fail_before_target_submission(self):
        for pages in (np.array([19, 20]), np.array([-1]), np.array([128])):
            manager, mapping = self.mixed_manager()
            with self.subTest(pages=pages), self.assertRaises(ValueError):
                list(
                    manager.send_kvcache_mixed(
                        "peer",
                        np.array([3]),
                        np.array([7]),
                        "7_kv_0_1_0",
                        device_kv_indices=pages,
                    )
                )
            mapping.submit.assert_not_called()
            manager.agent.transfer.assert_not_called()

    def test_mixed_draft_requires_device_pages(self):
        manager, mapping = self.mixed_manager()
        with self.assertRaisesRegex(ValueError, "device destination indices"):
            list(
                manager.send_kvcache_mixed(
                    "peer", np.array([3]), np.array([7]), "7_kv_0_1_0"
                )
            )
        mapping.submit.assert_not_called()
        manager.agent.transfer.assert_not_called()

    def test_mixed_chunk_waits_for_both_parts_in_either_order(self):
        from sglang.srt.disaggregation.nixl.conn import NixlKVManager

        for order in ((0, 1), (1, 0)):
            with self.subTest(order=order):
                manager = NixlKVManager.__new__(NixlKVManager)
                manager.transfer_statuses = {
                    7: SimpleNamespace(
                        received_kv_parts_per_pp=None, expected_kv_parts_per_pp=None
                    )
                }
                manager._track_kv_arrival = Mock()
                manager._track_kv_part_arrival(7, 0, True, 0, order[0], 2)
                manager._track_kv_arrival.assert_not_called()
                manager._track_kv_part_arrival(7, 0, True, 0, order[1], 2)
                manager._track_kv_arrival.assert_called_once_with(7, 0, True, 0)

    def test_mixed_submit_failure_retains_all_created_handles(self):
        manager, _ = self.mixed_manager()
        manager.agent.transfer.side_effect = RuntimeError("post failed")
        handles = []
        with self.assertRaisesRegex(RuntimeError, "post failed"):
            handles.extend(
                manager.send_kvcache_mixed(
                    "peer",
                    np.array([3]),
                    np.array([7]),
                    "7_kv_0_1_0",
                    device_kv_indices=np.array([19]),
                )
            )
        self.assertEqual(handles, ["cpu-handle", "gpu-handle"])

    def test_device_page_metadata_roundtrip_and_legacy(self):
        from sglang.srt.disaggregation.nixl.conn import TransferInfo

        host = np.array([7, 8], dtype=np.int32)
        device = np.array([19, 11], dtype=np.int32)
        msg = [
            b"7",
            b"127.0.0.1",
            b"30000",
            b"peer",
            host.tobytes(),
            b"0",
            b"1",
            b"",
            b"0",
            b"0",
        ]
        self.assertIsNone(TransferInfo.from_zmq(msg).device_kv_indices)
        info = TransferInfo.from_zmq(msg + [device.tobytes()])
        np.testing.assert_array_equal(info.dst_kv_indices, host)
        np.testing.assert_array_equal(info.device_kv_indices, device)
        with self.assertRaises(ValueError):
            TransferInfo.from_zmq(msg + [device[:1].tobytes()])

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
            self.assertIsNone(manager._get_local_shm_peer("peer").error)
            self.assertIsNone(manager._get_local_shm_peer("peer").retry_at)
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
            self.assertIsNone(manager._get_local_shm_peer("peer").retry_at)

    def test_dead_exporter_released_only_after_pending_copy_settles(self):
        manager, info = self.mapping_manager()
        mapping = Mock(pending={"copy"})
        mapping.exporter_closed.return_value = True
        manager._get_local_shm_peer("peer").mapping = mapping
        manager._reap_local_shm_mappings()
        mapping.close.assert_not_called()
        mapping.pending.clear()
        manager._reap_local_shm_mappings()
        mapping.close.assert_called_once()
        self.assertIsNone(manager._get_local_shm_peer("peer").mapping)
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
        manager._local_shm_peers = {
            "live": LocalShmPeer(mapping=live),
            "dead": LocalShmPeer(mapping=dead),
        }
        with self.assertLogs("sglang.srt.disaggregation.nixl.conn", level="ERROR"):
            manager._reap_local_shm_mappings()
        live.close.assert_not_called()
        self.assertIs(manager._get_local_shm_peer("live").mapping, live)
        self.assertIs(manager._get_local_shm_peer("dead").mapping, dead)

    def test_reaper_cannot_race_submission(self):
        manager, _ = self.mapping_manager()
        mapping = Mock(pending=set())
        mapping.exporter_closed.return_value = True
        manager._get_local_shm_peer("peer").mapping = mapping
        started = threading.Event()

        def reap():
            started.set()
            manager._reap_local_shm_mappings()

        with manager._get_local_shm_peer("peer").lock:
            worker = threading.Thread(target=reap)
            worker.start()
            self.assertTrue(started.wait(2))
            mapping.pending.add("submitted")
        worker.join(2)
        self.assertFalse(worker.is_alive())
        mapping.close.assert_not_called()

    def test_initializing_peer_does_not_block_healthy_peer(self):
        manager, info = self.mapping_manager()
        healthy = SimpleNamespace(**vars(info))
        healthy.agent_name = "healthy"
        healthy_mapping = Mock()
        manager._get_local_shm_peer("healthy").mapping = healthy_mapping
        manager.decode_kv_args_table = {"healthy": healthy}
        manager.kv_args.kv_data_lens = [64]
        manager.agent = Mock()
        entered, release, completed = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )
        errors = []

        def create(*args):
            entered.set()
            if not release.wait(5):
                raise TimeoutError("test initialization gate")
            return Mock()

        def initialize():
            try:
                manager._ensure_local_shm_mapping(info)
            except Exception as exc:
                errors.append(exc)

        def send():
            try:
                manager.send_kvcache("healthy", [0], [20], [0], 0, "done")
            except Exception as exc:
                errors.append(exc)
            finally:
                completed.set()

        module = "sglang.srt.disaggregation.nixl.conn"
        with (
            patch(module + ".same_host", return_value=True),
            patch(module + ".LocalShmMapping", side_effect=create),
        ):
            initializing = threading.Thread(target=initialize)
            sending = threading.Thread(target=send)
            initializing.start()
            try:
                self.assertTrue(entered.wait(2))
                sending.start()
                self.assertTrue(
                    completed.wait(2),
                    "healthy peer blocked by unrelated initialization",
                )
                healthy_mapping.submit.assert_called_once()
            finally:
                release.set()
                initializing.join(5)
                if sending.ident is not None:
                    sending.join(5)
        self.assertFalse(errors)

    def test_cleanup_callback_failure_can_be_retried(self):
        from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost

        pool = MLATokenToKVPoolHost.__new__(MLATokenToKVPoolHost)
        pool.pin_memory = False
        buffer = object()
        pool.kv_buffer = buffer
        first, retry, last = (
            Mock(),
            Mock(side_effect=[OSError("cleanup"), None]),
            Mock(),
        )
        for callback in (first, retry, last):
            pool.register_host_resource_cleanup(callback)
        with self.assertRaises(OSError):
            pool.destroy()
        self.assertIs(pool.kv_buffer, buffer)
        self.assertFalse(getattr(pool, "_destroyed", False))
        last.assert_not_called()
        pool.destroy()
        pool.destroy()
        first.assert_called_once()
        self.assertEqual(retry.call_count, 2)
        last.assert_called_once()
        self.assertIsNone(pool.kv_buffer)

    def test_export_initialization_rolls_back_resources(self):
        with tempfile.TemporaryFile() as file:
            pool = SimpleNamespace(
                fd=file.fileno(),
                kv_buffer=Mock(),
                get_contiguous_buf_infos=lambda: ([], [], []),
            )
            pool.kv_buffer.is_contiguous.return_value = True
            pool.kv_buffer.data_ptr.return_value = 1000
            pool.kv_buffer.numel.return_value = 4096
            pool.kv_buffer.element_size.return_value = 1
            module = "sglang.srt.disaggregation.nixl.local_shm"
            duplicated = []
            real_dup = os.dup

            def duplicate_fd(fd):
                result = real_dup(fd)
                duplicated.append(result)
                return result

            for stage in ("socket", "thread"):
                with (
                    self.subTest(stage=stage),
                    patch(module + ".os.dup", side_effect=duplicate_fd),
                    patch(module + ".socket.socket") as socket_factory,
                    patch(module + ".host_identity", return_value=["host", "net"]),
                    patch(
                        module + ".threading.Thread.start",
                        side_effect=RuntimeError("thread startup"),
                    ),
                ):
                    if stage == "socket":
                        socket_factory.side_effect = OSError("socket allocation")
                    with self.assertRaises((OSError, RuntimeError)):
                        LocalShmExport(pool)
                    with self.assertRaises(OSError):
                        os.fstat(duplicated[-1])
                    if stage == "thread":
                        socket_factory.return_value.close.assert_called_once()

    def test_export_close_retries_after_join_failure(self):
        with tempfile.TemporaryFile() as file:
            export = LocalShmExport.__new__(LocalShmExport)
            export.fd = os.dup(file.fileno())
            fd = export.fd
            export.stopped = threading.Event()
            export.socket = Mock()
            export.thread = Mock()
            export.thread.join.side_effect = [RuntimeError("join"), None, None]
            with self.assertRaises(RuntimeError):
                export.close()
            os.fstat(fd)
            export.close()
            export.close()
            with self.assertRaises(OSError):
                os.fstat(fd)

    def test_submit_failure_records_completion_for_already_queued_writes(self):
        self.check_submit_failure(event_record_fails=False)

    def test_submit_event_failure_keeps_mapping_pending(self):
        self.check_submit_failure(event_record_fails=True)

    def check_submit_failure(self, *, event_record_fails):
        mapping = LocalShmMapping.__new__(LocalShmMapping)
        mapping.ptrs, mapping.lengths, mapping.items = [100], [64], [16]
        mapping.gpu_id, mapping.pending = 0, set()
        event, stream, notify = Mock(), Mock(cuda_stream=1), Mock()
        event.query.return_value = False
        if event_record_fails:
            event.record.side_effect = RuntimeError("record event")
        runtime = SimpleNamespace(
            cudaMemcpyAsync=Mock(side_effect=[(0,), (1,)]),
            cudaMemcpyKind=SimpleNamespace(cudaMemcpyDeviceToHost=2),
        )
        bindings = SimpleNamespace(runtime=runtime)
        with (
            patch.dict(
                sys.modules,
                {"cuda": SimpleNamespace(bindings=bindings), "cuda.bindings": bindings},
            ),
            patch("torch.cuda.device"),
            patch("torch.cuda.Stream", return_value=stream),
            patch("torch.cuda.Event", return_value=event),
        ):
            handle = mapping.submit([200], [64], [0, 1], [0, 1], notify)
        self.assertEqual(runtime.cudaMemcpyAsync.call_count, 2)
        event.record.assert_called_once_with(stream)
        self.assertIn(handle, mapping.pending)
        if event_record_fails:
            with self.assertRaisesRegex(RuntimeError, "could not be recorded"):
                handle.poll()
            self.assertIn(handle, mapping.pending)
        else:
            self.assertEqual(handle.poll(), "PROC")
            event.query.return_value = True
            self.assertEqual(handle.poll(), "ERR")
            self.assertFalse(mapping.pending)
        notify.assert_not_called()

    def test_empty_chunk(self):
        self.assertEqual(build_copy_plan([64], [128], [16], [], []), [])


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
