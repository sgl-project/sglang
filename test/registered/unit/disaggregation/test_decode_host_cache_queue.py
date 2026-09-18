import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from sglang.srt.arg_groups.overrides import resolved_view
from sglang.srt.arg_groups.pd_disaggregation_hook import handle_pd_disaggregation
from sglang.srt.disaggregation.base.conn import KVPoll, KVTransferDestination
from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.srt.disaggregation.decode import (
    DecodePreallocQueue,
    DecodeRequest,
    DecodeTransferQueue,
    SchedulerDisaggregationDecodeMixin,
)
from sglang.srt.disaggregation.utils import (
    FAKE_BOOTSTRAP_HOST,
    DisaggregationMode,
    ReqToMetadataIdxAllocator,
)
from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.managers.schedule_batch import FINISH_ABORT, Req
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

MODULE = "sglang.srt.disaggregation.decode"


class TestDecodeHostCacheQueue(unittest.TestCase):
    def setUp(self):
        override = get_context().override_server_args(
            disaggregation_decode_enable_host_cache=True,
            disaggregation_decode_enable_radix_cache=False,
        )
        override.install()
        self.addCleanup(override.restore)
        for target, value in (
            ("torch.distributed.get_world_size", 1),
            ("torch.distributed.all_reduce", None),
            (f"{MODULE}.envs.SGLANG_TEST_DISAGG_FAILURE_PROB.get", 0),
        ):
            patcher = patch(target, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)

        self.req = Req(
            rid="host-request",
            origin_input_text="",
            origin_input_ids=array("q", range(5)),
            sampling_params=SamplingParams(max_new_tokens=4),
            bootstrap_room=42,
        )
        self.req.set_extend_range(0, 5)
        self.receiver = Mock(supports_host_destination=True)
        self.receiver.poll.return_value = KVPoll.WaitingForInput
        self.decode_req = DecodeRequest(self.req, self.receiver)
        self.host_cache = Mock()
        self.host_cache.allocate.return_value = torch.arange(20, 28)
        self.allocator = Mock(page_size=4)
        self.allocator.available_size.return_value = 0
        self.req_pool = Mock(size=4)
        self.req_pool.available_size.return_value = 1
        self.metadata_allocator = ReqToMetadataIdxAllocator(4)
        self.scheduler = SimpleNamespace(
            running_batch=SimpleNamespace(reqs=[]),
            waiting_queue=[],
            last_batch=None,
            server_args=SimpleNamespace(disaggregation_decode_enable_radix_cache=False),
            enable_priority_scheduling=False,
            enable_lora=False,
            enable_hisparse=False,
            enable_decode_hicache=False,
            decode_host_cache=self.host_cache,
            output_streamer=Mock(),
            metrics_reporter=SimpleNamespace(enable_metrics=False),
        )

        self.prealloc = DecodePreallocQueue.__new__(DecodePreallocQueue)
        self.prealloc.pp_size = 1
        self.prealloc.gloo_group = object()
        self.prealloc.scheduler = self.scheduler
        self.prealloc.queue = [self.decode_req]
        self.prealloc.pending_reqs = []
        self.prealloc.retracted_queue = []
        self.prealloc._prefill_dp_rank_queries = {}
        self.prealloc._num_published_destinations = 0
        self.prealloc.num_reserved_decode_tokens = 2
        self.prealloc.token_to_kv_pool = object()
        self.prealloc.token_to_kv_pool_allocator = self.allocator
        self.prealloc.req_to_token_pool = self.req_pool
        self.prealloc.req_to_metadata_buffer_idx_allocator = self.metadata_allocator

        self.transfer = DecodeTransferQueue.__new__(DecodeTransferQueue)
        self.transfer.queue = []
        self.transfer.scheduler = self.scheduler
        self.transfer.host_cache = self.host_cache
        self.transfer.tp_rank = 0
        self.transfer.gloo_group = self.prealloc.gloo_group
        self.transfer.enable_deferred_kv_release = False
        self.transfer.deferred_kv_release_timeout = 0
        self.transfer._deferred_releases = []
        self.transfer.enable_staging = False
        self.transfer.tree_cache = object()
        self.transfer.metadata_buffers = SimpleNamespace(
            bootstrap_room=torch.zeros((4, 1), dtype=torch.int64)
        )
        self.transfer.req_to_metadata_buffer_idx_allocator = self.metadata_allocator
        self.prealloc.transfer_queue = self.transfer
        self.scheduler.disagg_decode_prealloc_queue = self.prealloc
        self.scheduler.disagg_decode_transfer_queue = self.transfer

    def admit(self):
        with patch.object(self.prealloc, "_pre_alloc") as device_alloc:
            admitted, failed = self.prealloc.pop_preallocated()
        device_alloc.assert_not_called()
        self.assertEqual(admitted, [self.decode_req])
        self.assertEqual(failed, [])
        self.assertEqual(self.prealloc.queue, [])
        self.assertIsNone(self.req.kv.req_pool_idx)
        self.transfer.extend(admitted)
        return admitted

    def test_pressure_selects_host_without_device_reservations(self):
        self.admit()
        self.assertTrue(self.decode_req.host_staged)
        self.host_cache.allocate.assert_called_once_with(self.req, 5)
        args, kwargs = self.receiver.send_metadata.call_args
        np.testing.assert_array_equal(args[0], [5, 6])
        self.assertEqual(args[0].dtype, np.int32)
        self.assertEqual(kwargs["destination"], KVTransferDestination.HOST)
        self.assertEqual(kwargs["decode_prefix_len"], 0)
        self.assertEqual(self.metadata_allocator.available_size(), 3)
        self.assertFalse(self.prealloc.has_published_destinations)
        self.assertEqual(self.prealloc.num_tokens_pre_allocated, 0)
        self.assertEqual(self.prealloc._active_req_count(), 0)
        self.assertEqual(self.prealloc._active_reserved_tokens(), 0)

        direct = DecodeRequest(self.req, Mock())
        self.transfer.add(direct)
        self.assertEqual(self.prealloc.num_tokens_pre_allocated, 5)
        self.assertEqual(self.prealloc._active_req_count(), 1)
        self.assertEqual(self.prealloc._active_reserved_tokens(), 2)

    def test_request_slot_pressure_also_uses_host(self):
        self.req_pool.available_size.return_value = 0
        self.allocator.available_size.return_value = 64
        self.admit()
        self.assertTrue(self.decode_req.host_staged)
        self.req_pool.alloc.assert_not_called()

    def test_force_host_with_available_device_budget(self):
        self.allocator.available_size.return_value = 64
        with envs.SGLANG_TEST_DISAGG_FORCE_HOST_TRANSFER.override(True):
            self.admit()
        self.assertTrue(self.decode_req.host_staged)
        self.assertEqual(
            self.receiver.send_metadata.call_args.kwargs["destination"],
            KVTransferDestination.HOST,
        )

    def test_force_host_full_pool_waits_without_device_fallback(self):
        self.allocator.available_size.return_value = 64
        self.host_cache.allocate.return_value = None
        with (
            envs.SGLANG_TEST_DISAGG_FORCE_HOST_TRANSFER.override(True),
            patch.object(self.prealloc, "_pre_alloc") as device_alloc,
        ):
            self.assertEqual(self.prealloc.pop_preallocated(), ([], []))
        device_alloc.assert_not_called()
        self.receiver.send_metadata.assert_not_called()
        self.assertEqual(self.prealloc.queue, [self.decode_req])
        self.assertEqual(self.metadata_allocator.available_size(), 4)

    def test_force_host_bypasses_fake_and_rejects_incompatible_peer(self):
        self.allocator.available_size.return_value = 64
        self.receiver.supports_host_destination = False
        with (
            envs.SGLANG_TEST_DISAGG_FORCE_HOST_TRANSFER.override(True),
            patch.object(self.prealloc, "_pre_alloc") as device_alloc,
        ):
            with self.assertRaisesRegex(ValueError, "compatible prefill"):
                self.prealloc.pop_preallocated()
        device_alloc.assert_not_called()
        self.host_cache.allocate.assert_not_called()
        self.receiver.send_metadata.assert_not_called()
        self.assertEqual(self.metadata_allocator.available_size(), 4)

        self.req.bootstrap_host = FAKE_BOOTSTRAP_HOST
        with (
            envs.SGLANG_TEST_DISAGG_FORCE_HOST_TRANSFER.override(True),
            patch.object(
                self.prealloc,
                "_pre_alloc",
                side_effect=RuntimeError("device allocation reached"),
            ) as device_alloc,
        ):
            with self.assertRaisesRegex(RuntimeError, "device allocation reached"):
                self.prealloc.pop_preallocated()
        device_alloc.assert_called_once()
        self.host_cache.allocate.assert_not_called()

        self.allocator.available_size.return_value = 0
        with (
            envs.SGLANG_TEST_DISAGG_FORCE_HOST_TRANSFER.override(True),
            patch.object(self.prealloc, "_pre_alloc") as device_alloc,
        ):
            self.assertEqual(self.prealloc.pop_preallocated(), ([], []))
        device_alloc.assert_not_called()
        self.host_cache.allocate.assert_not_called()
        self.receiver.send_metadata.assert_not_called()
        self.assertEqual(self.prealloc.queue, [self.decode_req])
        self.assertEqual(self.metadata_allocator.available_size(), 4)

    def test_unavailable_host_or_old_peer_stays_in_preallocation(self):
        for supports_host in (False, True):
            with self.subTest(supports_host=supports_host):
                self.receiver.supports_host_destination = supports_host
                self.host_cache.allocate.return_value = None
                admitted, failed = self.prealloc.pop_preallocated()
                self.assertEqual((admitted, failed), ([], []))
                self.assertEqual(self.prealloc.queue, [self.decode_req])
                self.assertFalse(self.decode_req.host_staged)
                self.assertEqual(self.metadata_allocator.available_size(), 4)
        self.receiver.send_metadata.assert_not_called()

    def test_completed_transfer_waits_for_device_budget_before_committing(self):
        self.admit()
        with (
            patch.object(
                self.transfer, "_poll_with_metadata_gate", return_value=[KVPoll.Success]
            ),
            patch.object(self.transfer, "_commit_transfer_to_req") as commit,
            patch.object(self.prealloc, "_pre_alloc") as device_alloc,
        ):
            self.assertEqual(self.transfer.pop_transferred(), [])
            device_alloc.assert_not_called()
            commit.assert_not_called()
            self.assertEqual(self.transfer.queue, [self.decode_req])
            self.assertEqual(self.metadata_allocator.available_size(), 3)
            self.host_cache.release.assert_not_called()

            self.allocator.available_size.return_value = 32
            self.assertEqual(self.transfer.pop_transferred(), [self.req])
            device_alloc.assert_called_once_with(self.req)
            commit.assert_called_once_with(self.decode_req)
            self.assertFalse(self.decode_req.host_staged)
            self.assertEqual(self.transfer.queue, [])
            self.assertEqual(self.metadata_allocator.available_size(), 4)
            self.host_cache.release.assert_not_called()

    def test_abort_during_receive_keeps_host_pages_until_transfer_completion(self):
        self.admit()
        self.scheduler.chunked_req = None
        self.scheduler.mm_receiver = None
        self.scheduler.dllm_config = None
        self.scheduler.grammar_manager = Mock()
        self.scheduler.disaggregation_mode = DisaggregationMode.DECODE
        self.scheduler.collect_inflight_reqs = Mock(return_value=set())
        Scheduler.abort_request(self.scheduler, AbortReq(rid=self.req.rid))
        self.assertIsInstance(self.req.finished_reason, FINISH_ABORT)
        self.receiver.abort.assert_not_called()
        with (
            patch.object(self.transfer, "_poll_with_metadata_gate") as poll,
            patch.object(self.prealloc, "_pre_alloc") as device_alloc,
            patch(f"{MODULE}.release_kv_cache") as device_release,
        ):
            poll.return_value = [KVPoll.Transferring]
            self.assertEqual(self.transfer.pop_transferred(), [])
            self.host_cache.release.assert_not_called()
            self.receiver.clear.assert_not_called()
            self.assertEqual(self.metadata_allocator.available_size(), 3)

            poll.return_value = [KVPoll.Success]
            self.assertEqual(self.transfer.pop_transferred(), [])
            self.host_cache.release.assert_called_once_with(self.req)
            self.receiver.clear.assert_called_once_with()
            device_alloc.assert_not_called()
            device_release.assert_not_called()
            self.assertEqual(self.transfer.queue, [])
            self.assertEqual(self.metadata_allocator.available_size(), 4)

    def test_prebuilt_fences_previous_forward_before_loading_and_processing(self):
        order = []
        batch = Mock(reqs=[self.req])
        batch.prepare_for_prebuilt.side_effect = lambda: order.append("prepare")
        batch.process_prebuilt.side_effect = lambda _: order.append("process")
        self.host_cache.load.side_effect = lambda *_: order.append("load")
        self.scheduler.waiting_queue = [self.req]
        self.scheduler.grammar_manager = Mock()
        self.scheduler.grammar_manager.has_waiting_grammars.return_value = False
        self.scheduler.running_batch.batch_size = lambda: 0
        self.scheduler.req_to_token_pool = self.req_pool
        self.scheduler.token_to_kv_pool_allocator = self.allocator
        self.scheduler.max_running_requests = 4
        self.scheduler.tree_cache = object()
        self.scheduler.model_config = object()
        self.scheduler.enable_overlap = True
        self.scheduler.spec_algorithm = object()
        self.scheduler.future_map = object()
        self.scheduler.forward_stream = object()
        self.scheduler.schedule_stream = Mock()
        self.scheduler.schedule_stream.wait_stream.side_effect = lambda _: order.append(
            "fence"
        )
        with (
            patch.object(self.req, "init_next_round_input"),
            patch(f"{MODULE}.ScheduleBatch.init_new", return_value=batch),
        ):
            result = SchedulerDisaggregationDecodeMixin._get_new_prebuilt_batch(
                self.scheduler, self.scheduler.running_batch
            )
        self.assertIs(result, batch)
        self.assertEqual(order, ["prepare", "fence", "load", "process"])
        self.host_cache.load.assert_called_once_with([self.req], self.req_pool)

    def test_failure_waits_for_drain_and_rank_consensus_past_timeout(self):
        self.admit()
        manager = object.__new__(CommonKVManager)
        manager._deferred_abort_ack_tracker = {}
        self.receiver.kv_mgr = manager
        self.receiver.bootstrap_infos = [{"rank": 0}, {"rank": 1}]
        self.receiver.abort.side_effect = lambda: manager.register_deferred_abort_room(
            42
        )
        with (
            patch.object(
                self.transfer, "_poll_with_metadata_gate", return_value=[KVPoll.Failed]
            ),
            patch(f"{MODULE}.release_kv_cache") as device_release,
        ):
            self.assertEqual(self.transfer.pop_transferred(), [])
            self.receiver.abort.assert_called_once_with()
            self.assertEqual(self.transfer.queue, [])
            self.assertEqual(len(self.transfer._deferred_releases), 1)
            self.transfer.resolve_deferred_releases()
            manager.note_abort_ack(42, 0)
            self.transfer.resolve_deferred_releases()
            self.host_cache.release.assert_not_called()
            self.receiver.clear.assert_not_called()
            self.assertEqual(self.metadata_allocator.available_size(), 3)

            manager.note_abort_ack(42, 1)
            with (
                patch("torch.distributed.get_world_size", return_value=2),
                patch(
                    "torch.distributed.all_reduce",
                    side_effect=lambda count, **_: count.zero_(),
                ),
            ):
                self.transfer.resolve_deferred_releases()
            self.host_cache.release.assert_not_called()
            self.transfer.resolve_deferred_releases()
            self.host_cache.release.assert_called_once_with(self.req)
            self.receiver.clear.assert_called_once_with()
            device_release.assert_not_called()
            self.assertEqual(self.transfer._deferred_releases, [])
            self.assertEqual(self.metadata_allocator.available_size(), 4)


class TestDecodeHostCacheConfig(unittest.TestCase):
    def test_force_host_requires_enabled_feature(self):
        with envs.SGLANG_TEST_DISAGG_FORCE_HOST_TRANSFER.override(True):
            with self.assertRaisesRegex(ValueError, "requires decode host cache"):
                handle_pd_disaggregation(
                    ServerArgs(model_path="dummy", disaggregation_mode="decode")
                )

    def test_default_disabled_and_supported_decode(self):
        for mode, enabled in (("null", False), ("prefill", False), ("decode", True)):
            with self.subTest(mode=mode, enabled=enabled):
                args = ServerArgs(
                    model_path="dummy",
                    disaggregation_mode=mode,
                    disaggregation_decode_enable_host_cache=enabled,
                )
                handle_pd_disaggregation(args)
                if enabled:
                    config = resolved_view(args)
                    self.assertEqual(config.hicache_mem_layout, "layer_first")
                    self.assertEqual(
                        config.disaggregation_decode_retraction_backup, "host_pool"
                    )

    def test_unsupported_configuration(self):
        for changes in (
            {"disaggregation_mode": "prefill"},
            {"pp_size": 2},
            {"dcp_size": 2},
            {"speculative_algorithm": "EAGLE"},
            {"enable_hisparse": True},
            {"enable_hierarchical_cache": True},
            {"hicache_storage_backend": "mooncake"},
            {"disaggregation_decode_enable_radix_cache": True},
            {"disaggregation_enable_kv_checksum": True},
            {"enable_lora": True},
            {"disaggregation_decode_enable_offload_kvcache": True},
            {"disaggregation_decode_retraction_backup": "cpu_tensor"},
            {"hicache_io_backend": "direct"},
            {"enable_priority_scheduling": True, "disable_priority_preemption": False},
            {"enable_pd_role_switch": True},
        ):
            with self.subTest(changes=changes):
                kwargs = {
                    "disaggregation_mode": "decode",
                    "disaggregation_decode_enable_host_cache": True,
                    "hicache_mem_layout": "layer_first",
                    "hicache_io_backend": "kernel",
                    **changes,
                }
                with self.assertRaises(ValueError):
                    handle_pd_disaggregation(ServerArgs(model_path="dummy", **kwargs))
        with patch.object(envs.SGLANG_DISAGG_STAGING_BUFFER, "get", return_value=True):
            with self.assertRaisesRegex(ValueError, "staging"):
                handle_pd_disaggregation(
                    ServerArgs(
                        model_path="dummy",
                        disaggregation_mode="decode",
                        disaggregation_decode_enable_host_cache=True,
                    )
                )


if __name__ == "__main__":
    unittest.main()
