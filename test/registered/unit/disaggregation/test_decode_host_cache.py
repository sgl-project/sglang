import argparse
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
from sglang.srt.disaggregation.decode_host_cache import DecodeHostCache
from sglang.srt.disaggregation.utils import (
    FAKE_BOOTSTRAP_HOST,
    DisaggregationMode,
    ReqToMetadataIdxAllocator,
)
from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.managers.schedule_batch import FINISH_ABORT, Req
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.kv_cache_builder import resolve_decode_retraction_backup
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, MLATokenToKVPool
from sglang.srt.runtime_context import get_context, get_memory
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

MODULE = "sglang.srt.disaggregation.decode"


class TestDecodeHostCache(unittest.TestCase):
    def setUp(self):
        self.device_pool = object.__new__(MHATokenToKVPool)
        self.device_pool.kv_cache_layout = "nhd"
        self.device_pool.head_dim = self.device_pool.v_head_dim = 8
        self.device_pool.layer_shard_enabled = False
        self.device_pool.device = "cpu"
        self.device_pool.layer_num = 2
        self.host_pool = Mock(
            device_pool=self.device_pool,
            page_size=4,
            layout="layer_first",
            logical_size=24,
        )
        self.host_pool.available_size.return_value = 24
        self.event = Mock()
        self.event.query.return_value = False
        self.engine = Mock(io_backend="kernel")
        self.engine.submit_host_to_device.return_value.finish_event = self.event
        self.manager = DecodeHostCache(
            self.device_pool, 4, self.host_pool, 8, self.engine
        )

    @staticmethod
    def req(index=0):
        return Mock(rid="reused-rid", kv=SimpleNamespace(req_pool_idx=index))

    def test_page_ownership_preserves_shared_retraction_reservation(self):
        first, second = self.req(), self.req()
        indices = torch.arange(8)
        self.host_pool.available_size.return_value = 15
        self.assertIsNone(self.manager.allocate(first, 5))
        self.host_pool.alloc.assert_not_called()
        self.host_pool.available_size.return_value = 16
        self.host_pool.alloc.side_effect = [indices, None]
        self.assertIs(self.manager.allocate(first, 5), indices)
        self.assertIsNone(self.manager.allocate(second, 5))
        self.assertEqual(self.host_pool.alloc.call_args.args, (8,))
        self.assertTrue(self.manager.contains(first))
        self.assertFalse(self.manager.contains(second))
        with self.assertRaisesRegex(ValueError, "already owns"):
            self.manager.allocate(first, 1)
        self.manager.release(second)
        self.manager.clear()
        self.manager.release(first)
        self.host_pool.free.assert_called_once_with(indices)
        self.host_pool.destroy.assert_not_called()

    def test_batched_copy_and_abort_release_follow_completion_and_rank_consensus(self):
        waiting, first, second, direct = [self.req(i) for i in range(4)]
        self.host_pool.alloc.side_effect = [
            torch.arange(4),
            torch.arange(4, 8),
            torch.arange(8, 16),
        ]
        for req, tokens in ((waiting, 1), (first, 3), (second, 5)):
            self.manager.allocate(req, tokens)
        req_pool = SimpleNamespace(
            req_to_token=torch.arange(20, dtype=torch.int32).reshape(4, 5)
        )
        self.assertIs(self.manager.load([first, second, direct], req_pool), self.event)
        transfer = self.engine.submit_host_to_device.call_args.args[0][0]
        self.assertEqual(transfer.device_indices.dtype, torch.int64)
        self.assertTrue(
            torch.equal(
                transfer.host_indices, torch.tensor([4, 5, 6, 8, 9, 10, 11, 12])
            )
        )
        self.assertTrue(
            torch.equal(
                transfer.device_indices, torch.tensor([5, 6, 7, 10, 11, 12, 13, 14])
            )
        )
        self.event.wait.assert_called_once_with()
        self.event.synchronize.assert_not_called()
        self.assertIsNone(self.manager.load([first, second], req_pool))
        self.manager.poll()
        self.host_pool.free.assert_not_called()
        self.event.query.return_value = True
        with (
            patch("torch.distributed.get_world_size", return_value=2),
            patch(
                "torch.distributed.all_reduce",
                side_effect=lambda count, **_: count.fill_(1),
            ),
        ):
            self.manager.poll(object())
        self.assertFalse(self.manager.contains(first))
        self.assertTrue(self.manager.contains(waiting))
        self.assertTrue(self.manager.contains(second))
        order = []
        self.event.synchronize.side_effect = lambda: order.append("copy finished")
        self.host_pool.free.side_effect = lambda _: order.append("pages freed")
        self.manager.release(second)
        self.manager.poll()
        self.assertEqual(order, ["copy finished", "pages freed"])
        self.assertEqual(self.host_pool.free.call_count, 2)

    def test_registration_and_unsupported_pool_geometry(self):
        buffers = [torch.zeros((8, 2, 8), dtype=torch.float16) for _ in range(4)]
        self.host_pool.host_kv_data_refs = buffers
        self.host_pool.token_stride_size = 32
        self.assertEqual(
            self.manager.get_contiguous_buf_infos(),
            (
                [buffer.data_ptr() for buffer in buffers],
                [buffer.nbytes for buffer in buffers],
                [128] * 4,
            ),
        )
        for field, value in (
            ("kv_cache_layout", "hnd"),
            ("v_head_dim", 16),
            ("layer_shard_enabled", True),
        ):
            with (
                self.subTest(field=field),
                patch.object(self.device_pool, field, value),
            ):
                with self.assertRaises(ValueError):
                    DecodeHostCache(self.device_pool, 4, self.host_pool, 8, self.engine)
        self.host_pool.logical_size = 8
        with self.assertRaisesRegex(ValueError, "increase --hicache-size"):
            DecodeHostCache(self.device_pool, 4, self.host_pool, 8, self.engine)
        mla = object.__new__(MLATokenToKVPool)
        mla.use_dsa = True
        with self.assertRaisesRegex(ValueError, "plain MLA"):
            DecodeHostCache(mla, 4, self.host_pool, 8, self.engine)

    def test_host_sizing_reserves_receive_capacity_and_preserves_explicit_size(self):
        worker = Mock()
        worker.get_memory_pool.return_value = (
            SimpleNamespace(max_context_len=81),
            Mock(
                get_kvcache=Mock(return_value=SimpleNamespace(size=256, page_size=16))
            ),
        )
        for enabled, ratio, size, expected in (
            (False, None, 0, 80 / 256),
            (True, None, 0, 160 / 256),
            (True, 0.3, 0, 0.3),
            (True, None, 1.25, None),
        ):
            with (
                self.subTest(enabled=enabled, ratio=ratio, size=size),
                get_context().override_server_args(
                    disaggregation_mode="decode",
                    disaggregation_decode_retraction_backup="host_pool",
                    disaggregation_decode_enable_host_receive=enabled,
                    enable_hierarchical_cache=False,
                    hicache_ratio=ratio,
                    hicache_size=size,
                ),
            ):
                self.assertEqual(
                    resolve_decode_retraction_backup(tp_worker=worker), "host_pool"
                )
                self.assertEqual(get_memory().hicache_size, size)
                if expected is not None:
                    self.assertEqual(get_memory().hicache_ratio, expected)


class TestDecodeHostCacheQueue(unittest.TestCase):
    def setUp(self):
        override = get_context().override_server_args(
            disaggregation_decode_enable_host_receive=True,
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
        self.req_pool = Mock(size=4, mamba_allocator=None)
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

    def test_pressure_stages_without_device_accounting_then_waits_for_allocation(self):
        self.admit()
        indices = self.receiver.send_metadata.call_args.args[0]
        np.testing.assert_array_equal(indices, [5, 6])
        self.assertEqual(
            self.receiver.send_metadata.call_args.kwargs["destination"],
            KVTransferDestination.HOST,
        )
        self.assertEqual(self.prealloc.num_tokens_pre_allocated, 0)
        self.assertEqual(self.prealloc._active_req_count(), 0)
        self.assertEqual(self.prealloc._active_reserved_tokens(), 0)
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

    def test_forced_host_waits_for_capacity_with_available_device_budget(self):
        self.allocator.available_size.return_value = 64
        self.host_cache.allocate.side_effect = [None, torch.arange(20, 28)]
        with envs.SGLANG_TEST_DISAGG_FORCE_HOST_TRANSFER.override(True):
            with patch.object(self.prealloc, "_pre_alloc") as device_alloc:
                self.assertEqual(self.prealloc.pop_preallocated(), ([], []))
            device_alloc.assert_not_called()
            self.receiver.send_metadata.assert_not_called()
            self.assertEqual(self.metadata_allocator.available_size(), 4)
            self.admit()
        self.assertTrue(self.decode_req.host_staged)

    def test_incompatible_peers_wait_or_fail_when_forced_and_fake_uses_device(self):
        self.receiver.supports_host_destination = False
        self.assertEqual(self.prealloc.pop_preallocated(), ([], []))
        self.allocator.available_size.return_value = 64
        with envs.SGLANG_TEST_DISAGG_FORCE_HOST_TRANSFER.override(True):
            with self.assertRaisesRegex(ValueError, "compatible prefill"):
                self.prealloc.pop_preallocated()
            self.req.bootstrap_host = FAKE_BOOTSTRAP_HOST
            with patch.object(
                self.prealloc,
                "_pre_alloc",
                side_effect=RuntimeError("device allocation reached"),
            ) as device_alloc:
                with self.assertRaisesRegex(RuntimeError, "device allocation reached"):
                    self.prealloc.pop_preallocated()
            device_alloc.assert_called_once()
            self.allocator.available_size.return_value = 0
            self.assertEqual(self.prealloc.pop_preallocated(), ([], []))
        self.host_cache.allocate.assert_not_called()
        self.receiver.send_metadata.assert_not_called()
        self.assertEqual(self.metadata_allocator.available_size(), 4)

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
        self.scheduler.chunked_req = None
        self.scheduler.ngram_embedding_manager = Mock()
        self.scheduler.ngram_embedding_manager.prepare_for_forward.side_effect = (
            lambda *_args, **_kwargs: order.append("ngram")
        )
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
        self.assertEqual(order, ["prepare", "fence", "load", "ngram", "process"])
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
    def test_default_layout_and_forced_host_requires_enabled_feature(self):
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        base_args = ["--model-path", "dummy", "--disaggregation-mode", "decode"]
        self.assertFalse(
            parser.parse_args(base_args).disaggregation_decode_enable_host_receive
        )
        self.assertTrue(
            parser.parse_args(
                base_args + ["--disaggregation-decode-enable-host-receive"]
            ).disaggregation_decode_enable_host_receive
        )
        args = ServerArgs(
            model_path="dummy",
            disaggregation_mode="decode",
            disaggregation_decode_enable_host_receive=True,
        )
        handle_pd_disaggregation(args)
        config = resolved_view(args)
        self.assertEqual(config.hicache_mem_layout, "layer_first")
        self.assertEqual(config.disaggregation_decode_retraction_backup, "host_pool")
        with envs.SGLANG_TEST_DISAGG_FORCE_HOST_TRANSFER.override(True):
            with self.assertRaisesRegex(
                ValueError, "requires --disaggregation-decode-enable-host-receive"
            ):
                handle_pd_disaggregation(
                    ServerArgs(model_path="dummy", disaggregation_mode="decode")
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
                    "disaggregation_decode_enable_host_receive": True,
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
                        disaggregation_decode_enable_host_receive=True,
                    )
                )


if __name__ == "__main__":
    unittest.main()
