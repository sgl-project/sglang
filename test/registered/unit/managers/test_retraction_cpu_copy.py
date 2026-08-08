import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.decode import DecodePreallocQueue, DecodeRequest
from sglang.srt.disaggregation.fake.conn import FakeKVManager, FakeKVReceiver
from sglang.srt.disaggregation.utils import (
    FAKE_BOOTSTRAP_HOST,
    DisaggregationMode,
)
from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMHATokenToKVPool
from sglang.srt.managers.schedule_batch import FINISH_ABORT, Req, ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.hisparse_memory_pool import HiSparseDSATokenToKVPool
from sglang.srt.mem_cache.memory_pool import (
    HybridLinearKVPool,
    MHATokenToKVPool,
    MHATokenToKVPoolFP4,
    MHATokenToKVPoolMXFP8,
    MLATokenToKVPool,
    MLATokenToKVPoolFP4,
    NoOpMHATokenToKVPool,
    PageMajorMHATokenToKVPool,
)
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.mem_cache.unified_memory_pool import UnifiedMHATokenToKVPool
from sglang.srt.observability.req_time_stats import SchedulerReqTimeStats
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _req(rid: str):
    req = SimpleNamespace(
        rid=rid,
        origin_input_ids=[1],
        output_ids=[2],
        priority=None,
        to_finish=None,
        return_logprob=False,
        multimodal_inputs=None,
        input_embeds=None,
        positional_embed_overrides=None,
        token_type_ids=None,
        lora_id=None,
        session=None,
        session_id=None,
        bootstrap_host="127.0.0.1",
        pd_rebootstrap_forced_output_id=None,
        pd_rebootstrap_in_progress=False,
        time_stats=SimpleNamespace(
            set_retract_time=Mock(),
            set_decode_rebootstrap_prealloc_queue_entry_time=Mock(),
        ),
    )
    req.supports_pd_rebootstrap = lambda server_args: Req.supports_pd_rebootstrap(
        req, server_args
    )
    return req


def _server_args(transfer_backend: str = "mooncake"):
    return SimpleNamespace(
        disaggregation_mode="decode",
        disaggregation_transfer_backend=transfer_backend,
    )


def _batch(cpu_copy_supported: bool, req_count: int = 2):
    batch = ScheduleBatch(reqs=[_req(str(i)) for i in range(req_count)])
    batch.token_to_kv_pool_allocator = SimpleNamespace(
        supports_cpu_copy=Mock(return_value=cpu_copy_supported)
    )
    batch._get_decode_retraction_order = Mock(return_value=list(range(req_count)))
    batch.check_decode_mem = Mock(return_value=True)
    batch.release_req = Mock()
    batch.filter_batch = Mock()
    return batch


class TestKVCacheCPUCopyCapability(CustomTestCase):
    def test_mha_support_depends_on_layout(self):
        pool = object.__new__(MHATokenToKVPool)
        pool.kv_cache_layout = "nhd"
        pool.k_scale_buffer = None
        pool.v_scale_buffer = None
        self.assertTrue(pool.supports_cpu_copy())

        pool.kv_cache_layout = "hnd"
        self.assertFalse(pool.supports_cpu_copy())

        pool.kv_cache_layout = "vectorized_5d"
        self.assertFalse(pool.supports_cpu_copy())

    def test_mha_with_separate_scale_buffers_is_unsupported(self):
        pool = object.__new__(MHATokenToKVPool)
        pool.kv_cache_layout = "nhd"
        pool.k_scale_buffer = object()
        pool.v_scale_buffer = object()
        self.assertFalse(pool.supports_cpu_copy())

    def test_unsupported_mha_variants_report_false(self):
        self.assertFalse(object.__new__(PageMajorMHATokenToKVPool).supports_cpu_copy())
        self.assertFalse(object.__new__(MHATokenToKVPoolFP4).supports_cpu_copy())
        self.assertFalse(object.__new__(MHATokenToKVPoolMXFP8).supports_cpu_copy())
        self.assertFalse(object.__new__(NoOpMHATokenToKVPool).supports_cpu_copy())
        self.assertFalse(object.__new__(UnifiedMHATokenToKVPool).supports_cpu_copy())

    def test_npu_mha_defines_optional_scale_state(self):
        pool = NPUMHATokenToKVPool(
            size=4,
            page_size=1,
            dtype=torch.float16,
            head_num=1,
            head_dim=2,
            layer_num=1,
            device="cpu",
            enable_memory_saver=False,
            enable_alt_stream=False,
        )

        self.assertIsNone(pool.k_scale_buffer)
        self.assertIsNone(pool.v_scale_buffer)
        self.assertTrue(pool.supports_cpu_copy())

    def test_mla_variants_and_hisparse_report_distinct_capabilities(self):
        self.assertTrue(object.__new__(MLATokenToKVPool).supports_cpu_copy())
        self.assertFalse(object.__new__(MLATokenToKVPoolFP4).supports_cpu_copy())
        self.assertFalse(object.__new__(HiSparseDSATokenToKVPool).supports_cpu_copy())

    def test_swa_requires_both_subpools_to_support_cpu_copy(self):
        pool = object.__new__(SWAKVPool)
        pool.full_kv_pool = SimpleNamespace(supports_cpu_copy=lambda: True)
        pool.swa_kv_pool = SimpleNamespace(supports_cpu_copy=lambda: True)
        self.assertTrue(pool.supports_cpu_copy())

        pool.swa_kv_pool = SimpleNamespace(supports_cpu_copy=lambda: False)
        self.assertFalse(pool.supports_cpu_copy())

    def test_hybrid_requires_attention_and_mamba_cpu_copy(self):
        pool = object.__new__(HybridLinearKVPool)
        pool.full_kv_pool = SimpleNamespace(supports_cpu_copy=lambda: True)
        pool.mamba_pool = SimpleNamespace(supports_cpu_copy=lambda: True)
        self.assertTrue(pool.supports_cpu_copy())

        pool.mamba_pool = SimpleNamespace(supports_cpu_copy=lambda: False)
        self.assertFalse(pool.supports_cpu_copy())


class TestPDDecodeRetractionCPUCopy(CustomTestCase):
    def test_rebootstrap_requires_token_only_base_request(self):
        req = _req("eligibility")
        args = _server_args()
        self.assertTrue(req.supports_pd_rebootstrap(args))

        unsupported_fields = {
            "multimodal_inputs": object(),
            "input_embeds": [[0.0]],
            "positional_embed_overrides": object(),
            "token_type_ids": [0],
            "lora_id": "adapter",
            "session": object(),
            "session_id": "session",
        }
        for field, value in unsupported_fields.items():
            with self.subTest(field=field):
                setattr(req, field, value)
                self.assertFalse(req.supports_pd_rebootstrap(args))
                setattr(req, field, None)

    @patch(
        "sglang.srt.managers.schedule_batch."
        "NewTokenRatioTracker.estimate_new_token_ratio_after_retract",
        return_value=0.5,
    )
    def test_fake_transfer_cannot_rebootstrap(self, _estimate):
        """Fake transfer has no prefill-recompute dispatch implementation."""
        cases = (
            ("fake", None),
            ("fake", "127.0.0.1"),
            ("mooncake", FAKE_BOOTSTRAP_HOST),
        )
        for transfer_backend, bootstrap_host in cases:
            with self.subTest(
                transfer_backend=transfer_backend,
                bootstrap_host=bootstrap_host,
            ):
                batch = _batch(cpu_copy_supported=False)
                batch.reqs[1].bootstrap_host = bootstrap_host
                args = _server_args(transfer_backend)

                retracted, rebootstrap, _, aborted = batch.retract_decode(args)

                self.assertEqual(retracted, [])
                self.assertEqual(rebootstrap, [])
                self.assertEqual([req.rid for req in aborted], ["1"])
                self.assertIn(
                    "cannot be safely recomputed", aborted[0].to_finish.message
                )

    @patch(
        "sglang.srt.managers.schedule_batch."
        "NewTokenRatioTracker.estimate_new_token_ratio_after_retract",
        return_value=0.5,
    )
    def test_supported_pool_retracts_and_resumes(self, _estimate):
        batch = _batch(cpu_copy_supported=True)
        args = _server_args()

        retracted, rebootstrap, ratio, aborted = batch.retract_decode(args)

        self.assertEqual([req.rid for req in retracted], ["1"])
        self.assertEqual(rebootstrap, [])
        self.assertEqual(aborted, [])
        self.assertEqual(ratio, 0.5)
        batch.release_req.assert_called_once_with(1, 1, args, offload_kv=True)

    @patch(
        "sglang.srt.managers.schedule_batch."
        "NewTokenRatioTracker.estimate_new_token_ratio_after_retract",
        return_value=0.5,
    )
    def test_unsupported_pool_rebootstraps_retracted_request(self, _estimate):
        batch = _batch(cpu_copy_supported=False)
        args = _server_args()

        retracted, rebootstrap, _, aborted = batch.retract_decode(args)

        self.assertEqual(retracted, [])
        self.assertEqual([req.rid for req in rebootstrap], ["1"])
        self.assertEqual(aborted, [])
        batch.release_req.assert_called_once_with(1, 1, args, offload_kv=False)

    @patch(
        "sglang.srt.managers.schedule_batch."
        "NewTokenRatioTracker.estimate_new_token_ratio_after_retract",
        return_value=0.5,
    )
    def test_incomplete_round_trip_pools_rebootstrap_without_offload(self, _estimate):
        vectorized = object.__new__(MHATokenToKVPool)
        vectorized.kv_cache_layout = "vectorized_5d"
        vectorized.k_scale_buffer = None
        vectorized.v_scale_buffer = None

        for pool in (
            vectorized,
            object.__new__(MHATokenToKVPoolFP4),
            object.__new__(MLATokenToKVPoolFP4),
        ):
            with self.subTest(pool=type(pool).__name__):
                batch = _batch(cpu_copy_supported=pool.supports_cpu_copy())
                args = _server_args()

                retracted, rebootstrap, _, aborted = batch.retract_decode(args)

                self.assertEqual(retracted, [])
                self.assertEqual([req.rid for req in rebootstrap], ["1"])
                self.assertEqual(aborted, [])
                batch.release_req.assert_called_once_with(1, 1, args, offload_kv=False)

    @patch(
        "sglang.srt.managers.schedule_batch."
        "NewTokenRatioTracker.estimate_new_token_ratio_after_retract",
        return_value=0.5,
    )
    def test_request_that_cannot_rebootstrap_is_aborted(self, _estimate):
        batch = _batch(cpu_copy_supported=False)
        batch.reqs[1].multimodal_inputs = object()
        args = _server_args()

        retracted, rebootstrap, _, aborted = batch.retract_decode(args)

        self.assertEqual(retracted, [])
        self.assertEqual(rebootstrap, [])
        self.assertEqual([req.rid for req in aborted], ["1"])
        self.assertIsInstance(aborted[0].to_finish, FINISH_ABORT)
        self.assertIn("cannot be safely recomputed", aborted[0].to_finish.message)
        self.assertEqual(aborted[0].to_finish.err_type, "InternalServerError")
        batch.release_req.assert_called_once_with(1, 1, args, offload_kv=False)

    @patch(
        "sglang.srt.managers.schedule_batch."
        "NewTokenRatioTracker.estimate_new_token_ratio_after_retract",
        return_value=0.5,
    )
    def test_last_request_abort_never_wastes_cpu_copy(self, _estimate):
        batch = _batch(cpu_copy_supported=True, req_count=1)
        batch.check_decode_mem = Mock(return_value=False)
        args = _server_args()

        retracted, rebootstrap, _, aborted = batch.retract_decode(args)

        self.assertEqual(retracted, [])
        self.assertEqual(rebootstrap, [])
        self.assertEqual([req.rid for req in aborted], ["0"])
        self.assertEqual(aborted[0].to_finish.err_type, "InternalServerError")
        batch.release_req.assert_called_once_with(0, 0, args, offload_kv=False)


class TestSchedulerRetractionAbort(CustomTestCase):
    def test_memory_pressure_routes_rebootstrap_result_to_prefill(self):
        req = _req("rebootstrap")
        batch = Mock()
        batch.batch_size.return_value = 1
        batch.is_empty.return_value = False
        batch.check_decode_mem.return_value = False
        batch.retract_decode.return_value = ([], [req], 0.5, [])

        scheduler = object.__new__(Scheduler)
        scheduler.forward_ct = 1
        scheduler.server_args = _server_args()
        scheduler.token_to_kv_pool_allocator = Mock()
        scheduler.token_to_kv_pool_allocator.available_size.side_effect = [0, 1]
        scheduler.tree_cache = SimpleNamespace(req_to_token_pool=SimpleNamespace())
        scheduler.new_token_ratio_tracker = SimpleNamespace(current=0.8)
        scheduler.metrics_reporter = SimpleNamespace(
            num_retracted_reqs=0,
            enable_metrics=False,
        )
        scheduler._enqueue_pd_rebootstrap = Mock()

        result = scheduler.update_running_batch(batch)

        self.assertIs(result, batch)
        scheduler._enqueue_pd_rebootstrap.assert_called_once_with(req)
        self.assertEqual(scheduler.metrics_reporter.num_retracted_reqs, 1)
        batch.prepare_for_decode.assert_called_once_with()

    def test_rebootstrap_replays_boundary_and_enters_prealloc_queue(self):
        req = _req("rebootstrap")
        req.output_ids = [2, 3]
        scheduler = object.__new__(Scheduler)
        scheduler.disagg_decode_prealloc_queue = Mock()

        scheduler._enqueue_pd_rebootstrap(req)

        self.assertEqual(req.output_ids, [2])
        self.assertEqual(req.pd_rebootstrap_forced_output_id, 3)
        self.assertTrue(req.pd_rebootstrap_in_progress)
        req.time_stats.set_retract_time.assert_called_once_with()
        scheduler.disagg_decode_prealloc_queue.add.assert_called_once_with(
            req, is_rebootstrap=True
        )

    def test_rebootstrap_restarts_timing_at_concrete_prealloc_admission(self):
        """A second admission must not reuse the first bootstrap timestamps."""
        req = _req("timing")
        receiver = Mock()
        decode_req = DecodeRequest(req=req, kv_receiver=receiver)
        queue = object.__new__(DecodePreallocQueue)
        queue.scheduler = SimpleNamespace(server_args=_server_args())
        queue.retracted_queue = []
        queue.pending_reqs = []
        queue._check_if_req_exceed_kv_capacity = Mock(return_value=False)
        queue._create_receiver_and_enqueue = Mock(return_value=decode_req)
        queue._resolve_prefill_dp_rank = Mock(return_value=0)

        queue.add(req, is_rebootstrap=True)

        req.time_stats.set_decode_rebootstrap_prealloc_queue_entry_time.assert_called_once_with()
        receiver.init.assert_called_once_with(0)

    def test_fake_rebootstrap_is_stopped_at_concrete_dispatch_boundary(self):
        """A fake receiver must be aborted before recompute dispatch."""
        req = _req("fake-dispatch")
        req.bootstrap_host = None
        fake_manager = object.__new__(FakeKVManager)
        receiver = FakeKVReceiver(
            mgr=fake_manager,
            bootstrap_addr="2.2.2.2:0",
            bootstrap_room=1,
        )
        decode_req = DecodeRequest(
            req=req,
            kv_receiver=receiver,
            is_rebootstrap=True,
        )
        queue = object.__new__(DecodePreallocQueue)
        queue.scheduler = SimpleNamespace(server_args=_server_args("fake"))
        queue.kv_manager = fake_manager

        queue._submit_prefill_recompute(decode_req)

        self.assertEqual(receiver.conclude_state, KVPoll.Failed)

    def test_abort_finalizes_request_and_async_offload_state(self):
        req = _req("abort")
        abort_reason = FINISH_ABORT(
            "KV cache cannot be restored",
            status_code=500,
        )
        req.to_finish = abort_reason
        req.time_stats = SimpleNamespace(
            trace_ctx=SimpleNamespace(abort=Mock()),
            set_completion_time=Mock(),
        )

        scheduler = object.__new__(Scheduler)
        scheduler.decode_offload_manager = Mock()
        scheduler.ipc_channels = SimpleNamespace(
            send_to_tokenizer=SimpleNamespace(send_output=Mock())
        )

        scheduler._finalize_retraction_abort(req)

        self.assertIsNone(req.to_finish)
        self.assertIs(req.finished_reason, abort_reason)
        req.time_stats.trace_ctx.abort.assert_called_once_with(abort_info=abort_reason)
        req.time_stats.set_completion_time.assert_called_once_with()
        scheduler.decode_offload_manager.finalize_release_on_finish.assert_called_once_with(
            req
        )
        send_output = scheduler.ipc_channels.send_to_tokenizer.send_output
        send_output.assert_called_once()
        abort_req, sent_req = send_output.call_args.args
        self.assertIs(sent_req, req)
        self.assertEqual(abort_req.finished_reason, req.finished_reason.to_json())


class TestPDRebootstrapTiming(CustomTestCase):
    def test_second_bootstrap_uses_second_admission_interval(self):
        """Bootstrap/allocator metrics must describe the rebootstrap cycle."""
        collector = Mock()
        stats = SchedulerReqTimeStats(disagg_mode=DisaggregationMode.DECODE)
        stats.set_metrics_collector(collector)
        stats.decode_prealloc_queue_entry_time = 10.0
        stats.decode_transfer_queue_entry_time = 12.0
        stats.bootstrap_done_time = 11.0

        stats.set_decode_rebootstrap_prealloc_queue_entry_time(ts=20.0)
        self.assertEqual(stats.decode_transfer_queue_entry_time, 0.0)
        self.assertEqual(stats.bootstrap_done_time, 0.0)

        stats.set_bootstrap_done_time(ts=21.0)
        stats.set_decode_transfer_queue_entry_time(ts=22.0)

        self.assertEqual(stats.decode_prealloc_queue_entry_time, 20.0)
        self.assertEqual(stats.bootstrap_done_time, 21.0)
        collector.observe_kv_transfer_bootstrap.assert_called_once_with(
            bootstrap_ms=1000.0,
            alloc_ms=1000.0,
        )


if __name__ == "__main__":
    unittest.main()
