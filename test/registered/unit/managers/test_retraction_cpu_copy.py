import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMHATokenToKVPool
from sglang.srt.managers.schedule_batch import FINISH_ABORT, ScheduleBatch
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
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _req(rid: str):
    return SimpleNamespace(
        rid=rid,
        origin_input_ids=[1],
        output_ids=[2],
        priority=None,
        to_finish=None,
        return_logprob=False,
    )


def _server_args():
    return SimpleNamespace(disaggregation_mode="decode")


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
    @patch(
        "sglang.srt.managers.schedule_batch."
        "NewTokenRatioTracker.estimate_new_token_ratio_after_retract",
        return_value=0.5,
    )
    def test_supported_pool_retracts_and_resumes(self, _estimate):
        batch = _batch(cpu_copy_supported=True)
        args = _server_args()

        retracted, ratio, aborted = batch.retract_decode(args)

        self.assertEqual([req.rid for req in retracted], ["1"])
        self.assertEqual(aborted, [])
        self.assertEqual(ratio, 0.5)
        batch.release_req.assert_called_once_with(1, 1, args, offload_kv=True)

    @patch(
        "sglang.srt.managers.schedule_batch."
        "NewTokenRatioTracker.estimate_new_token_ratio_after_retract",
        return_value=0.5,
    )
    def test_unsupported_pool_aborts_only_retracted_request(self, _estimate):
        batch = _batch(cpu_copy_supported=False)
        args = _server_args()

        retracted, _, aborted = batch.retract_decode(args)

        self.assertEqual(retracted, [])
        self.assertEqual([req.rid for req in aborted], ["1"])
        self.assertIsInstance(aborted[0].to_finish, FINISH_ABORT)
        self.assertIn(
            "does not support synchronous CPU save and restore",
            aborted[0].to_finish.message,
        )
        self.assertEqual(aborted[0].to_finish.err_type, "InternalServerError")
        batch.release_req.assert_called_once_with(1, 1, args, offload_kv=False)

    @patch(
        "sglang.srt.managers.schedule_batch."
        "NewTokenRatioTracker.estimate_new_token_ratio_after_retract",
        return_value=0.5,
    )
    def test_incomplete_round_trip_pools_abort_without_offload(self, _estimate):
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

                retracted, _, aborted = batch.retract_decode(args)

                self.assertEqual(retracted, [])
                self.assertEqual([req.rid for req in aborted], ["1"])
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

        retracted, _, aborted = batch.retract_decode(args)

        self.assertEqual(retracted, [])
        self.assertEqual([req.rid for req in aborted], ["0"])
        self.assertEqual(aborted[0].to_finish.err_type, "InternalServerError")
        batch.release_req.assert_called_once_with(0, 0, args, offload_kv=False)


class TestSchedulerRetractionAbort(CustomTestCase):
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


if __name__ == "__main__":
    unittest.main()
